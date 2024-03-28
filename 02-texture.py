import argparse
import pickle

import numpy as np
import torch
import xatlas
import trimesh
import moderngl
from PIL import Image

from tsr.system import TSR


def run_tsr(args):
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.tsr_chunk_size)
    model.to(args.device)

    with open(args.input_scene_codes, "rb") as infile:
        scene_codes = pickle.load(infile)

    return {
        "model": model,
        "scene_codes": scene_codes,
    }


def run_mesh(args):
    return trimesh.load_mesh(args.input_mesh)


def run_xatlas(args, mesh_result):
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh_result.vertices, mesh_result.faces)
    options = xatlas.PackOptions()
    options.resolution = args.texture_resolution
    options.padding = args.texture_padding
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": vmapping,
        "indices": indices,
        "uvs": uvs,
    }


def run_rasterize(args, mesh_result, xatlas_result):
    ctx = moderngl.create_context(standalone=True)
    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 v_pos;
            void main() {
                v_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(v_pos, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 vg_pos;
            void main() {
                vg_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        geometry_shader="""
            #version 330
            uniform float u_resolution;
            uniform float u_dilation;
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 12) out;
            in vec3 vg_pos[];
            out vec3 vf_pos;
            void lineSegment(int aidx, int bidx) {
                vec2 a = gl_in[aidx].gl_Position.xy;
                vec2 b = gl_in[bidx].gl_Position.xy;
                vec3 aCol = vg_pos[aidx];
                vec3 bCol = vg_pos[bidx];

                vec2 dir = normalize((b - a) * u_resolution);
                vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                gl_Position = vec4(a + offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(a - offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(b + offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
                gl_Position = vec4(b - offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
            }
            void main() {
                lineSegment(0, 1);
                lineSegment(1, 2);
                lineSegment(2, 0);
                EndPrimitive();
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 vf_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(vf_pos, 1.0);
            }
        """,
    )
    uvs = xatlas_result["uvs"].flatten().astype("f4")
    pos = mesh_result.vertices[xatlas_result["vmapping"]].flatten().astype("f4")
    indices = xatlas_result["indices"].flatten().astype("i4")
    vbo_uvs = ctx.buffer(uvs)
    vbo_pos = ctx.buffer(pos)
    ibo = ctx.buffer(indices)
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    fbo = ctx.framebuffer(
        color_attachments=[
            ctx.texture(
                (args.texture_resolution, args.texture_resolution), 4, dtype="f4"
            )
        ]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gs_prog["u_resolution"].value = args.texture_resolution
    gs_prog["u_dilation"].value = args.texture_padding
    gs_vao.render()
    basic_vao.render()

    fbo_bytes = fbo.color_attachments[0].read()
    fbo_np = np.frombuffer(fbo_bytes, dtype="f4").reshape(
        args.texture_resolution, args.texture_resolution, 4
    )
    return fbo_np


def run_bake(args, tsr_result, rasterize_result):
    positions = torch.tensor(rasterize_result.reshape(-1, 4)[:, :-1])
    with torch.no_grad():
        queried_grid = tsr_result["model"].renderer.query_triplane(
            tsr_result["model"].decoder,
            positions,
            tsr_result["scene_codes"][0],
        )
    rgb_f = queried_grid["color"].numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, rasterize_result.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(args.texture_resolution, args.texture_resolution, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mesh",
        help="Path to input mesh",
        required=True,
    )
    parser.add_argument(
        "--input-scene-codes",
        help="Path to input scene codes",
        required=True,
    )
    parser.add_argument(
        "--output-mesh",
        help="Path to output mesh (.obj)",
        required=True,
    )
    parser.add_argument(
        "--output-texture",
        help="Path to output texture (.png)",
        required=True,
    )
    parser.add_argument(
        "--texture-resolution",
        help="Resolution of output texture",
        required=True,
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--texture-padding",
        help="Extra padding on edges of UV islands, to prevent filtering artifacts on edges of triangles",
        required=True,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tsr-chunk-size",
        help="Evaluation chunk size for surface extraction and rendering (smaller chunk size reduces VRAM usage but increases computation time), 0 for no chunking",
        required=True,
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--device",
        help="PyTorch device (CUDA, MPS or ROCm is preferred, CPU is default)",
        required=True,
        type=str,
        default="cpu",
    )
    args = parser.parse_args()

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.tsr_chunk_size)
    model.to(args.device)

    print("(1/5): Load TripoSR and scene codes")
    tsr_result = run_tsr(args)
    print("(2/5): Load mesh")
    mesh_result = run_mesh(args)
    print("(3/5): Generate UVs")
    xatlas_result = run_xatlas(args, mesh_result)
    print("(4/5): Rasterize UV atlas")
    rasterize_result = run_rasterize(args, mesh_result, xatlas_result)
    print("(5/5): Sample NeRF to UV atlas")
    bake_result = run_bake(args, tsr_result, rasterize_result)

    print("Writing atlased mesh to {}".format(args.output_mesh))
    xatlas.export(
        args.output_mesh,
        mesh_result.vertices[xatlas_result["vmapping"]],
        xatlas_result["indices"],
        xatlas_result["uvs"],
        mesh_result.vertex_normals[xatlas_result["vmapping"]],
    )
    print("Writing texture to {}".format(args.output_texture))
    bake_img = Image.fromarray(
        (
            bake_result.reshape(args.texture_resolution, args.texture_resolution, 4)
            * 255.0
        ).astype(np.uint8)
    ).transpose(Image.FLIP_TOP_BOTTOM)
    bake_img.save(args.output_texture)


if __name__ == "__main__":
    main()
