import argparse
import pickle
import sys

import numpy as np
import torch
import trimesh
import rembg
import skimage.measure
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, scale_tensor


def run_tsr(args):
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.tsr_chunk_size)
    model.to(args.device)

    if args.no_remove_background:
        rembg_session = None
        image = np.array(Image.open(args.input).convert("RGB"))
    else:
        rembg_session = rembg.new_session()
        image = remove_background(Image.open(args.input), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

    with torch.no_grad():
        scene_codes = model([image], device=args.device)
        assert len(scene_codes) == 1

    return {
        "model": model,
        "scene_codes": scene_codes,
    }


def run_grid(args, tsr_result):
    total_grid_resolution = args.marching_resolution * args.marching_oversampling
    sampling_range = (0.0, 1.0)
    x, y, z = (
        torch.linspace(*sampling_range, total_grid_resolution),
        torch.linspace(*sampling_range, total_grid_resolution),
        torch.linspace(*sampling_range, total_grid_resolution),
    )
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    verts = torch.cat(
        [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
    ).reshape(-1, 3)

    result = {}
    with torch.no_grad():
        queried_grid = tsr_result["model"].renderer.query_triplane(
            tsr_result["model"].decoder,
            scale_tensor(
                verts.to(
                    args.device,
                ),
                sampling_range,
                (
                    -tsr_result["model"].renderer.cfg.radius,
                    tsr_result["model"].renderer.cfg.radius,
                ),
            ),
            tsr_result["scene_codes"][0],
        )

        for k in queried_grid:
            result[k] = queried_grid[k].reshape(
                total_grid_resolution,
                total_grid_resolution,
                total_grid_resolution,
                -1,
            )

    return result


def run_mesh(args, tsr_result, grid_result):
    total_grid_resolution = args.marching_resolution * args.marching_oversampling
    max_pooled = skimage.measure.block_reduce(
        grid_result["density_act"]
        .cpu()
        .numpy()
        .reshape(
            total_grid_resolution,
            total_grid_resolution,
            total_grid_resolution,
        ),
        (
            args.marching_oversampling,
            args.marching_oversampling,
            args.marching_oversampling,
        ),
        np.max,
    )
    solid = max_pooled > args.density_threshold
    encoding = trimesh.voxel.base.DenseEncoding(solid)
    voxels = trimesh.voxel.VoxelGrid(encoding).hollow().fill()
    points = np.argwhere(voxels.encoding.data == True)
    mesh = trimesh.voxel.ops.points_to_marching_cubes(points)
    mesh.apply_translation(
        [
            -args.marching_resolution * 0.5,
            -args.marching_resolution * 0.5,
            -args.marching_resolution * 0.5,
        ]
    )
    mesh.apply_scale(
        2.0 * tsr_result["model"].renderer.cfg.radius / args.marching_resolution
    )
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to input image",
        required=True,
    )
    parser.add_argument(
        "--output-mesh",
        help="Path to output mesh (.obj)",
        required=True,
    )
    parser.add_argument(
        "--output-scene-codes", help="Path to output scene codes (.pkl)", required=True
    )
    parser.add_argument(
        "--no-remove-background",
        help="Skip the background removal step",
        action="store_true",
    )
    parser.add_argument(
        "--foreground-ratio",
        help="Ratio of foreground size to image size (ignored if --no-remove-background is specified)",
        required="--no-remove-background" not in sys.argv,
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--density-threshold",
        help="Minimum density of NeRF sample that is considered solid (i.e. part of the mesh surface)",
        required=True,
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--marching-resolution",
        help="Resolution of voxel grid used for marching cubes",
        required=True,
        type=int,
        default=128,
    )
    parser.add_argument(
        "--marching-oversampling",
        help="Number of NeRF samples for every marching cubes voxel (1 = no oversampling)",
        required=True,
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
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

    print("(1/3): TripoSR Image to NeRF")
    tsr_result = run_tsr(args)
    print("(2/3): Sample NeRF to uniform grid")
    grid_result = run_grid(args, tsr_result)
    print("(3/3): Marching cubes")
    mesh_result = run_mesh(args, tsr_result, grid_result)

    print("Writing scene codes to {}".format(args.output_scene_codes))
    with open(args.output_scene_codes, "wb") as outfile:
        pickle.dump(tsr_result["scene_codes"], outfile)
    print("Writing mesh to {}".format(args.output_mesh))
    mesh_result.export(args.output_mesh)


if __name__ == "__main__":
    main()
