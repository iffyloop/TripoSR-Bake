# TripoSR Bake

[TripoSR](https://github.com/VAST-AI-Research/TripoSR) is an excellent open-source model for inferring 3D shape and texture data from 2D meshes. Like many recent machine learning-based models for generative 3D graphics, TripoSR uses a NeRF (neural radiance field) volumetric representation for its 3D data, as opposed to a traditional polygon mesh. The TripoSR repository includes code for converting its generated NeRF to a mesh using the Marching Cubes algorithm, but stores color data in vertex colors instead of textures, severely limiting resolution and preventing further manipulation of geometry (e.g. mesh simplification) without destroying color data.

TripoSR Bake splits the process of exporting a mesh into two parts. In `01-mesh.py`, an approximation of the mesh is extracted using Marching Cubes and written to `.obj` (as in the original TripoSR repo), however it _also_ saves a `.pkl` file containing the NeRF prediction from the TripoSR model inference. The `.obj` file can then be edited manually (e.g. smoothed or simplified), as long as the bounds of the mesh are preserved (i.e. do not scale/rotate/translation the mesh as a whole). After mesh editing, the `02-texture.py` script is used to generate a UV atlas for the mesh and bake the NeRF data onto this UV map. **Using this method, even a low-poly mesh may contain the high-resolution color detail of the original NeRF output.** Keep in mind that texture quality will decrease as mesh vertices are deformed farther away from the original volume of the mesh. As usual, results will vary depending on the input image, the quality of the generated mesh, the filters/edits applied before texturing the mesh, and the efficiency of the packed UV atlas.

## Installation

```sh
git clone https://github.com/iffyloop/TripoSR-Bake.git
cd TripoSR-Bake

# TripoSR requires a real virtualenv environment,
# not the built-in venv that comes with Python
pip install virtualenv
python -m virtualenv venv

# Now we install dependencies for this repo
source venv/bin/activate # venv\Scripts\activate.bat on Windows
pip install torch # For CUDA support, follow the instructions at https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Quick Start

For a full explanation of available options, please run `python 01-mesh.py --help` and `python 02-texture.py --help`.

```sh
# Generate Marching Cubes mesh
python 01-mesh.py --input input/chair.png --output-mesh output/chair.obj --output-scene-codes output/chair.pkl --no-remove-background --density-threshold 15.0 --marching-resolution 512 --marching-oversampling 1 --tsr-chunk-size 8192 --device cpu
# Bake texture
# Before this stage, you probably want to smooth and remesh the generated chair.obj mesh in MeshLab,
# then use that as the --input-mesh below instead of the raw Marching Cubes mesh.
# Editing the mesh is merely a suggestion, not a requirement. The original mesh can also be textured.
python 02-texture.py --input-mesh output/chair.obj --input-scene-codes output/chair.pkl --output-mesh output/chair-textured.obj --output-texture output/chair-textured.png --texture-resolution 1024 --texture-padding 2 --tsr-chunk-size 8192 --device cpu
```

## License

- TripoSR Bake is public domain software (Unlicense)
- TripoSR is MIT licensed, Copyright (c) 2024 Tripo AI & Stability AI
