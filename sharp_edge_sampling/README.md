This code has been tested on `Ubuntu 20.04.6 LTS` and requires the following:

* Python 3.10 and PyTorch 2.4.0
* Anaconda (conda3) or Miniconda3
* A CUDA-capable GPU
```shell
sudo apt-get update &&  sudo apt-get install -y ninja-build libxi6
conda create -n Dora python=3.10 -y
conda activate Dora
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/ashawkey/cubvh
pip install diso 
pip install -r requirements.txt
```

## Overview

The data processing workflow consists of two steps. 

Step 1 is to convert non-watertight models into watertight ones normalized to (-1, 1). We provide an alternative solution for step 1, and its effect won't differ much from what we actually used. If your model is already watertight, you can directly proceed to the second step. Note that the .obj files in Dora-bench have already been converted into watertight models normalized to (-1, 1).

Step 2 is to perform sharp edge sampling on the watertight models. 

## Step 1: (GPU) Convert non-watertight models into watertight ones
```shell
python detect_path.py   --directory_to_search ./Objaverse \
                        --json_file_path ./mesh_path.json  \
                        --file_type .glb
```
Use the following comment for Dora-VAE v1.1
```shell
python to_watertight_mesh.py  --resolution 256 \
                              --json_file_path ./mesh_path.json \
                              --remesh_target_path ./remesh
```
Use the following comment for Dora-VAE v1.2 or for finetuning Dora-VAE v1.1.
```shell
python to_watertight_mesh.py  --resolution 512 \
                              --json_file_path ./mesh_path.json \
                              --remesh_target_path ./remesh
```

## Step 2: (CPU-Only) Perform sharp edge sampling on the watertight models
If your 3D asset is made watertight via your own approach and you've skipped our Step 1, ensure the vertex coordinates of your OBJ file are normalized to (-1, 1). Values either below or above this range may reduce VAE reconstruction accuracy.
```shell
python detect_path.py   --directory_to_search ./remesh \
                        --json_file_path ./watertight_path.json \
                        --file_type .obj
python sharp_sample.py  --json_file_path ./watertight_path.json  \
                        --point_number 65536 \
                        --angle_threshold 15 \
                        --sharp_point_path ./sharp_point_ply \
                        --sample_path ./sample
```
Our VAE model, trained solely on watertight data, may underperform on non-watertight data. Two reasons: non-manifold edges may lack two faces, nullifying the sharp-edge detection via dihedral angles and preventing salient-point sampling; the VAE encoder needs points and normals, but non-watertight data normals often have problems like flipping. If you want to improve the reconstruction performance of Dora-VAE for non-watertight data, you can make appropriate modifications to the algorithm based on the above analysis and then perform fine-tuning using the VAE training code we provided.
## Acknowledgement

- [cubvh](https://github.com/ashawkey/cubvh) provides a fast implementation to compute udf.
- [pysdf](https://github.com/sxyu/sdf) provides a fast implementation to compute sdf.
- [fpsample](https://github.com/leonardodalinky/fpsample) provides a fast implementation to compute fps
- [diso](https://github.com/SarahWeiii/diso) provides a fast implementation to extract iso-surface.
- [vscode-mesh-viewer](https://github.com/ashawkey/vscode-mesh-viewer) provides a plugin for previewing meshes in VSCode.