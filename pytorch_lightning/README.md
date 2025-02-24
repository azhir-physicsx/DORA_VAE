## Environment Setup
This code has been tested on `Ubuntu 20.04.6 LTS` and requires the following:

* Python 3.10 and PyTorch 2.4.0
* Anaconda (conda3) or Miniconda3
* CUDA-capable GPU
```shell
sudo apt-get update &&  sudo apt-get install -y ninja-build libxi6
conda create -n Dora python=3.10 -y
conda activate Dora
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install diso 
pip install -r requirements.txt 
```

## Download the checkpoint
```shell
python download.py
```

## Inference
```shell
bash test_autoencoder_single_gpu # single gpu
# bash test_autoencoder_multi_gpu # multi gpu
```
## Training
```shell
bash train_autoencoder_single_node # single node
# bash test_autoencoder_multi_gpu # multi nodes
```

## Folder structure (to be completed)
```
pytorch_lightning/
---- ckpt/
---- configs/
---- data/
--------- dataset1/
------------------ test.json
------------------ train.json
------------------ val.json
```


