export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python launch.py --config ./configs/shape-autoencoder/Dora-VAE-test.yaml \
                          --test --gpu 0