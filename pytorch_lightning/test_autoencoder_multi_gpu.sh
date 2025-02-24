export CUDA_LAUNCH_BLOCKING=1

python launch.py --config ./configs/shape-autoencoder/Dora-VAE-test.yaml \
                          --test --gpu 0,1,2,3,4,5,6,7