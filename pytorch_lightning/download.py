from huggingface_hub import snapshot_download

snapshot_download(repo_id="Seed3D/Dora-VAE-1.1",local_dir_use_symlinks=False, local_dir ="./ckpts/Dora-VAE-1.1")

