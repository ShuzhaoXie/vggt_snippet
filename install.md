```
conda create -n v python=3.10
conda activate v
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install trimesh Pillow matplotlib huggingface_hub einops safetensors opencv-python numpy==1.26.1
cd submodules/vggt
pip install -e .
cd submodules/vggt-omega
pip install -e .
```