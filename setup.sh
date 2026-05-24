# install python 3.12 before

uv pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt