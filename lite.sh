#!/bin/bash

cd /workspace
env TF_CPP_MIN_LOG_LEVEL=1
apt -y update -qq

sudo apt install build-essential cmake -y -qq

wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /workspace/libtcmalloc_minimal.so.4
env LD_PRELOAD=/workspace/libtcmalloc_minimal.so.4

pip install lit

apt -y install ffmpeg libsm6 libxext6
apt -y install -qq libunwind8-dev

apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev
pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U
pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U

git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui
git clone https://huggingface.co/embed/negative /workspace/stable-diffusion-webui/embeddings/negative
git clone https://huggingface.co/embed/lora /workspace/stable-diffusion-webui/models/Lora/positive
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /workspace/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth
wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O /workspace/stable-diffusion-webui/scripts/run_n_times.py
# git clone -b v2.4 https://github.com/camenduru/deforum-for-automatic1111-webui /workspace/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui
git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui-images-browser /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser
git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui-huggingface /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface
git clone -b v2.4 https://github.com/camenduru/sd-civitai-browser /workspace/stable-diffusion-webui/extensions/sd-civitai-browser
git clone -b v2.4 https://github.com/camenduru/sd-webui-additional-networks /workspace/stable-diffusion-webui/extensions/sd-webui-additional-networks
git clone -b v2.4 https://github.com/camenduru/sd-webui-tunnels /workspace/stable-diffusion-webui/extensions/sd-webui-tunnels
git clone -b v2.4 https://github.com/camenduru/batchlinks-webui /workspace/stable-diffusion-webui/extensions/batchlinks-webui
git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui-catppuccin /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin
git clone -b v2.4 https://github.com/camenduru/a1111-sd-webui-locon /workspace/stable-diffusion-webui/extensions/a1111-sd-webui-locon
git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui-rembg /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg
git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui-two-shot /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot
git clone -b v2.4 https://github.com/camenduru/sd-webui-aspect-ratio-helper /workspace/stable-diffusion-webui/extensions/sd-webui-aspect-ratio-helper
git clone -b v2.4 https://github.com/camenduru/asymmetric-tiling-sd-webui /workspace/stable-diffusion-webui/extensions/asymmetric-tiling-sd-webui

git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git /workspace/stable-diffusion-webui/extensions/ultimate-upscale-for-automatic1111
git clone https://github.com/Extraltodeus/depthmap2mask.git /workspace/stable-diffusion-webui/extensions/depthmap2mask
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen.git /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-promptgen
git clone https://github.com/yankooliveira/sd-webui-photopea-embed.git /workspace/stable-diffusion-webui/extensions/sd-webui-photopea-embed

cd /workspace/stable-diffusion-webui
git reset --hard
git -C /workspace/stable-diffusion-webui/repositories/stable-diffusion-stability-ai reset --hard

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=full&fp=fp16 -d /workspace/stable-diffusion-webui/models/Stable-diffusion -o realisticVisionV51_v51VAE.safetensors

sed -i -e '''/from modules import launch_utils/a\import os''' /workspace/stable-diffusion-webui/launch.py
sed -i -e '''/    prepare_environment()/a\    os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' /workspace/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py""")''' /workspace/stable-diffusion-webui/launch.py
sed -i -e 's/\"sd_model_checkpoint\"\,/\"sd_model_checkpoint\,sd_vae\,CLIP_stop_at_last_layers\"\,/g' /workspace/stable-diffusion-webui/modules/shared.py

cd /workspace/stable-diffusion-webui && python launch.py --listen --xformers --enable-insecure-extension-access --theme dark --gradio-queue --multiple --no-download-sd-model	