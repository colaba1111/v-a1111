#!/bin/bash

mkdir /content
cd /content
env TF_CPP_MIN_LOG_LEVEL=1
apt -y update -qq

wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb

apt -y install ffmpeg libsm6 libxext6
apt -y install -qq libunwind8-dev
dpkg -i *.deb
env LD_PRELOAD=libtcmalloc.so
rm *.deb
apt -y install -qq aria2

pip install -q torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U
pip install -q xformers==0.0.16 triton==2.0.0 -U

git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui
git clone https://huggingface.co/embed/negative /content/stable-diffusion-webui/embeddings/negative
git clone https://huggingface.co/embed/lora /content/stable-diffusion-webui/models/Lora/positive
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /content/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth
wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O /content/stable-diffusion-webui/scripts/run_n_times.py
git clone -b v1.9 https://github.com/camenduru/deforum-for-automatic1111-webui /content/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui
git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui-images-browser /content/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser
git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui-huggingface /content/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface
git clone -b v1.9 https://github.com/camenduru/sd-civitai-browser /content/stable-diffusion-webui/extensions/sd-civitai-browser
git clone -b v1.9 https://github.com/camenduru/sd-webui-additional-networks /content/stable-diffusion-webui/extensions/sd-webui-additional-networks
git clone -b v1.9 https://github.com/camenduru/sd-webui-tunnels /content/stable-diffusion-webui/extensions/sd-webui-tunnels
git clone -b v1.9 https://github.com/camenduru/batchlinks-webui /content/stable-diffusion-webui/extensions/batchlinks-webui
git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui-catppuccin /content/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin
git clone -b v1.9 https://github.com/camenduru/a1111-sd-webui-locon /content/stable-diffusion-webui/extensins/a1111-sd-webui-locon
git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui-rembg /content/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg
git clone -b v1.9 https://github.com/camenduru/stable-diffusion-webui-two-shot /content/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot
git clone -b v1.9 https://github.com/camenduru/sd_webui_stealth_pnginfo /content/stable-diffusion-webui/extensions/sd_webui_stealth_pnginfo

cd /content/stable-diffusion-webui
git reset --hard
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/29461 -d /content/stable-diffusion-webui/models/Stable-diffusion -o realisticVisionV20_v13-inpainting.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/15670 -d /content/stable-diffusion-webui/models/Stable-diffusion -o URPMv1.3.inpainting.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d /content/stable-diffusion-webui/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors

sed -i -e '''/    prepare_environment()/a\    os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' /content/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py""")''' /content/stable-diffusion-webui/launch.py
sed -i -e 's/fastapi==0.90.1/fastapi==0.89.1/g' /content/stable-diffusion-webui/requirements_versions.txt

mkdir /content/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui/models
sed -i -e 's/\"sd_model_checkpoint\"\,/\"sd_model_checkpoint\,sd_vae\,CLIP_stop_at_last_layers\"\,/g' /content/stable-diffusion-webui/modules/shared.py

