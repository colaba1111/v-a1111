#!/bin/bash

cd /workspace
env TF_CPP_MIN_LOG_LEVEL=1
apt -y update -qq

wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb


apt -y install cmake
pip install lit

apt -y install ffmpeg libsm6 libxext6
apt -y install -qq libunwind8-dev
dpkg -i *.deb
!apt -y install -qq libcairo2-dev pkg-config python3-dev
env LD_PRELOAD=libtcmalloc.so
rm *.deb

apt -y install -qq aria2
pip install -q torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U
pip install -q xformers==0.0.16 triton==2.0.0 -U

git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui
# git clone https://huggingface.co/embed/negative /workspace/stable-diffusion-webui/embeddings/negative
# git clone https://huggingface.co/embed/lora /workspace/stable-diffusion-webui/models/Lora/positive
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /workspace/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth
wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O /workspace/stable-diffusion-webui/scripts/run_n_times.py
# git clone -b v2.1 https://github.com/camenduru/deforum-for-automatic1111-webui /workspace/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui
git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui-images-browser /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser
# git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui-huggingface /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface
git clone -b v2.1 https://github.com/camenduru/sd-civitai-browser /workspace/stable-diffusion-webui/extensions/sd-civitai-browser
git clone -b v2.1 https://github.com/camenduru/sd-webui-additional-networks /workspace/stable-diffusion-webui/extensions/sd-webui-additional-networks
git clone -b v2.1 https://github.com/camenduru/sd-webui-tunnels /workspace/stable-diffusion-webui/extensions/sd-webui-tunnels
git clone -b v2.1 https://github.com/camenduru/batchlinks-webui /workspace/stable-diffusion-webui/extensions/batchlinks-webui
git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui-catppuccin /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin
git clone -b v2.1 https://github.com/camenduru/a1111-sd-webui-locon /workspace/stable-diffusion-webui/extensions/a1111-sd-webui-locon
git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui-rembg /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg
git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui-two-shot /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot
git clone -b v2.1 https://github.com/camenduru/sd_webui_stealth_pnginfo /workspace/stable-diffusion-webui/extensions/sd_webui_stealth_pnginfo

git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git /workspace/stable-diffusion-webui/extensions/ultimate-upscale-for-automatic1111
git clone https://github.com/Extraltodeus/depthmap2mask.git /workspace/stable-diffusion-webui/extensions/depthmap2mask
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen.git /workspace/stable-diffusion-webui/extensions/stable-diffusion-webui-promptgen
git clone https://github.com/yankooliveira/sd-webui-photopea-embed.git /workspace/stable-diffusion-webui/extensions/sd-webui-photopea-embed

cd /workspace/stable-diffusion-webui
git reset --hard
git -C /workspace/stable-diffusion-webui/repositories/stable-diffusion-stability-ai reset --hard

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/29461 -d /workspace/stable-diffusion-webui/models/Stable-diffusion -o realisticVisionV20_v20NoVAE-inpainting.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d /workspace/stable-diffusion-webui/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors

sed -i -e '''/    prepare_environment()/a\    os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' /workspace/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py""")''' /workspace/stable-diffusion-webui/launch.py
sed -i -e 's/\"sd_model_checkpoint\"\,/\"sd_model_checkpoint\,sd_vae\,CLIP_stop_at_last_layers\"\,/g' /workspace/stable-diffusion-webui/modules/shared.py

cd /workspace/stable-diffusion-webui && python launch.py --listen --xformers --enable-insecure-extension-access --theme dark --gradio-queue --multiple