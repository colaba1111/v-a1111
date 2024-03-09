sudo apt install build-essential aria2 -y
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0 -y

# A1111
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui

# Install Aspect Ratio Helper
git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git ~/stable-diffusion-webui/extensions/sd-webui-aspect-ratio-helper

# Install Civitai Browser
git clone https://github.com/BlafKing/sd-civitai-browser-plus.git ~/stable-diffusion-webui/extensions/sd-civitai-browser-plus

# Install Image Browser
git clone https://github.com/zanllp/sd-webui-infinite-image-browsing.git ~/stable-diffusion-webui/extensions/sd-webui-infinite-image-browsing

# Install Controlnet
git clone https://github.com/Mikubill/sd-webui-controlnet ~/stable-diffusion-webui/extensions/sd-webui-controlnet

# Install Reactor
git clone https://github.com/Gourieff/sd-webui-reactor ~/stable-diffusion-webui/extensions/sd-webui-reactor
pip install insightface==0.7.3

# Install CLIP Interrogator
git clone https://github.com/pharmapsychotic/clip-interrogator-ext.git ~/stable-diffusion-webui/extensions/clip-interrogator-ext

# Install Upscaler
git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git ~/stable-diffusion-webui/extensions/ultimate-upscale-for-automatic1111

# Install adetailer
git clone https://github.com/Bing-su/adetailer.git ~/stable-diffusion-webui/extensions/adetailer

# Download controlnet
# Newer models for faceid
# aria2c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-full-face_sd15.safetensors
# aria2c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-plus-face_sd15.safetensors
# aria2c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-plus_sd15.safetensors
# aria2c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter_sd15.safetensors

# https://huggingface.co/h94/IP-Adapter-FaceID
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid-plus_sd15.bin
# ip-adapter-faceid-plus_sd15_lora.safetensors
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid-plusv2_sd15.bin
# ip-adapter-faceid-plusv2_sd15_lora.safetensors
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid-plusv2_sdxl.bin
# ip-adapter-faceid-plusv2_sdxl_lora.safetensors
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid_sd15.bin
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid_sdxl.bin
aria2c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter-faceid-portrait_sd15.bin

# https://huggingface.co/InstantX/InstantID
aria2c https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter_instant_id_sdxl.bin
aria2c https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o ip-adapter_instant_id_sdxl.safetensors

aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_canny_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_lineart_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_mlsd_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_openpose_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_scribble_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_seg_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_softedge_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15s2_lineart_anime_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11p_sd15_softedge.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o t2iadapter_color_sd14v1.pth
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o control_v11f1e_sd15_tile_fp16.safetensors
aria2c https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth -d ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models -o t2iadapter_color_sd14v1.pth
