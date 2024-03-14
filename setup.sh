# Common Paths
STABLE_DIFFUSION_DIR=~/stable-diffusion-webui
EXTENSION_DIR=$STABLE_DIFFUSION_DIR/extensions
CONTROLNET_MODELS_DIR=$STABLE_DIFFUSION_DIR/extensions/sd-webui-controlnet/models
DIFFUSION_MODELS_DIR=$STABLE_DIFFUSION_DIR/models/Stable-diffusion/
LORA_MODELS_DIR=$STABLE_DIFFUSION_DIR/models/Lora
EMBEDDING_MODELS_DIR=$STABLE_DIFFUSION_DIR/models/Lora

# Dependencies
sudo apt install build-essential aria2 -y
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0 -y
pip install -U "huggingface_hub[cli]"

# A1111
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui $STABLE_DIFFUSION_DIR

# Install Aspect Ratio Helper
git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git $EXTENSION_DIR/sd-webui-aspect-ratio-helper

# Install Civitai Browser
git clone https://github.com/BlafKing/sd-civitai-browser-plus.git $EXTENSION_DIR/sd-civitai-browser-plus

# Install Image Browser
git clone https://github.com/zanllp/sd-webui-infinite-image-browsing.git $EXTENSION_DIR/sd-webui-infinite-image-browsing

# Install Controlnet
git clone https://github.com/Mikubill/sd-webui-controlnet $EXTENSION_DIR/sd-webui-controlnet

# Install Reactor
git clone https://github.com/Gourieff/sd-webui-reactor $EXTENSION_DIR/sd-webui-reactor
pip install insightface==0.7.3

# Install CLIP Interrogator
git clone https://github.com/pharmapsychotic/clip-interrogator-ext.git $EXTENSION_DIR/clip-interrogator-ext

# Install Upscaler
git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git $EXTENSION_DIR/ultimate-upscale-for-automatic1111

# Install adetailer
git clone https://github.com/Bing-su/adetailer.git $EXTENSION_DIR/adetailer

# Install 
git clone https://github.com/ahgsql/StyleSelectorXL $EXTENSION_DIR/StyleSelectorXL

# InstantX/InstantID
# TODO

# IP-Adapter
huggingface-cli download h94/IP-Adapter models/ip-adapter-plus-face_sd15.safetensors --local-dir $CONTROLNET_MODELS_DIR
huggingface-cli download h94/IP-Adapter models/ip-adapter_sd15.safetensors --local-dir $CONTROLNET_MODELS_DIR
mv $CONTROLNET_MODELS_DIR/models/* $CONTROLNET_MODELS_DIR
rmdir $CONTROLNET_MODELS_DIR/models

# IP-Adapter SDXL
huggingface-cli download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors --local-dir $CONTROLNET_MODELS_DIR
huggingface-cli download h94/IP-Adapter sdxl_models/ip-adapter_sdxl_vit-h.safetensors --local-dir $CONTROLNET_MODELS_DIR
mv $CONTROLNET_MODELS_DIR/sdxl_models/* $CONTROLNET_MODELS_DIR
rmdir $CONTROLNET_MODELS_DIR/sdxl_models

# IP-Adapter-FaceID
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid-plusv2_sd15.bin --local-dir $CONTROLNET_MODELS_DIR
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid-plusv2_sd15_lora.safetensors --local-dir $LORA_MODELS_DIR

# IP-Adapter-FaceID SDXL
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid-plusv2_sdxl.bin --local-dir $CONTROLNET_MODELS_DIR
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid-plusv2_sdxl_lora.safetensors --local-dir $LORA_MODELS_DIR

# Controlnet-v1-1
huggingface-cli download lllyasviel/ControlNet-v1-1 --include *.pth *.yaml --local-dir $CONTROLNET_MODELS_DIR

# MediaPipeFace
huggingface-cli download CrucibleAI/ControlNetMediaPipeFace control_v2p_sd15_mediapipe_face.safetensors control_v2p_sd15_mediapipe_face.yaml --local-dir $CONTROLNET_MODELS_DIR

# Model
aria2c "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16" -d $DIFFUSION_MODELS_DIR -o realisticVisionV51_v51VAE.safetensors

# Texual Inversion
# TODO