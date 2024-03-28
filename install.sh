## ninja 
sudo wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
sudo gunzip /usr/local/bin/ninja.gz
sudo chmod a+x /usr/local/bin/ninja

## gcc 
sudo apt install build-essential

## detectron2

pip install 'git+https://github.com/facebookresearch/detectron2.git'


pip install gradio==3.16.2
pip install albumentations==1.3.0
pip install opencv-contrib-python
pip install imageio==2.9.0
pip install imageio-ffmpeg==0.4.2
pip install pytorch-lightning==1.5.0
pip install omegaconf==2.1.1
pip install test-tube>=0.7.5
pip install streamlit==1.12.1
pip install einops==0.3.0
pip install webdataset==0.2.5
pip install kornia==0.6
pip install open_clip_torch==2.0.2
pip install invisible-watermark>=0.1.5
pip install streamlit-drawable-canvas==0.8.0
pip install torchmetrics==0.6.0
pip install timm==0.6.12
pip install addict==2.4.0
pip install yapf==0.32.0
pip install prettytable==3.6.0
pip install safetensors==0.2.7
pip install basicsr==1.4.2
pip install numpy==1.23.1
pip install fvcore
pip install pycocotools
pip install wandb
pip install ipykernel
pip install diffusers


pip install pillow==9.0.1

pip install accelerate==0.24.0

pip install datasets

pip install transformers==4.29.2

pip install evaluate

pip install huggingface_hub

pip install wand

pip install dask

sudo apt-get install -y libmagickwand-dev



# idk why but this solves the cv2 problem
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python
sudo apt-get install -y libgl1


# for waterbirds
pip install wilds

probably not necessary
# pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers



pip install scikit-image==0.20.0

pip install git+https://github.com/openai/CLIP.git
  
pip install xformers==0.0.22.post7

pip install --upgrade --no-deps --force-reinstall torch==2.1.0 
pip install --upgrade --no-deps --force-reinstall torchvision==0.16.1
pip install --upgrade --no-deps --force-reinstall torchaudio==2.1.1


