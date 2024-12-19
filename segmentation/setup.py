from google.colab import drive
drive.mount('/content/drive')

#######################################################

!git clone https://github.com/facebookresearch/sam2.git

%cd sam2

pip install -e .

##### 세션 다시 시작 #####
  
%cd checkpoints

! ./download_ckpts.sh

#######################################################
  
%cd ..
  
pwe   # /content/sam2
