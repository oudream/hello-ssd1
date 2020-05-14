#!/usr/bin/env bash

# 1、准备VOC数据
# 2、修改Process.py并运行
# 3、修改configs/vgg_ssd300_voc0712.yaml
# 4、修改ssd/data/datasets/voc.py 类别
# 5、python train.py --config-file configs/vgg_ssd300_voc0712.yaml
# 6、修改demo.py第37行图片后缀
# 7、python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --score_threshold 0.7 --ckpt ./outputs/vgg_ssd300_voc0712/model_final.pth


from google.colab import drive
drive.mount("/content/drive")

!cd /content/drive/'My Drive'/'colab' && git clone https://github.com/oudream/hello-ssd1.git && !ls -R

!pip install torch torchvision yacs tqdm opencv-python vizer tensorboardX

!cd /content/drive/'My Drive'/colab/hello-ssd1 && python train.py --config-file configs/vgg_ssd300_voc0712.yaml

!cd /content/drive/'My Drive'/colab/hello-ssd1 && git pull origin master

!cd /content/drive/'My Drive'/colab/hello-ssd1 && python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --score_threshold 0.7 --ckpt ./outputs/vgg_ssd300_voc0712/model_final.pth
