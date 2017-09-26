#!/usr/bin/env bash
source myVE/bin/activate
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH


source ./save_images.sh

#usage: vgg16_pascal_deepset.py
# [-h] [--lr LR] [--grid GRID] [--epochs EPOCHS]
# [--lw LW] [--rw RW] [--lf LF] [--af AF] [--data PV]
fix="--gpu 0 --epochs 10 --data COCO"

#usage: vgg16.py
# [-h] [--lr LR] [--grid GRID] [--epochs EPOCHS]
# [--lw LW] [--rw RW] [--lf LF] [--af AF] [--data PV]
#change to see the different learning rate
#text="--lr 1e-3 "
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-3 --lf yes"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-3 --lf yes --lw 300"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para

text="--lr 1e-3 --lf yes --rw 300"
python vgg16.py $text $fix | tee coco.out
para=$(echo $text $fix | tr -d ' ')
save_images $para
#
#text="--lr 1e-3 --lf yes --lw 300 --rw 300"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 "
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
##custom loss function with default weights
#text="--lr 1e-4 --lf yes"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --lw 300"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --rw 300"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --lw 300 --rw 300"
#python vgg16.py $text $fix | tee coco.out
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#

