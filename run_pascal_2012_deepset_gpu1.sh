#!/usr/bin/env bash
source myVE/bin/activate
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH


source ./save_images.sh


fix="--gpu 1 --epochs 3 --data VOC2012 --grid 7"

#usage: vgg16.py
# [-h] [--lr LR] [--grid GRID] [--epochs EPOCHS]
# [--lw LW] [--rw RW] [--lf LF] [--af AF] [--data PV]


##change to see the different learning rate
#text="--lr 1e-3 "
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-3 --lf yes"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para

#text="--lr 1e-3 --lf yes --lw 10"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#mv 2012deepdc.txt $?


text="--lr 1e-3 --lf yes"
python vgg16.py $text $fix | tee 2012deepdc.txt
para=$(echo $text $fix | tr -d ' ')
dst=$(save_images $para)
mv 2012deepdc.txt $dst
#
#text="--lr 1e-3 --lf yes --lw 14"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#dst=$(save_images $para)
#mv 2012deepdc.txt $dst
#
#text="--lr 1e-3 --lf yes --lw 13"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#dst=$(save_images $para)
#mv 2012deepdc.txt $dst
#
#text="--lr 1e-3 --lf yes --lw 12"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#dst=$(save_images $para)
#mv 2012deepdc.txt $dst
#
#text="--lr 1e-3 --lf yes --lw 11"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#dst=$(save_images $para)
#mv 2012deepdc.txt $dst

#
#text="--lr 1e-3 --lf yes --rw 300"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-3 --lf yes --lw 300 --rw 300"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#text="--lr 1e-4 "
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
##custom loss function with default weights
#text="--lr 1e-4 --lf yes"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --lw 300"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --rw 300"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
#text="--lr 1e-4 --lf yes --lw 300 --rw 300"
#python vgg16.py $text $fix | tee 2012deepdc.txt
#para=$(echo $text $fix | tr -d ' ')
#save_images $para
#
