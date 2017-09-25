#!/usr/bin/env bash
source myVE/bin/activate
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH


save_images(){
    time_stamp=$(date +%Y_%m_%d_%H_%M_%S)$1
    mkdir ./reports/$time_stamp

    #move the files to the folder as a record, with the time
    mv losses.png ./reports/$time_stamp/
    mv train_precision_recall.png ./reports/$time_stamp/
    mv val_precision_recall.png ./reports/$time_stamp/
    mv log.out ./reports/$time_stamp/

}


#usage: vgg16_pascal_deepset.py
# [-h] [--lr LR] [--grid GRID] [--epochs EPOCHS]
# [--lw LW] [--rw RW] [--lf LF] [--af AF] [--data PV]
text="--lr 1e-4 --epochs 40 --data VOC2012 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

#custom loss function with default weights
text="--lr 1e-4 --epochs 40 --data VOC2012 --lf yes --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data VOC2012 --lf yes --lw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data VOC2012 --lf yes --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data VOC2012 --lf yes --lf 300 --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para


#change to see the different learning rate
text="--lr 1e-3 --epochs 40 --data VOC2012 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data VOC2012 --lf yes --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data VOC2012 --lf yes --lw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data VOC2012 --lf yes --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data VOC2012 --lf yes --lf 300 --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para
