#!/usr/bin/env bash
source myVE/bin/activate
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH


save_images(){
    time_stamp=$(date +%Y_%m_%d_%H_%M_%S)$1
    mkdir $time_stamp

    #move the files to the folder as a record, with the time
    #done: should extract some configuration from the script and name the folder with the difference configuration
    mv losses.png $time_stamp/
    mv train_precision_recall.png $time_stamp/
    mv val_precision_recall.png $time_stamp/
    mv log.out $time_stamp/

    # 1.check the general process
    #2.use the tee to put the output into the logs
}


#usage: vgg16_pascal_deepset.py
# [-h] [--lr LR] [--grid GRID] [--epochs EPOCHS]
# [--lw LW] [--rw RW] [--lf LF] [--af AF] [--data PV]
text="--lr 1e-4 --epochs 40 --data COCO --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

#custom loss function with default weights
text="--lr 1e-4 --epochs 40 --data COCO --lf yes --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data COCO --lf yes --lw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data COCO --lf yes --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-4 --epochs 40 --data COCO --lf yes --lf 300 --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para


#change to see the different learning rate
text="--lr 1e-3 --epochs 40 --data COCO --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data COCO --lf yes --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data COCO --lf yes --lw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data COCO --lf yes --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para

text="--lr 1e-3 --epochs 40 --data COCO --lf yes --lf 300 --rw 300 --grid 7"
python vgg16_pascal_deepset.py $text | tee log.out
para=$(echo $text | tr -d ' ')
save_images $para
