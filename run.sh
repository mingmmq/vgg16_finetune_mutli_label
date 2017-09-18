#!/usr/bin/env bash
nohup python vgg16_deepset.py &

echo "This will wait until the script are done"
date
wait
date
echo "Done"

time_stamp=$(date +%Y_%m_%d_%H_%M_%S)
mkdir $time_stamp


#move the files to the folder as a record, with the time
#todo: should extract some configuration from the script and name the folder with the difference configuration
mv losses.png $time_stamp/
mv train_precision_recall.png $time_stamp/
mv val_precision_recall.png $time_stamp/
mv nohup.out $time_stamp/


