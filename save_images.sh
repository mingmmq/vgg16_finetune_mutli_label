#!/usr/bin/env bash

save_images(){
    time_stamp="_"$1$(date +%Y_%m_%d_%H_%M_%S)
    mkdir ./reports/$time_stamp

    #move the files to the folder as a record, with the time
    mv losses.png ./reports/$time_stamp/
    mv train_precision_recall.png ./reports/$time_stamp/
    mv val_precision_recall.png ./reports/$time_stamp/
    mv acc_1.png ./reports/$time_stamp/
    mv acc_2.png ./reports/$time_stamp/

}