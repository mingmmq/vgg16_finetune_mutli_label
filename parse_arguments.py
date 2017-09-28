global random_sample
global learning_rate
global grids_per_row
global nb_epoch
global left_weight
global right_weight
global use_custom_loss_function
global use_custom_accuracy_function
global dataset
global gpu
global set_type

def print_args():
    print("learning rate: ", learning_rate)
    print("grid: ", grids_per_row)
    print("number of epochs: ", nb_epoch)
    print("use custom loss function: ", use_custom_loss_function)
    print("loss left weight: ", left_weight)
    print("loss right weight: ", right_weight)
    print("use custom accuracy function:", use_custom_accuracy_function)
    print("pascal voc version", dataset)
    print("the gpu", gpu)



def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate')
    parser.add_argument('--grid', help="grid per row and column")
    parser.add_argument('--epochs', help="number of epochs")
    parser.add_argument('--lw', help="left weight on the loss function")
    parser.add_argument('--rw', help="right weight on the loss function")
    parser.add_argument('--lf', help="loss function")
    parser.add_argument('--af', help="custom accuracy")
    parser.add_argument('--data', help="choose the data")
    parser.add_argument('--gpu', help="choose the gpu")
    parser.add_argument('--st', help="choose the set type")
    parser.add_argument('--rs', help="random sampe times of max")


    args = parser.parse_args()

    global learning_rate
    global grids_per_row
    global nb_epoch
    global left_weight
    global right_weight
    global use_custom_loss_function
    global use_custom_accuracy_function
    global dataset
    global gpu
    global set_type
    global random_sample

    learning_rate = float(args.lr) if args.lr else 0.01
    grids_per_row = int(args.grid) if args.grid else 1
    nb_epoch = int(args.epochs) if args.epochs else 60
    left_weight = int(args.lw) if args.lw else 1
    right_weight = int(args.rw) if args.rw else 1
    use_custom_loss_function = True if args.lf else False
    use_custom_accuracy_function = True if args.af else False
    dataset = args.data if args.data else "VOC2007"
    gpu = int(args.gpu) if args.gpu else 0
    set_type = args.st if args.st else "all"
    random_sample = int(args.rs) if args.rs else None



    #print the args here
    print_args()

