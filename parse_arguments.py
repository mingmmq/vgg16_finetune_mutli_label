global learning_rate
global grids_per_row
global nb_epoch
global left_weight
global right_weight
global use_custom_loss_function
global use_custom_accuracy_function
global dataset

def print_args():
    print("learning rate: ", learning_rate)
    print("grid: ", grids_per_row)
    print("number of epochs: ", nb_epoch)
    print("use custom loss function: ", use_custom_loss_function)
    print("loss left weight: ", left_weight)
    print("loss right weight: ", right_weight)
    print("use custom accuracy function:", use_custom_accuracy_function)
    print("pascal voc version", dataset)



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

    args = parser.parse_args()

    global learning_rate
    global grids_per_row
    global nb_epoch
    global left_weight
    global right_weight
    global use_custom_loss_function
    global use_custom_accuracy_function
    global dataset

    learning_rate = float(args.lr) if args.lr else 0.01
    grids_per_row = int(args.grid) if args.grid else 7
    nb_epoch = int(args.epochs) if args.epochs else 60
    left_weight = int(args.lw) if args.lw else 1
    right_weight = int(args.rw) if args.rw else 1
    use_custom_loss_function = True if args.lf else False
    use_custom_accuracy_function = True if args.af else False
    dataset = args.data if args.data else "VOC2007"


    #print the args here
    print_args()

