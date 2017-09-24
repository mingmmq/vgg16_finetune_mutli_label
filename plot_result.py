def plot_result(plt, history):
    # # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()
    plt.savefig('losses.png')
    plt.clf()

    # # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()
    plt.savefig('acc.png')
    plt.clf()

    print(history.history)


    plt.plot(history.history['precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['f1'])
    plt.title('train precision recall')
    plt.ylabel('score')
    plt.legend(['precision', 'recall', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('train_precision_recall.png')
    plt.clf()

    plt.plot(history.history['val_precision'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['val_f1'])
    plt.title('val precision recall')
    plt.ylabel('score')
    plt.legend(['precision', 'recall', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('val_precision_recall.png')
    plt.clf()

