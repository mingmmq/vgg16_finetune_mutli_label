from keras.datasets import cifar10
import numpy as np
from cs231n.NearestNeighbor import NearestNeighbor

(Xtr, Ytr), (Xte, Yte) = cifar10.load_data()
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]

Xtr_rows = Xtr_rows[1000:, :]
Ytr = Ytr[1000:]

validation_accuracies = []
for  k in [1, 3, 5, 10, 20, 50, 100]:
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)

    Yval_predict = nn.predict(Xval_rows, k = k)
    acc = np.mean(Yval_predict == Yval)
    print('accuracy: %f'% (acc, ))


    validation_accuracies.append((k, acc))

print(validation_accuracies)

