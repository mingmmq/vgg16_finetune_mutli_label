from keras.datasets import cifar10
import numpy as np

from cs231n.NearestNeighbor import NearestNeighbor

(Xtr, Ytr), (Xte, Yte) = cifar10.load_data()
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)



nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)


print('accuracy: %f' %(np.mean(Yte_predict == Yte)))

