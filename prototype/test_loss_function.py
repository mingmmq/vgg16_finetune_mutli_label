import numpy as np
from keras import objectives
from keras import backend as K

_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)

    np_y_pred = K.eval(y_pred)
    sum_of_each = np.round(np.sum(np_y_pred, axis=1))


    np_y_true = K.eval(y_true)
    np_y_ones = np.ones(np.shape(np_y_true))

    for i in range(np.shape(sum_of_each)[0]):
        index = np.random.choice(np.shape(np_y_true)[1], 3*int(sum_of_each[i]))
        np_y_ones[i][np.array(index)] = 0
        np_y_ones[np_y_true>0.5] = 1

    new_y_true = K.constant(np_y_ones)


    out = -(y_true * K.log(y_pred) + (1.0 - new_y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def _loss_np(y_true, y_pred):

    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def check_loss(_shape):
    if _shape == '2d':
        shape = (6, 7)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.randint(2, size = shape)
    y_b = np.random.randint(2, size = shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1 - out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')


if __name__ == '__main__':
    test_loss()