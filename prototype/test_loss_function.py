import numpy as np
from keras import objectives
from keras import backend as K

_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):

    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    sum_of_each = K.round(K.sum(y_true, axis=1))
    keep_of_each = sum_of_each * 3;
    max = K.max(keep_of_each)
    shape = K.shape(y_true)
    random_tensor = K.random_binomial(shape=shape, p= (shape[1]-2)/(shape[1]))
    n_true =K.clip(y_true + random_tensor,K.epsilon(),1.0-K.epsilon())

    print(K.eval(K.sum(random_tensor)))

    out = -(y_true * K.log(y_pred) + (1.0 - n_true) * K.log(1.0 - y_pred))

    out1 = K.mean(out, axis=-1)
    out2 = K.mean(out, axis=1)
    return K.mean(out, axis=-1)

def _loss_np(y_true, y_pred):

    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def check_loss(_shape):
    if _shape == '2d':
        shape = (2, 3)
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