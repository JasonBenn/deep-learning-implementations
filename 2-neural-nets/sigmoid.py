# -*- coding: utf8 -*-

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_grad(f):
    # Calculus is neat :)
    return f - np.power(f, 2)

def test_sigmoid():
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    assert np.allclose([[0.73105858, 0.88079708], [0.26894142, 0.11920292]], f)
    print("✅")
    assert np.allclose([[0.19661193, 0.10499359], [0.19661193, 0.10499359]], g)
    print("✅")

if __name__ == "__main__":
    test_sigmoid()
