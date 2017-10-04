# -*- coding: utf8 -*-

import numpy as np
import random

def softmax(x):
    # Softmax is invariant to bias: you can shift the entire input up or down by a scalar and the output will still be the same!
    # Because using a large input as an exponent will cause overflow errors, but using a very negative input as an exponent will just result in 0, we take advantage of this property to avoid overflows.
    # Subtract the largest value from the entire array.
    scaled = x - np.max(x)
    e_to_x = np.exp(scaled)
    return e_to_x / e_to_x.sum()

def test_softmax():
    test1 = softmax(np.array([1,2]))
    assert np.allclose(test1, np.array([0.26894142,  0.73105858]))
    print("✅")

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    assert np.allclose(test2, np.array([[0.26894142, 0.73105858], [0, 0]]))
    print("✅")

    test3 = softmax(np.array([[-1001,-1002]]))
    assert np.allclose(test3, np.array([0.73105858, 0.26894142]))
    print("✅")

if __name__ == "__main__":
    test_softmax()
