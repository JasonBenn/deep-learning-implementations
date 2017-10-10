import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradient_checker import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    # First layer weights
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    # First layer bias
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H

    # Second layer weights
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    # Second layer bias
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    activations_1 = sigmoid(np.dot(data, W1) + b1)
    activations_2 = softmax(activations_1 @ W2 + b2)

    cost = -np.sum(np.log(np.max(activations_2 * labels, axis=1)))
    print(cost)

    # ???
    gradW1 = 0
    gradb1 = 0
    gradW2 = 0
    gradb2 = 0

    assert gradb2.shape == b2.shape
    assert gradW2.shape == W2.shape
    assert gradb1.shape == b1.shape
    assert gradW1.shape == W1.shape

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20  # number of data points
    dimensions = [10, 5, 10]  # first element becomes dimensionality of each datum
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    # print(data)  # will be 20 rows of 10 columsn. 20 datums, each 10d.

    # ---  ---
    labels = np.zeros((N, dimensions[2]))  # Create a matrix of size 20 x 10. These are one-hot encoded labels for each datum.
    for i in np.arange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    # This line makes no frickin sense to me. 115 random numbers? For what? Just random.randn(115)
    # Could this be randomly initialized weights and biases?
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )
    print(params)

    # This is going to take the gradient for the params, check that it makes sense.
    # forward_backward_prop returns cost and a gradient for any params.
    # So pass it any params (as above), and check that backward prop computed numerically (passing values really close to param, taking slope manually) matches what backprop has computed. Will only be correct if all the components of forward_backward_prop are working properly.
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)

if __name__ == "__main__":
    sanity_check()
