# -*- coding: utf8 -*-

import numpy as np
import random
from sklearn.metrics import log_loss
import sys

def gradcheck_naive(f, x):
    """
    Check that the gradient for a function f is correct
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)
    h = 1e-4



    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # This approximates the derivative of this function at x[ix].
        slightly_higher, _ = f(x[ix] + h)
        slightly_lower, _ = f(x[ix] - h)
        numgrad = (slightly_higher - slightly_lower) / (2 * h)

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > 1e-5:
            print("❌  First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            sys.exit(0)

        it.iternext() # Step to next dimension

def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    gradcheck_naive(quad, np.array(123.456))      # scalar test
    print("✅")
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    print("✅")
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("✅")

if __name__ == "__main__":
    sanity_check()
