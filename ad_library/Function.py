import numpy as np
from AutoDiff import AutoDiff

class Function:

    # Elemental exponential function
    def exp(x):
        return AutoDiff(np.exp(x.val), x.der*np.exp(x.val))

    # Elemental logarithm function
    def log(x):
        return AutoDiff(np.log(x.val), x.der/x.val)

    # Elemental sine function
    def sin(x):
        return AutoDiff(np.sin(x.val), np.cos(x.val)*x.der)

    # Elemental cosine function
    def cos(x):
        return AutoDiff(np.cos(x.val), -np.sin(x.val)*x.der)

    # Elemental tangent function
    def tan(x):
        return AutoDiff(np.tan(x.val), np.arctan(x.val)*x.der)
