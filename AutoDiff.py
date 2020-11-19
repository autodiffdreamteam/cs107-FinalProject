import numpy as np

class AutoDiff():

    def __init__(self, val, der=1):
        self.val = val
        self.der = der

    # Overload add dunder method
    def __add__(self, other):
        try:
            self.val += other.val
            self.der += other.der
        except AttributeError:
            self.val += other
        return self

    # Overload the right side addition dunder method
    def __radd__(self, other):
        return self.__add__(other)

    # Overload the subtraction dunder method
    def __sub__(self, other):
        return self.__add__(-other)

    # Overload the right subtraction dunder method
    def __rsub__(self, other):
        return self.__sub__(other)

    # Overload multiplication dunder method
    def __mul__(self, other):
        try:
            # Use the product rule to calculate derivative
            self.der = self.val*other.der + other.val*self.der
            self.val *= other.val
        except AttributeError:
            self.val *= other
            self.der *= other
        return self

    # Overload the right side multiplication dunder method
    def __rmul__(self, other):
        return self.__mul__(other)

    # Overload the division dunder method
    def __truediv__(self, other):
        try:
            # Use the quotient rule to calculate derivative
            self.der = (self.der*other.val - self.val*other.der) / other.val**2
            self.val /= other.val
        except AttributeError:
            self.val /= other
            self.der /= other
        return self

    # Overload the right side division dunder method
    def __rtruediv__(self, other):
        return self.__truediv__(other)

    # Overload the power dunder method
    def __pow__(self, other):
        try:
            # Use the chain rule to calculate derivative
            self.der = np.log(self.val)*(self.val**other.val)
            self.val **= other.val
        except AttributeError:
            self.val **= other
            self.der = other.val*(self.val**(other - 1))
        return self

    # Overload the right power dunder method
    def __rpow__(self, other):
        return self.__pow__(other)

    # Overload the negation dunder method
    def __neg__(self):
        self.val *= -1
        self.der *= -1

    # Define a function value getter
    def value(self):
        return self.val

    # Define a function derivative getter
    def derivative(self):
        return self.der
