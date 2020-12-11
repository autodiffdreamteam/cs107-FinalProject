import numpy as np
from math import log, sin, cos, exp, tan
import math


class AutoDiff():
    
    def __init__(self, val, der=[1], seed=None, input_function=None):
        '''
        Constructs an AutoDiff object to perform forward automatic differentation.
        Parameters:
        self(AutoDiffPy): Self
        val (int/float): This is the seed value that we will evaluate at. 
        der (int/float): The derivative.
        input_function (str): An input function in string format in the general form of 'log(sin(3*x^2 + 5*x - x^5 * 5*x))' where 
        the inner function can be any polynomial, and there can be as many operations to the left of the parentheses as needed.
        Refer to the AutoDiffPy input_function user documentation for further syntax details.
        Returns:
        AutoDiffPy object with a value and derivative.
        '''
        if input_function != None:
            self.val = 0
            self.der = 0
            self.parse_input(input_function, val)
        else:
            if isinstance(val, (int, float)):
                val = [val]
            if len(val) == 1:
                try:
                    self.val = val[0].val
                    self.der = val[0].der
                except:
                    if isinstance(der, (int, float)):
                        der = [der]
                    self.val = np.array(val)
                    self.der = np.array(der)
            else:
                all_scalar = self._check_all_scalar(val)
                if all_scalar:
                    self.val = np.array(val)
                    self.der = np.array(der)
                else:
                    vals = []
                    ders = []
                    total_vars = self._get_total_vars(val)
                    for v in val:
                        try:
                            vals.append(v.val)
                            ders.append(v.der)
                        except:
                            vals.append(v)
                            ders.append(np.zeros(total_vars))
                    self.val = np.hstack((vals))
                    self.der = np.vstack((ders))

    def _check_all_scalar(self, val):
        '''
        Helper method for initialization. Returns True if all inputs are scalars.
        '''
        scalar_list = []
        for v in val:
            if isinstance(v, AutoDiff):
                scalar_list.append(v)
        return len(scalar_list) == 0

    def _get_total_vars(self, val):
        '''
        Helper method for initialization. Returns the total number of variables
        for a vector-valued function input.
        '''
        num_vars = []
        for v in val:
            if isinstance(v, AutoDiff):
                try:
                    num_vars.append(len(v.der[0]))
                except: # TypeError?
                    num_vars.append(v.der)
            else:
                num_vars.append(0)
        return np.max(num_vars)
    
    # Define a function value getter
    def val(self):
        return self.val

    # Define a function derivative getter
    def der(self):
        return self.der

    # Overload add dunder method
    def __add__(self, other):
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            return AutoDiff(self.val + other, self.der)

    # Handle right side addition
    def __radd__(self, other):
        return self.__add__(other)

    # Overload the subtraction dunder method
    def __sub__(self, other):
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            return AutoDiff(self.val - other, self.der)

    # Overload the right subtraction dunder method
    def __rsub__(self, other):
        return self.__sub__(other)

    # Overload the multiplication dunder method
    def __mul__(self, other):
        try:
            # Use the product rule to calculate derivative
            return AutoDiff(self.val * other.val, self.val*other.der + other.val*self.der)
        except AttributeError:
            return AutoDiff(self.val*other, self.der*other)

    # Handle right side multiplication
    def __rmul__(self, other):
        return self.__mul__(other)

    # Overload the division dunder method
    def __truediv__(self, other):
        try:
            # Use the quotient rule to calculate derivative
            return AutoDiff(self.val / other.val, (self.der*other.val - self.val*other.der) / other.val**2)
        except AttributeError:
            return AutoDiff(self.val / other, self.der / other)

    # Overload the right side division dunder method
    def __rtruediv__(self, other):
        return self.__truediv__(other)

    # Overload the power dunder method
    def __pow__(self, other):
        try:
            # Use the chain rule to calculate derivative
            return AutoDiff(self.val**other.val, other.val*(self.val**(other.val-1))*self.der + np.log(np.abs(self.val))*(self.val**other.val)*other.der)
        except AttributeError:
            return AutoDiff(self.val**other, other*self.der*(self.val**(other-1)))

    # Overload the right power dunder method
    def __rpow__(self, other):
        return self.__pow__(other)
    
    # Overload the negation dunder method
    def __neg__(self):
        return AutoDiff(-1*self.val, -1*self.der)

    def __str__(self):
        return 'Values: {}\nJacobian: {}'.format(self.val, self.der)

        # Define a function value getter
    def val(self):
        return self.val

    # Define a function derivative getter
    def der(self):
        return self.der

    def parse_input(self, input_function, val):
        func_list = ['exp(', 'cos(', 'sin(', 'tan(', 'log('] # etc
        expression = ''
        for i in range(len(input_function)):
            try:
                if input_function[i:i+4] in func_list:
                    expression +='fun.'
            except IndexError:
                pass
        
            if input_function[i] == '^':
                expression += '**'
            else:
                expression += input_function[i]
            # broken by minus constants
        function = lambda x: eval(expression)
        self.function = function
        self.val, self.der = self.evaluate_function(val)

    def evaluate_function(self, a):
        x = AutoDiff(a)
        new_vals = self.function(x)
        val = new_vals.val
        der = new_vals.der
        return val, der

        # check if this should return a new autodiff object?

    def newtons_method(self, x0, epsilon, max_iters):
        xn = x0
        for i in range(0,max_iters):
            fxn, Dfxn = self.evaluate_function(xn)
            print(fxn, "this is fxn")
            print(Dfxn, "this is dfxn")
            if abs(fxn) < epsilon:
                print('Found solution after {i} iterations.'.format(i=i))
                return xn
            if Dfxn == 0:
                print('Reached zero derivative. No solution found.')
                return None
            xn = xn - fxn/Dfxn
            print(xn, "this is xn")
        print('No solution found after max iterations.')
        return None


class fun:

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