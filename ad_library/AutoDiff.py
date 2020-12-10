import numpy as np
from math import log, sin, cos, exp, tan
import math

class AutoDiff():
    
    def __init__(self, val, der=[1], input_pos=None, input_function=None):
        '''
        Constructs an AutoDiff object on which to perform forward automatic differentation.
        INPUTS
        ======
        val: int, float, list, or np.ndarray, required
            Value of the function(s) at which to compute the derivative.
        der: list or np.ndarray, optional, default is [1]
            Derivative or Jacobian of the function(s) passed. Note that
            when creating a new AutoDiff object, this is equivalent to 
            passing a seed vector. When pasing in 'n' inputs, der will
            be a length 'n' array of zeros, with a non-zero element 
            indicating the value of the variable at that index to be
            seeded at.
        input_pos: list or np.ndarray, optional, default is None
            Allows the user to more easily create a seed vector for 
            functions with many inputs by passing in a length 2 array,
            where the first element is the total number of inputs and
            the second element is the index of that variable.
        input_function: str, optional, default is None
            Allows the user to more easily define a function by passing
            in a string.
        RETURNS
        =======
        An AutoDiff object with the calculated value(s) and derivative(s).
        EXAMPLES
        ========
        # Create a scalar input
        >>> f = AutoDiff(2.0)
        >>> print(f)
        Values:
        [2.]
        Jacobian:
        [1]
        
        # Manually create a vector input and seed
        >>> x = AutoDiff(2.0, [1, 0])
        >>> y = AutoDiff(3.0, [0, 1])
        >>> f = AutoDiff([x, y])
        >>> print(f)
        Values:
        [2. 3.]
        Jacobian:
        [[1 0]
         [0 1]]
        
        # Automatically create a vector input with default seed
        >>> v = ad(2.0, input_pos=[5, 0])
        >>> w = ad(3.0, input_pos=[5, 1])
        >>> x = ad(4.0, input_pos=[5, 2])
        >>> y = ad(5.0, input_pos=[5, 3])
        >>> z = ad(6.0, input_pos=[5, 4])
        >>> f = ad([v, w, x, y, z])
        >>> print(f)
        Values:
        [2. 3. 4. 5. 6.]
        Jacobian:
        [[1. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0.]
         [0. 0. 1. 0. 0.]
         [0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 1.]]
        '''
        if input_function != None:
            self._val = 0
            self._der = 0
            self.parse_input(input_function, val)
        else:
            if input_pos != None:
                if not isinstance(input_pos, (list, np.ndarray)):
                    raise AssertionError('input_pos must be a list or np.ndarray')
                if len(input_pos) != 2:
                    raise AssertionError('input_pos must be of length 2')
                if input_pos[0] <= input_pos[1]:
                    raise AssertionError('variable index must not exceed total number of variables')
                else:
                    der = np.zeros((input_pos[0]))
                    der[input_pos[1]] = 1
            if isinstance(val, (int, float)):
                val = [val]
            if len(val) == 1:
                try:
                    self._val = val[0].val
                    self._der = val[0].der
                except:
                    self._val = np.array(val)
                    self._der = np.array(der)
            else:
                all_scalar = self._check_all_scalar(val)
                if all_scalar:
                    self._val = np.array(val)
                    self._der = np.array(der)
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
                    self._val = np.hstack((vals))
                    self._der = np.vstack((ders))
            
    def _check_all_scalar(self, val):
        '''
        Helper method for __init__ that returns True if all inputs are scalars,
        False otherwise.
        '''
        scalar_list = []
        for v in val:
            if isinstance(v, AutoDiff):
                scalar_list.append(v)
        return len(scalar_list) == 0

    def _get_total_vars(self, val):
        '''
        Helper method for __init__ that returns the total number of variables for
        a function with multiple inputs.
        '''
        num_vars = []
        for v in val:
            if isinstance(v, AutoDiff):
                try:
                    num_vars.append(len(v.der[0]))
                except:
                    num_vars.append(v.der)
            else:
                num_vars.append(0)
        return np.max(num_vars)
    
    @property
    def val(self):
        '''
        Returns the value(s) of the AutoDiff object.
        '''
        return self._val

    @property
    def der(self):
        '''
        Returns the derivative(s) (Jacobian) of the AutoDiff object.
        '''
        return self._der

    def __add__(self, other):
        '''
        Performs addition of self and other.
        '''
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            return AutoDiff(self.val + other, self.der)

    def __radd__(self, other):
        '''
        Performs addition of other and self (commutative).
        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        Performs subtraction of other from self.
        '''
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            return AutoDiff(self.val - other, self.der)

    def __rsub__(self, other):
        '''
        Performs subtraction of self from other.
        '''
        try:
            return AutoDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            return AutoDiff(other - self.val, -self.der)

    def __mul__(self, other):
        '''
        Performs multiplication of self with other.
        '''
        try:
            # Use the product rule to calculate derivative
            return AutoDiff(np.multiply(self.val, other.val), np.multiply(self.val, other.der) + np.multiply(other.val, self.der))
        except AttributeError:
            return AutoDiff(self.val*other, self.der*other)

    def __rmul__(self, other):
        '''
        Performs multiplication of other with self (commutative).
        '''
        return self.__mul__(other)

    def __truediv__(self, other):
        '''
        Performs division of self by other.
        '''
        try:
            # Use the quotient rule to calculate derivative
            return AutoDiff(np.divide(self.val, other.val), np.divide(np.multiply(self.der, other.val) - np.multiply(self.val, other.der), other.val**2))
        except AttributeError:
            return AutoDiff(np.divide(self.val, other), np.divide(self.der, other))

    def __rtruediv__(self, other):
        '''
        Performs division of other by self.
        '''
        try:
            return AutoDiff(np.divide(other.val, self.val), np.divide(np.multiply(other.der, self.val) - np.multiply(other.val, self.der), self.val**2))
        except AttributeError:
            return AutoDiff(np.divide(other, self.val), -np.divide(np.multiply(other, self.der), self.val**2))
        
    def __pow__(self, other):
        '''
        Raises self to the power of other.
        '''
        try:
            # Use the chain rule to calculate derivative
            return AutoDiff(np.power(self.val, other.val), ((other.val / self.val)*self.der + np.log(self.val)*other.der)*np.power(self.val, other.val))
        except AttributeError:
            if other == 0:
                return AutoDiff(1, der=[0])
            return AutoDiff(np.power(self.val, other), other*np.multiply(self.der, (self.val**(other-1))))

    def __rpow__(self, other):
        '''
        Raises other to the power of self.
        '''
        return AutoDiff(other**self.val, (other**self.val)*np.log(other)*self.der)

    def __neg__(self):
        '''
        Negates the function value(s) and derivative(s).
        '''
        return AutoDiff(-1*self.val, -1*self.der)

    def __pos__(self):
        '''
        Applies the unary + operator to self.
        '''
        return self

    def __abs__(self):
        '''
        Applies the absolute value to the function(s).
        '''
        return AutoDiff(abs(self.val), (self.val*self.der / abs(self.val)))

    def __eq__(self, other):
        '''
        Returns True if self and other have the same values and derivatives,
        False otherwise.
        '''
        try:
            return (np.all(np.equal(self.val, other.val)) and np.all(np.equal(self.der, other.der)))
        except AttributeError:
            return ((self.val == other) and (self.der == [1]))

    def __ne__(self, other):
        '''
        Returns False if self and other have the same values and derivatives,
        True otherwise.
        '''
        return not self.__eq__(other)

    def __lt__(self, other):
        '''
        Returns an array of booleans, where each is True if the value of self
        is less than the corresponding value of other, False otherwise.
        '''
        try:
            return self.val < other.val
        except AttributeError:
            return self.val < other

    def __gt__(self, other):
        '''
        Returns an array of booleans, where each is True if the value of self
        is greater than the corresponding value of other, False otherwise.
        '''
        try:
            return self.val > other.val
        except AttributeError:
            return self.val > other

    def __le__(self, other):
        '''
        Returns an array of booleans, where each is True if the value of self
        is less than or equal to the corresponding value of other, False otherwise.
        '''
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        '''
        Returns an array of booleans, where each is True if the value of self
        is greater than or equal to the corresponding value of other, False otherwise.
        '''
        return self.__gt__(other) or self.__eq__(other)

    def parse_input(self, input_function, val):
        func_list = ['exp(', 'cos(', 'sin(', 'tan(', 'log('] # etc
        expression = ''
        for i in range(len(input_function)):
            try:
                if input_function[i:i+4] in func_list:
                    expression +='Function.'
                    print(expression)
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
        print(self.val, self.der)

    def evaluate_function(self, a):
        x = AutoDiff(a)
        new_vals = self.function(x)
        print(new_vals)
        val = new_vals.val
        der = new_vals.der
        print(val, der)
        return val, der

    def __str__(self):
        '''
        Returns a string representation of the function value(s) and 
        derivative(s) (Jacobian).
        '''
        return 'Values:\n{}\nJacobian:\n{}'.format(self.val, self.der)
    

class Function: # dummy class for internal utility

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


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



