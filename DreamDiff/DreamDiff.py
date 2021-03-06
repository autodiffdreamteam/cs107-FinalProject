import numpy as np


class DreamDiff():
    
    def __init__(self, val, der=[1], input_pos=None):
        '''
        Constructs an DreamDiff object on which to perform forward automatic differentation.

        INPUTS
        ======
        val: int, float, list, or np.ndarray, required
            Value of the function(s) at which to compute the derivative.
        der: list or np.ndarray, optional, default is [1]
            Derivative or Jacobian of the function(s) passed. Note that
            when creating a new DreamDiff object, this is equivalent to 
            passing a seed vector. When pasing in 'n' inputs, der will
            be a length 'n' array of zeros, with a non-zero element 
            indicating the value of the variable at that index to be
            seeded at.
        input_pos: list or np.ndarray, optional, default is None
            Allows the user to more easily create a seed vector for 
            functions with many inputs by passing in a length 2 array,
            where the first element is the total number of inputs and
            the second element is the index of that variable.

        RETURNS
        =======
        An DreamDiff object with the calculated value(s) and derivative(s).

        EXAMPLES
        ========
        # Create a scalar input
        >>> f = DreamDiff(2.0)
        >>> print(f)
        Values:
        [2.]
        Jacobian:
        [1]
        
        # Manually create a vector input and seed
        >>> x = DreamDiff(2.0, [1, 0])
        >>> y = DreamDiff(3.0, [0, 1])
        >>> f = DreamDiff([x, y])
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
                self._val = np.array(vals)
                self._der = np.array(ders)
            
    def _check_all_scalar(self, val):
        '''
        Helper method for __init__ that returns True if all inputs are scalars,
        False otherwise.
        '''
        scalar_list = []
        for v in val:
            if isinstance(v, DreamDiff):
                scalar_list.append(v)
        return len(scalar_list) == 0

    def _get_total_vars(self, val):
        '''
        Helper method for __init__ that returns the total number of variables for
        a function with multiple inputs.
        '''
        num_vars = []
        for v in val:
            if isinstance(v, DreamDiff):
                try:
                    num_vars.append(len(v.der[0]))
                except:
                    num_vars.append(len(v.der))
            else:
                num_vars.append(0)
        return np.max(num_vars)
    
    @property
    def val(self):
        '''
        Returns the value(s) of the DreamDiff object.
        '''
        return self._val

    @property
    def der(self):
        '''
        Returns the derivative(s) (Jacobian) of the DreamDiff object.
        '''
        return self._der

    def __add__(self, other):
        '''
        Performs addition of self and other.
        '''
        try:
            return DreamDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            return DreamDiff(self.val + other, self.der)

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
            return DreamDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            return DreamDiff(self.val - other, self.der)

    def __rsub__(self, other):
        '''
        Performs subtraction of self from other.
        '''
        try:
            return DreamDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            return DreamDiff(other - self.val, -self.der)

    def __mul__(self, other):
        '''
        Performs multiplication of self with other.
        '''
        try:
            # Use the product rule to calculate derivative
            return DreamDiff(self.val*other.val, self.val*other.der + other.val*self.der)
        except AttributeError:
            return DreamDiff(self.val*other, self.der*other)

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
            return DreamDiff(self.val / other.val, (self.der*other.val - self.val*other.der) / other.val**2)
        except AttributeError:
            return DreamDiff(self.val / other, self.der / other)

    def __rtruediv__(self, other):
        '''
        Performs division of other by self.
        '''
        try:
            # Use the quotient rule to calculate derivative
            return DreamDiff(other.val / self.val, (other.der*self.val - other.val*self.der) / self.val**2)
        except AttributeError:
            return DreamDiff(other / self.val, -other*self.der / self.val**2)
        
    def __pow__(self, other):
        '''
        Raises self to the power of other.
        '''
        try:
            # Use the chain rule to calculate derivative
            return DreamDiff(self.val**other.val, other.val*(self.val**(other.val-1))*self.der + np.log(np.abs(self.val))*(self.val**other.val)*other.der)
        except AttributeError:
            if other == 0:
                return DreamDiff(1, der=[0])
            return DreamDiff(self.val**other, other*(self.val**(other-1))*self.der)

    def __rpow__(self, other):
        '''
        Raises other to the power of self.
        '''
        try:
            # Use the chain rule to calculate derivative
            return DreamDiff(other.val**self.val, other.val*(self.val**(other.val-1))*self.der + np.log(np.abs(self.val))*(self.val**other.val)*other.der)
        except AttributeError:
            return DreamDiff(other**self.val, np.log(other)*(other**self.val)*self.der)

    def __neg__(self):
        '''
        Negates the function value(s) and derivative(s).
        '''
        return DreamDiff(-1*self.val, -1*self.der)

    def __pos__(self):
        '''
        Applies the unary + operator to self.
        '''
        return self

    def __abs__(self):
        '''
        Applies the absolute value to the function(s).
        '''
        return DreamDiff(abs(self.val), (self.val*self.der / abs(self.val)))

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

    def _parse_input(self, input_function):
        '''
        Internal method that takes in an input function in string format, 
        e.g. 'x^2 + sin(x) + cos(log(x))'
        Returns a lambda expression with functions replaced by their Function.
        '''
        func_list = ['exp(', 'cos(', 'sin(', 'tan(', 'log(', 'arcsin(', 'arccos(', 
                'arctan(', 'sinh(', 'cosh(', 'tanh(', 'sqrt(', 'log2(', 'log10(', 'logistic('] 
        expression = ''
        for i in range(len(input_function)):
            for j in range(5):
                try:
                    if input_function[i:i+j+4] in func_list:
                        expression +='Function.'
                        if input_function[i+3] != '(':
                            input_function[i+3] = '0' # nullifies to make prefix free
                except IndexError:
                    pass
            if input_function[i] == '^':
                expression += '**'
            else:
                expression += input_function[i]
        function = lambda x: eval(expression)
        return function

    def _evaluate_function(self, function, a):
        '''
        Internal method to evaluate a lambda function at a given point 'a'.
        Returns an DreamDiff object with the value and derivative of the 
        function evaluated at 'a'.
        '''
        ad_object = DreamDiff(a)
        new_vals = function(ad_object)
        return new_vals

    def __str__(self):
        '''
        Returns a string representation of the function value(s) and 
        derivative(s) (Jacobian).
        '''
        return 'Values:\n{}\nJacobian:\n{}'.format(self.val, self.der)
    

class Function: 
    '''
    Class to hold elementary functions for DreamDiff objects, including
    trigonometric, inverse trigonometric, exponential, hyperbolic, logistic,
    logarithm, and square root functions.
    '''

    def sin(x):
        '''
        Returns the sine of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.sin(x.val), np.cos(x.val)*x.der)

    def cos(x):
        '''
        Returns the cosine of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.cos(x.val), -np.sin(x.val)*x.der)

    def tan(x):
        '''
        Returns the tangent of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.tan(x.val), (1 / (np.cos(x.val)**2))*x.der)

    def arcsin(x):
        '''
        Returns the arcsine of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.arcsin(x.val), (1 / np.sqrt(1 - (x.val**2))*x.der))

    def arccos(x):
        '''
        Returns the arccosine of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.arccos(x.val), (-1 / np.sqrt(1 - (x.val**2))*x.der))
    
    def arctan(x):
        '''
        Returns the arctangent of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.arctan(x.val), (1 / (1 + x.val**2))*x.der)

    def sinh(x):
        '''
        Returns the sinh of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.sinh(x.val), (np.cosh(x.val))*x.der)

    def cosh(x):
        '''
        Returns the cosh of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.cosh(x.val), (np.sinh(x.val))*x.der)

    def tanh(x):
        '''
        Returns the tanh of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.tanh(x.val), (1 / (np.cosh(x.val)**2))*x.der)

    def sqrt(x):
        '''
        Returns the square root of an DreamDiff object with its updated derivative.
        '''
        return DreamDiff(np.sqrt(x.val), 0.5*(x.val**(-0.5))*x.der)

    def exp(x):
        '''
        Returns the exponential of an DreamDiff object and its updated derivative.
        '''
        return DreamDiff(np.exp(x.val), np.exp(x.val)*x.der)

    def log(x):
        '''
        Returns the log (base e) of an DreamDiff object and its updated derivative.
        '''
        return DreamDiff(np.log(x.val), (1 / (x.val))*x.der)

    def log2(x):
        '''
        Returns the log (base 2) of an DreamDiff object and its updated derivative.
        '''
        return DreamDiff(np.log2(x.val), (1 / (x.val*np.log(2)))*x.der)
    
    def log10(x):
        '''
        Returns the log (base 10) of an DreamDiff object and its updated derivative.
        '''
        return DreamDiff(np.log10(x.val), (1 / (x.val*np.log(10)))*x.der)

    def logistic(x):
        '''
        Applies the logistic function to an DreamDiff object and returns its
        updated derivative.
        '''
        return DreamDiff(1 / (1 + np.exp(-x.val)), (np.exp(x.val) / (1 + np.exp(x.val))**2)*x.der)


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
