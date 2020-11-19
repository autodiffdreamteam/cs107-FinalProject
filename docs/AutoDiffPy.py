import numpy as np
from math import log, sin, cos, exp, tan
import math


class AutoDiffPy():
    
    def __init__(self, val, der = 1, input_function = None):
        '''
        Constructs an AutoDiffPy object to perform forward automatic differentation.

        Parameters:
        self(AutoDiffPy): Self
        val (int/float): This is the seed value that we will evaluate at. 
        der (int/float): The derivative.
        input_function (str): An input function in string format in the general form of 'log(sin(3*x^2 + 5*x - x^5 * 5*x))' where the inner function can be any polynomial, and there can be as many operations to the left of the parentheses as needed.
        Refer to the AutoDiffPy input_function user documentation for further syntax details.

        Returns:
        AutoDiffPy object with a value and derivative.
        '''


        if input_function !=None:
            self.val = 0
            self.der = 0
            self.parse_input(input_function, val)
        else:        
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

    # Handle right side addition
    def __radd__(self, other):
        return self.__add__(other)

    # Overload the subtraction dunder method
    def __sub__(self, other):
        return self.__add__(-other)

    # Overload the right subtraction dunder method
    def __rsub__(self, other):
        return self.__sub__(other)

    # Overload the multiplication dunder method
    def __mul__(self, other):
        try:
            # Use the product rule to calculate derivative
            self.der = self.val*other.der + other.val*self.der
            self.val *= other.val
        except AttributeError:
            self.val *= other
            self.der *= other
        return self

    # Handle right side multiplication
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
    def val(self):
        return self.val

    # Define a function derivative getter
    def der(self):
        return self.der

    # Define sin function for true value and derivative
    def sin(self, x):
        return AutoDiffPy(np.sin(x.val), np.cos(x.val)*x.der)

    # Define cos function for true value and derivative
    def cos(self, x):
        return AutoDiffPy(np.cos(x.val), -np.sin(x.val)*x.der)

    # Define tan function for true value and derivative
    def tan(self, x):
        return AutoDiffPy(np.tan(x.val), (np.arccos(x.val)**2)*x.der)

    # Define log function for true value and derivative
    def log(self, x):
        return AutoDiffPy(np.log(x.val), (1/x.val)*x.der)

    # Define exp function for true value and derivative
    def exp(self, x):
        return AutoDiffPy(np.exp(x.val), (np.exp(x.val))*x.der)
    
    # Define method to access AutoDiffPy functions by string
    def derivative_dict(self, function):
        '''
        Takes in an a string representing a function and returns a tuple of the update value and derivative

        Parameter(s):

        self(AutoDiffPy): Self, an AutoDiffPy object
        function(str): A 3 character string denoting an elementary function like cos or log
        
        Returns:

        autodiff_new.val(float/int): The new value after applying the specified function
        autodiff_new.der(float/int): The new derivative after applying the specified function
        '''

        derivative_dict = {'sin': self.sin, 'cos': self.cos, 'tan': self.tan, 'log': self.log, 'exp': self.exp}
        autodiff_new = derivative_dict[function](self)
        return autodiff_new.val, autodiff_new.der

    def parse_input(self, input_function, a): 
        '''
        parse_input parses an input function of the form (write format parameters) , sequentially feeding all operations and derivatives of such to perform forward automatic differentation on the given input_function.

        Parameters:
        self(AutoDiffPy): Self, an AutoDiffPy object
        input_function(str):
        a(int/float): The value which the input function will be evaluated at.

        Returns: Void
        '''


        # Example input: 'log(x^2 + 5)', seed = a

        lefts_par = input_function.split('(')
        parsed_eq = []
        for rights_par in lefts_par:
            elements = rights_par.split(')')
            for el in elements:
                parsed_eq.append(el)
    
        center = math.floor(len(parsed_eq)/2)
        self.val = self.simplify(parsed_eq[center], a)
        self.der = self.simplify(parsed_eq[center], a, der=True)

        for func in parsed_eq[0:center]:
            self.val, self.der = self.derivative_dict(func) 
        # no return value 

    # Splits a coefficient
    def coef_split(self, x):
        split = x.split('*')
        if len(split) > 1:
            coef = float(split[0])
            power = split[1].split('^')
        else:
            coef = 1
            power = split[0].split('^')

        if len(power)>1:
            power = float(power[1])
        else:
            power = 1
        return coef, power

    # Evaluates an x term
    def coef_evaluate(self, x, a, der = False):
        '''
        Evaluates an x term with an optional coefficent and power. If der == True evaluates the derivative instead.

        Parameters:
        self(AutoDiffPy)
        x(str): A string representing an x term with an optional coefficient and power
        a(float/int): The value to evaluate x at
        der(bool): If True, the derivative value is evaluated using the power rule. 
        '''
        if not 'x' in x:
            if der == True:
                return 0
            else:
                return float(x)
        split = x.split('*')
        if len(split) > 1:
            coef = float(split[0])
            power = split[1].split('^')
        else:
            coef = 1
            power = split[0].split('^')

        if len(power)>1:
            power = float(power[1])
        else:
            power = 1
        if der == True:
            value = (coef * power)*(a**(power-1))
        else:
            value = (a**power) * coef
        return value

    # Performs the power rule on a string representing an x term
    def power_rule(self, x, realx):
        '''
        Params:
        x(str): A string representing an x term with a optional coefficient and power such as '3*x^5'
        realx(int/float): The seed value for the expression to be evaluated at. 
        
        Returns:
        Value after application of power rule
        '''
        split = x.split('*')
        coef = float(split[0])
        power = split[1].split('^')
        if len(power)>1:
            power = float(power[1])
        else:
            return coef * realx
        return (coef * power)*realx**(power-1)


    def simplify(self, x, seed = None, der = False): # where x is string of polynomial such as '3*x^2 * 5*x^3 + 5', if seed == True, we evaluate
        '''
        Simplifies a polynomial expression

        Parameters: 
        self(AutoDiffPy): An AutoDiffPy object
        x(str): A string representation of a polynomial expression
        seed(int/float): The value to evaluated at
        der (bool): If der == True, then it will both simplify the polynomial, and evaluate its derivative.

        Returns:
        runningval: The polynomial expression evaluted at the seed if seed != None
        OR
        simplified: The polynomial expression simplified and returned as a string  
        '''
        # split by addition/subtraction
        minus_split = x.split('-')
        plusminus_split = []    
        for i in range(len(minus_split)):
            plus_split = minus_split[i].split('+')
            for j in range(len(plus_split)):
                plusminus_split.append(plus_split[j])
                if j < len(plus_split) - 1:
                    plusminus_split.append('+')
            if i < len(minus_split) - 1:
                plusminus_split.append('-')

        final_array = []

        # split my multiplication/division
        for section in plusminus_split:
            bsymb = False
            multdiv_split = []

            # simplify sections
            mult_split = section.split(' * ')
            for i in range(len(mult_split)):
                div_split = mult_split[i].split('/')
                for j in range(len(div_split)):
                    multdiv_split.append(div_split[j])
                    if j < len(div_split) - 1:
                        multdiv_split.append('/')
                if i < len(mult_split) - 1:
                    multdiv_split.append('*')
            if len(multdiv_split) == 1:
                final_array.append(section)
            else:
                for i in range(len(multdiv_split)):
                    # Set symbol to false if index is at an operator
                    symbol = False
                    if '*' == multdiv_split[i]:
                        symbol = True
                        try:
                            formatted = False
                            if 'x' in multdiv_split[i-1]: 
                                print(multdiv_split[i-1], multdiv_split[i+1])
                                if 'x' in multdiv_split[i+1]:
                                    x1_coef, x1_power = self.coef_split(multdiv_split[i-1])
                                    x2_coef, x2_power = self.coef_split(multdiv_split[i+1])
                                    newval = str(x1_coef+x2_coef)+'*x^'+ str(x1_power+x2_power)
                                    formatted = True

                            if multdiv_split[i-1].isnumeric() and 'x' in multdiv_split[i+1]:
                                x2_coef, x2_power = self.coef_split(multdiv_split[i+1])
                                newval =  str(int(multdiv_split[i-1])*x2_coef) + '*x^' + str(x2_power)
                                formatted = True

                            if 'x' in multdiv_split[i-1] and multdiv_split[i+1].isnumeric():
                                x1_coef, x1_power = self.coef_split(multdiv_split[i-1])
                                newval = str(x1_coef) + '*x^' + str(int(multdiv_split[i+1])+x1_power)
                                formated = True
                            if formatted == False:
                                raise Exception('Incorrect format in multiplication')
                        except IndexError:
                            print("Loop ended")

                    elif '/' == multdiv_split[i]:
                        symbol = True
                        try:
                            if 'x' in multdiv_split[i-1] and 'x' in multdiv_split[i+1]:
                                x1_coef, x1_power = self.coef_split(multdiv_split[i-1])
                                x2_coef, x2_power = self.coef_split(multdiv_split[i+1])
                                newval =  str(x1_coef+x2_coef)+'*x^'+ str(x1_power+x2_power)

                        except IndexError:
                            raise Exception('Syntax error on input, hanging /')

                        if multdiv_split[i-1].isnumeric() and 'x' in multdiv_split[i+1]:
                            x2_coef, x2_power = self.coef_split(multdiv_split[i+1])
                            newval = str(int(multdiv_split[i-1])/x2_coef) + '*x^' + str(x2_power)

                        if 'x' in multdiv_split[i-1] and multdiv_split[i+1].isnumeric():
                            x1_coef, x1_power = self.coef_split(multdiv_split[i-1])
                            newval = str(x1_coef) + '*x^' + str(int(multdiv_split[i+1])-x1_power)

                        else:
                            raise Exception('Invalid syntax!!!')
                
                    if symbol == True:
                        bsymb = True
                        multdiv_split[i+1] = newval
                        final_array.append(newval)
        simplified = ''
        runningval = 0
        for i in range(len(final_array)):
            if seed != None:
                try:
                    if final_array[i] == '+':
                        try:
                            runningval += self.coef_evaluate(final_array[i+1], seed, der)
                        except IndexError:
                            print("Syntax error, hanging +")
                    if final_array[i] == '-':
                        try:
                            runningval -= self.coef_evaluate(final_array[i+1], seed, der)
                        except IndexError:
                            print("Syntax error, hanging -")

                    if i==0 and 'x' in final_array[i]:    
                        runningval = self.coef_evaluate(final_array[0], seed, der)

                except TypeError:
                    raise Exception('Seed in simplify-evaluate is not a valid number')
            else:
                simplified += final_array[i]
        if seed != None:
            return runningval
        else:
            return simplified


    


# Demo
a = 2.0 # value to evaluate f at

neweq = 'sin(x^5 + 5)'


MyDiff = AutoDiffPy(a, input_function=neweq)
print(MyDiff.der, "this is derivative")
print(MyDiff.val, "this is value")
