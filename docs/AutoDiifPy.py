import numpy as np
from math import log, sin, cos, exp, tan
import compiler
import parser
import math


class AutoDiffPy():

    def __init__(self, val, der=1):
        self.val = val
        self.der = der
    
    def __init__(self, input_function, seed):
        self.val = 0
        self.der = 0
        self.parse_input(input_function, seed)

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

    # Define a function value getter
    def val(self):
        return self.val

    # Define a function derivative getter
    def der(self):
        return self.der

    def sin(self, x):
        return AutoDiffPy(np.sin(x.val), np.cos(x.val)*x.der)

    def cos(self, x):
        return AutoDiffPy(np.cos(x.val), -np.sin(x.val)*x.der)

    def tan(self, x):
        return AutoDiffPy(np.tan(x.val), np.arctan(x.val)*x.der)

    def log(self, x):
        return AutoDiffPy(np.log(x.val), (1/x.val)*x.der)

    def exp(self, x):
        return AutoDiffPy(np.exp(x.val), (np.exp(x.val))*x.der)
    
    def derivative_dict(self, function):
        derivative_dict = {'sin': self.sin, 'cos': self.cos, 'tan': self.tan, 'log': self.log, 'exp': self.exp}
        return derivative_dict[function](self)

    def function_dict(self, function):
        derivative_dict = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'log': np.log, 'exp': np.exp}
        return derivative_dict[function](self)

    def parse_input(self, input_function, a): # figure out when this is CALLED!
        # Example input: 'log(x^2 + 5)', seed = a

        lefts_par = input_function.split('(')
        parsed_eq = []
        for rights_par in lefts_par:
            elements = rights_par.split(')')
            for el in elements:
                parsed_eq.append(el)
    
        center = math.ceil(len(parsed_eq)/2)
        self.val = simplify(parsed_eq[center], a)
        self.der = simplify(parsed_eq[center], a, der=True)
        "Implement a second option to run power rule on polynomials"
        for func in parsed_eq[0:center]:
            self = self.derivative_dict(func) # this is the derviative value # issue is it might reset the object???? prolly not tho
        # no return value 

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

    def coef_evaluate(self, x, a, der = False):
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

    def power_rule(self, x, realx): # x is a string like 3*x^2
        split = x.split('*')
        coef = float(split[0])
        power = split[1].split('^')
        if len(power)>1:
            power = float(power[1])
        else:
            return coef * realx
        return (coef * power)*realx**(power-1)

    def simplify(self, x, seed = None, der = False): # where x is string of polynomial such as '3*x^2 * 5*x^3 + 5', if seed == True, we evaluate
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
                    # check if symbol
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


    


        # first parse input into functions, polynomial
        # evaluate inner function
        # take center with x
            # use simplify-evaluate()
        # for func in center-left
            # self = self.derivative_dict(func) 


# Demo
a = 2.0 # value to evaluate f at
alpha = 2.0 # slope value
beta = 3.0 # intercept value

x = AutoDiffPy(a) 
x = AutoDiffPy.tan(x)
f2 = AutoDiffPy.sin(x)
#f = alpha * x + beta
print(f2.val, f2.der)



eq= "log(3x**2 *cos(sin(3*x + 5*x**2)**3))"

#def parse_input(input_function):  
    # parse by parentheses
#    lefts_par = input_function.split('(')
#    parsed_eq = []
#    for rights_par in lefts_par:
#        elements = rights_par.split(')')
#        for el in elements:
#            parsed_eq.append(el)  
#    center = math.ceil(len(parsed_eq)/2)
#    simplified_poly = simplify(parsed_eq[center])
#    "first place in the center, simplify, and will incorporate power rule into simplify if not already implemented by Blake"
#    "Will I need to create an evaluate function to plug in seed into all x values -- might need to create parameters in simplify to include evaluate derivative, evaluate reg val"
    # feed input to self for derivative and value operations
    #for i in range(len(parsed_eq)/2):
    #    self.derivative(parsed_eq[center-1])
#    return parsed_eq    
#print(parse_input(eq))
#parsed = parse_input(eq)
#for el in parsed:
#    if 'x' in el:
#        center = el
#print(center)
#center_index = math.ceil(len(parsed)/2)
#print(center_index)
#num_funcs = (math.floor(len(parsed)/2))
#print(num_funcs)

 
# sec and arctan are NOT THE SAME

# did sin(tan(x)), which means we do everything in inner order
"CORRECT ORDER"
# derivative of 3*x + derivative of 5x^2
# derivative of sin(inner)
# derivative of cos(inner)
# derivative of 3x^2 * inner
# derivative of log(inner)


def coef_split(x):
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

def coef_evaluate(self, x, a, der = False):
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

def power_rule(self, x, realx): # x is a string like 3*x^2
    split = x.split('*')
    coef = float(split[0])
    power = split[1].split('^')
    if len(power)>1:
        power = float(power[1])
    else:
        return coef * realx
    return (coef * power)*realx**(power-1)

def simplify(self, x, seed = None, der = False): # where x is string of polynomial such as '3*x^2 * 5*x^3 + 5', if seed == True, we evaluate
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
                # check if symbol
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


neweq = '3*x^5 + 5'
neweq = '3*x^5 * 2*x^4 + 5*x^2 - 3*x^2 * x'
print(simplify(neweq))


value = power_rule('3*x^2', 2)
print(value)

    #return AutoDiffPy(c*(a) * x**(a-1) )


# f1 = 3x + 5x^2
# f2 = sin(f1)
# f3 = evaluate(f2)**3
# f4= cos(f3)
# derivative = log(f4) 





# f1 = 3x + 5x^2  /// val: 3*seed, + 5*seed^2       /// der: power_rule(3*x) + power_rule(5*seed^2)
# f2 = sin(f1)    ///  val: sin(f1)                 /// der: der(sin(f1))
# f3 = evaluate(f2)**3  /// val: f3**2              /// **2 ? Would we use power rule here?
# f4= cos(f3)           /// val: cos(f3)            /// der: der(cos(f3))     
# derivative = log(f4)  /// log(f4)                 /// der: der(log(f4))



#def sequenced_input(parsed_eq):
    