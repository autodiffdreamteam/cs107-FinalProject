from AutoDiff import AutoDiff as ad
from Function import Function as fun
import numpy as np

# Demo with f(x) = exp(sin(x))-cos(x^0.5)*sin((cos(x)^2+x^2)^0.5)

x = ad(2.0)
f = fun.exp(fun.sin(x)) - fun.cos(x**0.5)*fun.sin((fun.cos(x)**2 + x**2)**0.5)
print(f.val, f.der)

# use option 2, string expression input

input_function = 'exp(sin(x)) - cos(x^0.5)*sin((cos(x)^2 + x^2)^0.5)'
f2 = ad(2.0, input_function= input_function)
print(f2.val, f2.der)

# newton's method demo

input_2 = 'x^2 - x  - 1'
f3 = ad(1, input_function=input_2)
print(f3.newtons_method(1, 1e-8, 10))

print(np.zeros((2, 2)))