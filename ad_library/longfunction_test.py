from AutoDiff import AutoDiff as ad
from Function import Function as fun

# Demo with f(x) = exp(sin(x))-cos(x^0.5)*sin((cos(x)^2+x^2)^0.5)

#x = ad(2.0)
#f = fun.exp(fun.sin(x)) - fun.cos(x**0.5)*fun.sin((fun.cos(x)**2 + x**2)**0.5)
#print(f.val, f.der)

## DEMOS

# Scalar input with a scalar function
x = ad(3.0)
f = x**2
print('Demo for scalar input with a scalar function:')
print(f)

# Vector input with a scalar function
x = ad(2.0, [1, 0])
y = ad(3.0, [0, 1])
f = 2*x + y
print('Demo for a vector input with a scalar function:')
print(f)

# Scalar input with a vector function
x = ad(3.0)
f1 = x**2
f2 = 5 + x
f3 = x - 4
f = ad([f1, f2, f3])
print('Demo for a scalar input with a vector function:')
print(f)

# Vector input with a vector function
x = ad(2.0, [1, 0])
y = ad(3.0, [0, 1])
f1 = x + y*x
f2 = y**2
f = ad([f1, f2])
print('Demo for a vector input with a vector function:')
print(f)

