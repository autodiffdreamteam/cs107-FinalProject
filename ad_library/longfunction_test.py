from AutoDiff import AutoDiff as ad
from Function import Function as fun

# Demo with f(x) = exp(sin(x))-cos(x^0.5)*sin((cos(x)^2+x^2)^0.5)

x = ad(2.0)
f = fun.exp(fun.sin(x)) - fun.cos(x**0.5)*fun.sin((fun.cos(x)**2 + x**2)**0.5)
print(f.val, f.der)
