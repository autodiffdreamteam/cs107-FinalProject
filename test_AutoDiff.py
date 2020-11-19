from AutoDiff import AutoDiff as ad
from Function import Function as fun

# Demo for f = cos(sin(x)) at x=2.0
a = 2.0
f = ad(a)
f = fun.sin(f)
f = fun.cos(f)

print(f.val, f.der)
