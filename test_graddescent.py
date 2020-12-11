#test_graddescent
from AutoDiff_CarlosFinal import AutoDiff as ad

x = ad(1.5)
returned_val = x.graddescent(1.5, 'x^2')

print(returned_val)
