#-------------------------------------------------------------------------
# This pytest script contains tests for the optimization algorithms in the
# script Optimize.py
#-------------------------------------------------------------------------


#Import the relevant classes and modules
from DreamDiff import DreamDiff as ad
from DreamDiff import Function as fun
from DreamOptimize import Optimize

import numpy as np
import pytest 
import math
import matplotlib.pyplot as plt

# Test 1: Newton's method
def test_Newton():

    assert (np.round(Optimize.newtons_method('x^2 - 1', 0.25, 0.0001, max_iters=5000)[0], 3) == 1.000)
    assert (np.round(Optimize.newtons_method('x^2 - 2', 2.0, 0.001, max_iters=500)[0], 3) == round(math.sqrt(2),3))
    assert (np.round(Optimize.newtons_method('x^3 - 3*x + 1', 2.0, 0.001, max_iters=500)[0], 3) == 1.532)
    
# Test 2: Gradient Descent
def test_GradDescent():

    assert (np.round(Optimize.grad_descent('x^2', 1.0, 0.0001, max_iters = 5000, eta = 0.5)[0], 3) == 0.00)
    assert (np.round(Optimize.grad_descent('x^2 - 1', 1, 0.0001, max_iters=5000, eta = 0.5)[0], 3) == round(0.000, 3))
    assert (np.round(Optimize.grad_descent('x^2 - 2', 2.0, 0.001, max_iters=5000, eta = 0.5)[0], 3) == round(0.000,3))


# Test 3: Nesterov Gradient Descent
def test_nesterov_grad_descent():

    assert (np.round(Optimize.nesterov_grad_descent('x^2', 1.0, 0.0001, max_iters = 5000, eta = 0.5)[0],3) == 0.00)
    assert (np.round(Optimize.nesterov_grad_descent('x^2 - 1', 1, 0.0001, max_iters=5000, eta = 0.5)[0], 3) == round(0.000, 3))
    assert (np.round(Optimize.nesterov_grad_descent('x^2 - 2', 2.0, 0.001, max_iters=5000, eta = 0.5)[0], 3) == round(0.000,3))

# Test 4: Quadratic Splines
def test_quadratic_spline():
    assert (int(abs(Optimize.quadratic_spline([2,3,4], [3,8,11])[0])) == 0)
    assert (int(Optimize.quadratic_spline([2,3,4,5,6], [3,8,11,14,18])[1]) == 5)
    
    with pytest.raises(Exception) as excinfo:
        Optimize.quadratic_spline([1,2], [10,15,20])
    assert "do not match" in str(excinfo.value)

    values = Optimize.quadratic_spline([2,3,4,5,6], [-3,-8,-11,-14,-18])
    assert (int(values[0]*4 + values[1]*3 + values[2]) == -8)

# Test 5: Plot Coefficients
def test_plot_coeffs():
    with pytest.raises(Exception) as excinfo:
        Optimize.plot_coeffs([-0.,  5.,  5., -0.,  5.,  5.], [1,2], [10,15,20])
    assert "do not match" in str(excinfo.value)

    #assert(type(Optimize.plot_coeffs([-0.,  5.,  5., -0.,  5.,  5.], [1,2,3], [10,15,20])) == plt)
    assert(Optimize.plot_coeffs([-0.,  5.,  5., -0.,  5.,  5.], [1,2,3], [10,15,20]) == True)

# Test 6: Animate Gradient Descent
def test_animate_gradient():
    f = 'tan(sin(x) + 3)'
    assert(Optimize.animate_grad_desc(f, 4, epsilon=0.00001, max_iters=500, eta=0.1, runtime=20, method='grad') == None)


# Test 7: Animate Newton's Method
def test_animate_newtons():
    f = 'x^3 - 3*x^2 + 4'
    assert(Optimize.animate_newtons(f, 0.3, epsilon=0.000001, max_iters=500, runtime=20) == None)
