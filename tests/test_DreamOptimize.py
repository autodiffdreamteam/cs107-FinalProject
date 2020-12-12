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

# Test 1: Newton's method
def test_Newton():

    assert (round(Optimize.newtons_method('x^2 - 1', 0.25, 0.0001, max_iters=5000), 3) == 1.000)
    assert (round(Optimize.newtons_method('x^2 - 2', 2.0, 0.001, max_iters=500), 3) == round(math.sqrt(2),3))
    assert (round(Optimize.newtons_method('x^3 - 3*x + 1', 2.0, 0.001, max_iters=500), 3) == 1.532)
    
# Test 2: Gradient Descent
def test_GradDescent():

    assert (round(Optimize.grad_descent('x^2', 1.0, 0.0001, max_iters = 5000, eta = 0.5), 3) == 0.00)
    assert (round(Optimize.grad_descent('x^2 - 1', 1, 0.0001, max_iters=5000, eta = 0.5), 3) == round(0.000, 3))
    assert (round(Optimize.grad_descent('x^2 - 2', 2.0, 0.001, max_iters=5000, eta = 0.5), 3) == round(0.000,3))


# Test 3: Nesterov Gradient Descent
def test_nesterov_grad_descent():

    assert (round(Optimize.nesterov_grad_descent('x^2', 1.0, 0.0001, max_iters = 5000, eta = 0.5),3) == 0.00)
    assert (round(Optimize.nesterov_grad_descent('x^2 - 1', 1, 0.0001, max_iters=5000, eta = 0.5), 3) == round(0.000, 3))
    assert (round(Optimize.nesterov_grad_descent('x^2 - 2', 2.0, 0.001, max_iters=5000, eta = 0.5), 3) == round(0.000,3))

# Test 4: Quadratic Splines
def test_quadratic_spline():
    assert (int(abs(Optimize.quadratic_spline([2,3,4], [3,8,11])[0])) == 0)
    assert (int(Optimize.quadratic_spline([2,3,4,5,6], [3,8,11,14,18])[1]) == 5)
    
    with pytest.raises(Exception) as excinfo:
        Optimize.quadratic_spline([1,2], [10,15,20])
    assert "do not match" in str(excinfo.value)

    values = Optimize.QuadraticSpline([2,3,4,5,6], [-3,-8,-11,-14,-18])
    assert (int(values[0]*4 + values[1]*3 + values[2]) == -8)

# Test 5: Plot Coefficients
def test_plot_coeffs():
    with pytest.raises(Exception) as excinfo:
        Optimize.plot_coeffs([1,2], [10,15,20])
    assert "do not match" in str(excinfo.value)