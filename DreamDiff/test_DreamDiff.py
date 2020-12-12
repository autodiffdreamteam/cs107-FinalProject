# Test suite for DreamDiff using pytest

import pytest
import numpy as np

#from DreamDiff import DreamDiff as ad
#from DreamDiff import Function as fun
from DreamDiff.DreamDiff import DreamDiff as ad
from DreamDiff.DreamDiff import Function as fun


# Test initialization

def test_scalar_input():
    x = ad(2.0)
    assert x.val == [2]
    assert x.der == [1]

def test_vector_input():
    x = ad(2.0, [1, 0])
    assert x.val == [2]
    assert np.all(x.der == np.array([1, 0]))
   
def test_vector_function():
    x = ad(2.0, [1, 0])
    y = ad(3.0, [0, 1])
    f = ad([x, y])
    assert np.all(np.ravel(f.val) == np.array([2, 3]))
    assert np.all(f.der == np.array([[1, 0], [0, 1]]))

def test_get_all_scalar():
    x = ad(2.0)
    var_list = [1, 2, 3]
    assert x._check_all_scalar(var_list) == True

def test_get_total_vars():
    x = ad(2.0)
    y = ad(3.0, [1, 0])
    z = ad(4.0, [0, 1])
    var_list = [y, z]
    assert x._get_total_vars(var_list) == 2


# Test basic operations

def test_add():
    x = ad(2.0)
    f = x + 3
    assert f.val == [5]
    assert f.der == [1]
    
    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = x1 + y1
    assert f1.val == [6]
    assert np.all(f1.der == np.array([1, 1]))

def test_radd():
    x = ad(3.0)
    f = 2 + x
    assert f.val == [5]
    assert f.der == [1]
    
    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = y1 + x1
    assert f1.val == [6]
    assert np.all(f1.der == np.array([1, 1]))

def test_sub():
    x = ad(2.0)
    f = x - 3
    assert f.val == [-1]
    assert f.der == [1]

    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = x1 - y1
    assert f1.val == -2
    assert np.all(f1.der == np.array([1, -1]))

def test_rsub():
    x = ad(3.0)
    f = 2 - x
    assert f.val == [-1]
    assert f.der == [-1]

    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = y1 - x1
    assert np.all(f1.der == np.array([-1, 1]))

def test_mul():
    x = ad(2.0)
    f = x*3
    assert f.val == [6]
    assert f.der == [3]
    
    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = x1*y1
    assert f1.val == [8]
    assert np.all(f1.der == np.array([4, 2]))

def test_rmul():
    x = ad(3.0)
    f = 2*x
    assert f.val == [6]
    assert f.der == [2]

    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = y1*x1
    assert f1.val == 8
    assert np.all(f1.der == np.array([4, 2]))

def test_truediv():
    x = ad(2.0)
    f = x / 4
    assert f.val == [0.5]
    assert f.der == [0.25]

    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = x1 / y1
    assert f1.val == [0.5]
    assert np.all(f1.der == np.array([0.25, -0.125]))
   
def test_rtruediv():
    x = ad(2.0)
    f = 1 / x
    assert f.val == [0.5]
    assert f.der == [-0.25]

    x1 = ad(2.0, [1, 0])
    y1 = ad(4.0, [0, 1])
    f1 = y1 / x1
    assert f1.val == [2]
    assert np.all(f1.der == np.array([-1, 0.5]))

def test_pow():
    x = ad(2.0)
    f = x**3
    assert f.val == [8]
    assert f.der == [12]

    x1 = ad(2.0, [1, 0])
    y1 = ad(3.0, [0, 1])
    f1 = x1**y1
    assert f1.val == [8]
    assert np.all(np.round(f1.der, 2) == np.array([12, 5.55]))

def test_rpow():
    x = ad(3.0)
    f = 2**x
    assert f.val == [8]
    assert np.round(f.der, 2) == [5.55]

    x1 = ad(2.0, [1, 0])
    y1 = ad(3.0, [0, 1])
    f1 = y1**x1
    assert f1.val == [9]
    assert np.all(np.round(f1.der, 2) == np.array([9.89, 6]))

def test_neg():
    x = ad(2.0)
    f = -x
    assert f.val == [-2]
    assert f.der == [-1]

    x1 = ad(2.0, [1, 0])
    y1 = ad(3.0, [0, 1])
    f1 = ad([-x1, -y1])
    assert np.all(np.ravel(f1.val) == np.array([-2, -3]))
    assert np.all(f1.der == np.array([[-1, 0], [0, -1]]))

def test_pos():
    x = ad(2.0)
    f = +x
    assert f.val == [2]
    assert f.der == [1]

def test_abs():
    x = ad(-3.0)
    f = abs(x)
    assert f.val == [3]
    assert f.der == [-1]


# Test comparison operators

def test_eq():
    x = ad(2.0)
    y = ad(2.0)
    z = ad(3.0)
    assert x == y
    assert not (y == z)

def test_ne():
    x = ad(2.0)
    y = ad(2.0)
    z = ad(3.0)
    assert x != z
    assert not (x != y)

def test_lt():
    x = ad(2.0)
    y = ad(3.0)
    assert x < 3
    assert x < y

def test_gt():
    x = ad(3.0)
    y = ad(2.0)
    assert x > 2
    assert x > y

def test_le():
    x = ad(2.0)
    y = ad(3.0)
    assert x <= 2
    assert x <= 3
    assert x <= y

def test_ge():
    x = ad(3.0)
    y = ad(2.0)
    assert x >= 3
    assert x >= 2
    assert x >= y


# Test elementary functions

def test_sin():
    x = ad(np.pi / 2)
    f = fun.sin(x)
    assert f.val == [1]
    assert np.round(f.der, 2) == [0]
   
def test_cos():
    x = ad(np.pi)
    f = fun.cos(x)
    assert f.val == [-1]
    assert np.round(f.der, 2) == [0]

def test_tan():
    x = ad(1.0)
    f = fun.tan(x)
    assert np.round(f.val, 2) == [1.56]
    assert np.round(f.der, 2) == [3.43]

def test_arcsin():
    x = ad(0.5)
    f = fun.arcsin(x)
    assert np.round(f.val, 2) == [0.52]
    assert np.round(f.der, 2) == [1.15]

def test_arccos():
    x = ad(0)
    f = fun.arccos(x)
    assert np.round(f.val, 2) == [1.57]
    assert f.der == [-1]

def test_arctan():
    x = ad(2.0)
    f = fun.arctan(x)
    assert np.round(f.val, 2) == [1.11]
    assert f.der == [0.2]

def test_sinh():
    x = ad(1.0)
    f = fun.sinh(x)
    assert np.round(f.val, 2) == [1.18]
    assert np.round(f.der, 2) == [1.54]

def test_cosh():
    x = ad(0)
    f = fun.cosh(x)
    assert f.val == [1]
    assert f.der == [0]

def test_tanh():
    x = ad(2.0)
    f = fun.tanh(x)
    assert np.round(f.val, 2) == [0.96]
    assert np.round(f.der, 2) == [0.07]

def test_sqrt():
    x = ad(2.0)
    f = fun.sqrt(x)
    assert np.round(f.val, 2) == [1.41]
    assert np.round(f.der, 2) == [0.35]

def test_exp():
    x = ad(2.0)
    f = fun.exp(x)
    assert np.round(f.val, 2) == [7.39]
    assert np.round(f.der, 2) == [7.39]

def test_log():
    x = ad(np.e)
    f = fun.log(x)
    assert f.val == [1]
    assert np.round(f.der, 2) == [0.37]

def test_log2():
    x = ad(2.0)
    f = fun.log2(x)
    assert f.val == [1]
    assert np.round(f.der, 2) == [0.72]

def test_log10():
    x = ad(10.0)
    f = fun.log10(x)
    assert f.val == [1]
    assert np.round(f.der, 2) == [0.04]

def test_logistic():
    x = ad(2.0)
    f = fun.logistic(x)
    assert np.round(f.val, 2) == [0.88]
    assert np.round(f.der, 2) == [0.10]


# Extra tests

def test_str():
    x = ad(2.0)
    assert x.__str__() == 'Values:\n{}\nJacobian:\n{}'.format(np.array([2.0]), [1])


def test_vector_input_scalar_function():
    x = ad(2.0, [1, 0])
    y = ad(3.0, [0, 1])
    f = 2*x + y
    assert f.val == 7
    assert np.all(np.ravel(f.der) == np.array([2, 1]))

def test_scalar_input_vector_function():
    x = ad(3.0, input_pos=[3, 0])
    f1 = x**2
    f2 = 5 + x
    f3 = x - 4
    f = ad([f1, f2, f3])
    assert np.all(np.ravel(f.val) == np.array([9, 8, -1]))
    assert np.all(f.der == np.array([[6, 0, 0], [1, 0, 0], [1, 0, 0]]))

def test_vector_input_vector_function():
    x = ad(2.0, [1, 0])
    y = ad(3.0, [0, 1])
    f1 = x + y*x
    f2 = y**2
    f = ad([f1, f2])
    assert np.all(np.ravel(f.val) == np.array([8, 9]))
    assert np.all(f.der == np.array([[4, 2], [0, 6]]))

