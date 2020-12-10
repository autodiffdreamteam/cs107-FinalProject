# Using pytest to test Autodiff implementation
import pytest
import numpy as np
import math
import numpy
from AutoDiff import AutoDiff as ad
from AutoDiff import Function as fun
#from Optimize import optimize as optim


# Scalar input with scalar function
def test_case_1():    
    a = 1.0   
    x = ad(a)
    f1 = x**2
    value = a**2
    der_value = 2*a
    print(f1)
    assert (f1.val == value) and (f1.der == der_value)


# Vector input with a scalar function
def test_case_2():
    
    x = ad(2.0, [1, 0])
    y = ad(3.0, [0, 1])

    f = 2*x + y
    value = 2*2.0 + 3.0
    value_derx = 2.0
    value_dery = 1.0
    #print(f)
    assert (f.val == value) and (f.der[0] == value_derx) and (f.der[1] == value_dery)


# Scalar input with a vector function
def test_case_3():

    x = ad(3.0, input_pos=[3, 0])
    f1 = x**2
    f2 = 5 + x
    f3 = x - 4
    f = ad([f1, f2, f3])
    f1value = 9.0
    f2value = 8.0
    f3value = -1.0
    jacobian_list = numpy.array([[6.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    test = f.der == jacobian_list
    assert (f.val[0] == f1value) and (f.val[1] == f2value) and (f.val[2] == f3value) and (test.all())



# Vector input with a vector function
def test_case_4():

    x = ad(2.0, [1, 0])
    y = ad(3.0, [0, 1])
    f1 = x + y*x
    f2 = y**2
    f = ad([f1, f2])

    f1value = 8.0
    f2value = 9.0
    jacobian_list = numpy.array([[4.0, 2.0], [0.0, 6.0]])
    test = f.der == jacobian_list
    assert (f.val[0] == f1value) and (f.val[1] == f2value) and (test.all())
    


def test_case_5():

    a = 1.0   
    x = ad(a)
    f1 = 1/x
    value = 1/a
    der_value = -1/a**2
    #print(f1)
    assert (f1.val == value) and (f1.der == der_value)
    
    
    

def test_case_6():

    a = math.pi/2.0
    x = ad(a)
    f = fun.sin(x)
    value = numpy.sin(a)
    der_value = numpy.cos(a)
    assert (f.val == value) and (f.der == der_value)


def test_case_7():

    a = math.pi/2.0
    x = ad(a)
    f = fun.cos(x)
    value = numpy.cos(a)
    der_value = -numpy.sin(a)
    assert (f.val == value) and (f.der == der_value)

def test_case_8():

    a = math.pi/4.0
    x = ad(a)
    f = fun.tan(x)
    value = numpy.tan(a)
    der_value = numpy.arctan(a)
    assert (f.val == value) and (f.der == der_value)
    

def test_case_9():

    a = 1.0
    x = ad(a)
    f = fun.log(x)
    value = numpy.log(a)
    der_value = 1/a
    assert (f.val == value) and (f.der == der_value)


def test_case_10():

    a = 0.0
    x = ad(a)
    f = fun.exp(x)
    value = numpy.exp(a)
    der_value = numpy.exp(a)
    assert (f.val == value) and (f.der == der_value)

