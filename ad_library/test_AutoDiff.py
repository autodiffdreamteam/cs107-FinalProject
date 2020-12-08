# Using pytest to test Autodiff implementation
import pytest
import numpy as np
import math
from AutoDiff import AutoDiff as ad
from Optimize import optimize as optim


#Test 1: Test the derivative and values of elementary function
#[e.g., sin(x), cos(x), tan(x), log(x), exp(x)]
def test_case_1():    
    a = math.pi/4.0    
    #function_list = [fun.sin(ad(a)), fun.cos(ad(a)), fun.tan(ad(a)),fun.log(ad(a)),fun.exp(ad(a))]
    #((np.arccos(a))**2)
    function_list = ['sin(x)', 'cos(x)', 'log(x)', 'exp(x)', 'tan(x)']
    value_list = [np.sin(a),np.cos(a),np.log(a),np.exp(a), np.tan(a)]
    der_list = [np.cos(a),-np.sin(a),1/a, np.exp(a), (np.arccos(a))**2]

    for l in range(0,1):
        returned_val = ad(a, input_function=function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))


#Test 2: Test the simple power functions
#[e.g., x, x^2]
def test_cast2():
    a = 4.0
    function_list = ['x', 'x^0.5', 'x^2', 'x^3']
    value_list = [4.0, 2.0, 16.0, 64.0]
    der_list = [1.0, 0.25, 8.0, 48.0]

    for l in range(0,4):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))


#Test 3: Test polynomials
def test_cast3():
    a = 1.0
    function_list = ['x+5','x^2+1', 'x^2+x+2', 'x^3+x^2+x+1']
    value_list = [6.0,2.0, 4.0, 4.0]
    der_list = [1.0,2.0, 3.0, 6.0]

    for l in range(0,4):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))


#Test 4: Test composites
def test_cast4():
    a = 0.0
    function_list = ['log(x+1)', 'cos(x^2)', 'sin(x^2)']
    value_list = [0.0, 1.0, 0.0]
    der_list = [1.0, 0.0,0.0]

    for l in range(0,2):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((round(returned_val.val,1) == value_list[l]) and (round(returned_val.der,1) == der_list[l]))


#Test 5: Test sum and difference of functions

def test_case5():
    a = math.pi/4.0
    function_list = ['sin(x) + cos(x)', 'sin(x) - cos(x)']
    value_list = [1.414, 0.0]
    der_list = [0.0, 1.414]

    for l in range(0,2):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((round(returned_val.val,3) == value_list[l]) and (round(returned_val.der,3) == der_list[l]))


#Test 6: Test product of functions
def test_case6():
    a = math.pi/4.0
    function_list = ['sin(x) * cos(x)', 'exp(x)*sin(x)']
    value_list = [math.sin(a)*math.cos(a), math.exp(a)*math.sin(a)]
    der_list = [-(math.sin(a)**2)+(math.cos(a)**2), math.exp(a)*(math.sin(a)+math.cos(a))]

    for l in range(0,2):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((round(returned_val.val,3) == round(value_list[l], 3)) and (round(returned_val.der,3) == round(der_list[l]),3))



#Test 7: Test negation of functions
def test_case7():
    a = math.pi/4.0
    function_list = ['-sin(x)', '-tan(x)', '-exp(x)']
    value_list = [-math.sin(a), -math.tan(a), -math.exp(a)]
    der_list = [-math.cos(a), -(np.arccos(a))**2, -math.exp(a)]

    for l in range(0,3):
        returned_val = ad(a, input_function = function_list[l])
        print(returned_val.val, returned_val.der)
        assert ((round(returned_val.val,3) == round(value_list[l], 3)) and (round(returned_val.der,3) == round(der_list[l]),3))



#Test 8: Test divison of functions
#def test_case8():
#    a = 1.0
#    function_list = ['1/x', '1/exp(x)']
#    value_list = [1/a, 1/math.exp(a)]
#    der_list = [-1/a**2, -1/math.exp(a)]

#    for l in range(0,2):
#        returned_val = ad(a, input_function = function_list[l])
#        print(returned_val.val, returned_val.der)
#        assert ((round(returned_val.val,3) == round(value_list[l], 3)) and (round(returned_val.der,3) == round(der_list[l]),3))





def test_case9():
    func_of_interest = ['x^2', 'x^2 + 1']
    starting_point = [-1.0, -2.0]
    optimum_point = [0.0, 0.0]

    for l in range(0, 2):
        returned_val = optim.graddescent(starting_point[l], func_of_interest[l])
        assert(round(returned_val,3) == round(optimum_point[l], 3))
    

    
