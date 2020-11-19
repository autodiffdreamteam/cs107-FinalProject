# Using pytest to test Autodiff implementation
import pytest
import numpy as np
import math
from AutoDiff import AutoDiff as ad
from Function import Function as fun

#Test 1: Test the derivative and values of elementary function
#[e.g., sin(x), cos(x), tan(x), log(x), exp(x)]

def test_case_1():
    
    a = math.pi/4.0    
    function_list = [fun.sin(ad(a)), fun.cos(ad(a)), fun.tan(ad(a)),fun.log(ad(a)),fun.exp(ad(a))]
    value_list = [np.sin(a),np.cos(a),np.tan(a),np.log(a),np.exp(a)]
    der_list = [np.cos(a),-np.sin(a),((np.arccos(a))**2),1/a, np.exp(a)]

    for l in range(0,3):
        returned_val = function_list[l]
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))


#Test 2: Test the derivative and values of composite functions (f(g(x))
#[e.g. sin(cos(x)), cos(sin(x)), sin(tan(x)), cos(tan(x)), log(sin(x), log(cos(x)), log(tan(x))]


#Test 2: Test the derivative and values of product of functions (f(x) * g(x))
#[e.g. sin(x)*cos(x), sin(x)*tan(x), cos(x)*tan(x), sin(x)*log(x), cos(x)*log(x), 
def test_case_2():
    a = math.pi/3.0    
    #function_list = [fun.sin(ad(a)), fun.cos(ad(a)), fun.tan(ad(a)),fun.log(ad(a)),fun.exp(ad(a))]
    function_list = [fun.sin(ad(a))*fun.cos(ad(a)), fun.sin(ad(a))*fun.tan(ad(a)), fun.cos(ad(a))*fun.tan(ad(a)),
                     fun.sin(ad(a))*fun.log(ad(a)), fun.cos(ad(a))*fun.log(ad(a)), fun.sin(ad(a))*fun.exp(ad(a)),
                     fun.cos(ad(a))*fun.exp(ad(a))]
    value_list = [np.sin(a)*np.cos(a), np.sin(a)*np.tan(a), np.cos(a)*np.tan(a), np.sin(a)*np.log(a), np.cos(a)*np.log(a),
                  np.sin(a)*np.exp(a), np.cos(a)*np.exp(a)]
    der_list = [np.sin(a)*(-np.cos(a)) + np.cos(a)*np.cos(a), np.sin(a)*((np.arccos(a))**2) + np.cos(a)*np.tan(a),
                np.cos(a)*((np.arccos(a))**2) - np.sin(a)*np.tan(a), np.sin(a)*(1/a) + np.log(a)*np.cos(a),
                np.cos(a)*(1/a) - np.sin(a)* np.log(a), np.sin(a)*np.exp(a) + np.cos(a)*np.exp(a),
                np.cos(a)*np.exp(a) - np.sin(a)*np.exp(a)]
    #value_list = [np.sin(a),np.cos(a),np.tan(a),np.log(a),np.exp(a)]
    #der_list = [np.cos(a),-np.sin(a),((np.arccos(a))**2),1/a, np.exp(a)]

    for l in range(0,7):
        returned_val = function_list[l]
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))


#Test 3: Test the derivative and values of sum of functions (f(x) + g(x))
#[e.g. sin(x) + cos(x), sin(x) + log(x), sin(x) + exp(x)]
def test_case_3():
    a = math.pi/4.0    
    function_list = [fun.sin(ad(a))+ fun.cos(ad(a)), fun.sin(ad(a)) + fun.log(ad(a)), fun.sin(ad(a)) + fun.exp(ad(a))]
    value_list = [np.sin(a) + np.cos(a), np.sin(a) + np.log(a), np.sin(a) + np.exp(a)]
    der_list = [np.cos(a) - np.sin(a),  np.cos(a) + 1/a, np.cos(a) + np.exp(a)]
    

    for l in range(0,3):
        returned_val = function_list[l]
        print(returned_val.val, returned_val.der)
        assert ((returned_val.val == value_list[l]) and (returned_val.der == der_list[l]))



