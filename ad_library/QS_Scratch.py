import numpy as np
from AutoDiffBlend import AutoDiff





def QuadraticSpline(x_array, y_array):
    pass
    # n + 1 data points
    # n splines
    # 3n unknown variables
    # 3n quadratic equations! 
        # system of equations
        # a1x^2 + b1x + c1 = y_i
        # a1x^2 + b1x + c1 = y_i+1
        #  d(a1x^2 + b1x + c1) = d(a2x^2 + b2x + c2) evaluated at x_1
        # == 2a1x + b1 = 2a2x + b2 evaluated at x_1
        # == 2a1x1 + b1 = 2a2x1 + b2
        # 2a1x1 + b1 - 2a2z1 - b2 = 0 ---> n-1 of these equations

    if not len(x_array == len(y_array)):
        raise Exception('number of x values for data points do not match number of y values')
    
    if len(x_array)<3:
        raise Exception('Too few data points for quadratic spline interpolation')
    n = len(x_array) - 1
    # 2n equations
    sys_of_equations = []

    # define AutoDiff objects
    def a(ad):
        return ad ** 2
    def b(ad):
        return ad
    def c(ad):
        return ad(1)
    

    for i in range(n):
        sys_of_equations.append([])
        # deal with solving system of equations either using jacobian, or using the sys solver? Only problem w/ sys solver is it doesn't really even use your own package
    
    # create right hand side of linear equations
    y = []
    for i in len(n):
        y.append(y_array[i])
        y.append(y_array[i+1])
    for i in len(n):
        y.append(0)

    coeffs = np.linalg.solve(A, y_array)





    " Ensure > 2 data points"
    " Take as input 2 arrays of n+1 data points"
    " 2n + n-1 equations == "
    " last constraint is a1 = 0"