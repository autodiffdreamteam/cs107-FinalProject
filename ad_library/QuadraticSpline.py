import numpy as np
from AutoDiffBlend import AutoDiff as ad
import matplotlib.pyplot as plt


def Spline(x, y, n):

#    x = [1, 2, 3, 4, 5]
#    y = [3, 6, 9, 12, 15]
#    n = 4

    # uncomment for user input
    # x = []; y = []
    # n = int(input('Enter n: '))
    # for index in range(n):
    # 	x.append(int(input('Enter x_{}: '.format(index))))
    # 	y.append(int(input('Enter y_{}: '.format(index))))
    c = list(y)
    a = []
    b = []
    for index in range((n - 1), -1, -1):
        a.append(c[index] - c[index - 1])
    for index in range((n - 1), -1, -1):
        b.append(2 * c[index - 1] - 2 * c[index])
    a = a[::-1]
    b = b[::-1]
    b[-1] = 0
    print(a)
    print(b)
    for index in range(n):
        print(
            "S_{}(x) = {} + {}x + {}x^2 for x ∈ [{}, {}]".format(
                index, y[index], b[index], a[index], x[index], x[index + 1]
            )
        )

x = [1, 2, 3, 4, 5]
y = [3, 6, 9, 12, 15]
n = 2

Spline(x, y, n)


# Calculate the coefficients of the quadratic functions
def quad_spline_coeff(f, xMin, xMax, nIntervals):
    """ Constructs the matrix for quadratic spline calculation
        and returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        xMin: float
              left endpoint of the :math:`x` interval
        
        xMax: float
              right endpoint of the :math:`x` interval
        
        nIntervals: integer
                    number of intervals that you want to slice the original function
            
        Returns
        -------
        y: list of floats
           the right hand side of Ax=y
        
        A: numpy.ndarray
            the sqaure matrix in the left hand side of Ax=y
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        """
    
    h = 1/nIntervals
    ks = []
    for i in np.linspace(xMin, xMax, nIntervals+1):
        k = ad(i)
        ks.append(k)
    
    # Construct the quadratic functions
    def a(ad_object):
        return ad_object ** 2
    def b(ad_object):
        return ad_object
    def c(ad_object):
        return ad(1)
    
    # Construct y
    y = []
    for i in range(nIntervals):
        y.append(f(ks[i]).val)
        y.append(f(ks[i+1]).val)
    for i in range(nIntervals):
        y.append([0])
    y = np.vstack(y)
    
    # Construct A
    A = np.zeros((3*nIntervals, 3*nIntervals))
    # Constraint 1:
    for i in range(nIntervals):
        A[2*i, 3*i] = a(ks[i]).val
        A[2*i, 3*i+1] = b(ks[i]).val
        A[2*i, 3*i+2] = c(ks[i]).val
        A[2*i+1, 3*i] = a(ks[i+1]).val
        A[2*i+1, 3*i+1] = b(ks[i+1]).val
        A[2*i+1, 3*i+2] = c(ks[i+1]).val
    # Constraint 2:
    for i in range(nIntervals-1):
        A[2*nIntervals+i, 3*i] = a(ks[i+1]).der
        A[2*nIntervals+i, 3*i+1] = b(ks[i+1]).der
        A[2*nIntervals+i, 3*i+3] = -1*a(ks[i+1]).der
        A[2*nIntervals+i, 3*i+4] = -1*b(ks[i+1]).der
    # Constraint 3:
    A[3*nIntervals-1, 1] = 10*b(ks[0]).der
    A[3*nIntervals-1, -3] = -1*a(ks[-1]).der
    A[3*nIntervals-1, -2] = -1*b(ks[-1]).der
    
    coeffs = np.linalg.solve(A, y)
    
    return y, A, coeffs, ks

# Get the positions of the spline points
def spline_points(f, coeffs, ks, nSplinePoints):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
            function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        nSplinePoints: integer
                       number of points to draw each spline
        
        Returns
        -------
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        """
    
    spline_points = []

    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, nSplinePoints)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])

    return spline_points

# Plot the spline and the orignal function
def quad_spline_plot(f, coeffs, ks, nSplinePoints):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        nSplinePoints: integer
                       number of points to draw each spline
        
        Returns
        -------
        fig: matplotlib.figure
             the plot of :math:`f(x)` and splines
        
        """
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    # Plot the original function
    fx = []
    fy = []
    for k in ks:
        fx.append(k.val)
        fy.append(f(k).val)
    ax.plot(fx, fy, 'o-', linewidth=2, label='original')
    
    spline_points = []
    # Plot the splines
    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, nSplinePoints)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])
        ax.plot(sx, sy, label=r'$s_{%s}(x)$' % i)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    box = ax.get_position()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.show()

    return fig

# Calculate spline squared error
def spline_error(f, spline_points):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        Returns
        -------
        error: float
               average absolute error of the spline and the original function on one given interval
        
        """
    
    error = 0
    for spline_point in spline_points:
        xs = spline_point[0]
        original_ys = []
        for x in xs:
            original_y = f(ad(x)).val
            original_ys.append(original_y)
        
        error += abs(sum((np.hstack(original_ys) - spline_point[1])) / len(spline_point[0])) ** 1
    
    return error / len(spline_points)



### BEGIN DEMO ###



def test_spline():
    
    def f1(var):
        return var**2

    xMin1 = -1
    xMax1 = 100
    nIntervals1 = 4
    nSplinePoints1 = 5

    y1, A1, coeffs1, ks1 = quad_spline_coeff(f1, xMin1, xMax1, nIntervals1)
    fig1 = quad_spline_plot(f1, coeffs1, ks1, nSplinePoints1)
    spline_points1 = spline_points(f1, coeffs1, ks1, nSplinePoints1)
    error = spline_error(f1, spline_points1)

    print("y1, A1, coeffs1, ks1: ", y1, A1, coeffs1, ks1)
    print("Spline points:", spline_points1)
    print("this is the error:", error)
    
    fig1.show()
    

print ("Testing spline suite.")
test_spline()
print ("All tests passed!")
