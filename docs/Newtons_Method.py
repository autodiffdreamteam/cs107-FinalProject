#from AutoDiff import AutoDiffPy
import numpy as np
def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None



"Current issue: "
" All the example methods use a hard-coded derivative, which we could get for a specific value, but now we will need to save the derivative function itself in order for this to work"
" If we're using specific values for the seed, then this newton's method implementation would require the user to re-input their function every time"
" "



# convert exp(sin(x)) - cos(x^0.5)*sin(cos(x)^2 + x^2)^0.5)
# x = ad(2.0)
# to f = fun.exp(fun.sin(x)) - fun.cos(x**0.5)*fun.sin((fun.cos(x)**2 + x**2)**0.5)

string = 'exp(sin(x)) - cos(x^0.5)*sin(cos(x)^2 + x^2)^0.5)'
def string_convert(string): # if fun is a var name, add as parameter
    string = string
    func_dict = ['exp(', 'cos(', 'sin(', 'tan('] # etc
    expression = ''
    for i in range(len(string)):
        try:
            if string[i:i+4] in func_dict:
                expression +='fun.'
        except IndexError:
            pass
        
        if string[i] == '^':
            expression += '**'
        else:
            expression += string[i]
    return expression
            # add func to recognize parentheses
            # idea: Create a queue, such that you take off the queue once you find a right parentheses, and the "value" on that queue is the function

print(string_convert(string))
func = string_convert(string)
f = lambda x: eval(func)
print(f)

print(f(1))
# TRY THIS WITH BLAKE'S NEW CODE!!!

l_fun = lambda x: np.exp(np.sin(x)) - np.cos(x**0.5)*np.sin(np.cos(x)**2 + x**2)**0.5
