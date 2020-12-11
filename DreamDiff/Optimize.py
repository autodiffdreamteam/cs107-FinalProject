from DreamDiff import DreamDiff as ad
from DreamDiff import Function as fun
import numpy as np
import matplotlib.pyplot as plt


class Optimize:
    '''
    Class containing a suite of root-finding, optimization, and
    animation tools for Newton's method, gradient descent, and 
    Nesterov's accelerated gradient descent.
    '''
    
    def newtons_method(f, x0, epsilon=0.000001, max_iters=500):
        '''
        Implements Newton's root-finding method for scalar functions.

        INPUTS
        ======
        f: str, required
            Input function on which to perform Newton's method.
        x0: int or float, required
            Starting x-value at which to initialize the algorithm.
        epsilon: int or float, optional
            Solution accuracy threshold.
        max_iters: int, optional
            The maximum number of times to run the algorithm.

        RETURNS
        =======
        If a root is found, the algorithm returns its x-value, a list
        containing the previous points at which the derivative was 
        evaluated, a list of the function's values at those points, and
        a list of the function's derivatives at those points.
        If no root is found or a zero derivative is reacahed, None is 
        returned.
        '''
        # Initialize xn at the starting point
        xn = x0

        # Create a DreamDiff object to access private methods
        x = ad(1.0) 

        # Parse the input function string
        f_parsed = x._parse_input(f)

        # Initialize empty lists to hold results
        xn_history = []
        yn_history = []
        der_history = []

        # Run the algorithm at most 'max_iters' times
        for i in range(max_iters):

            # Calculate values, derivatives, and store in lists
            xn_history.append(xn)
            new_f = x._evaluate_function(f_parsed, xn)
            yn = new_f.val[0]
            der = new_f.der[0]
            yn_history.append(yn)
            der_history.append(der)

            # If threshold is met, terminate the algorithm and return results
            if abs(yn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(xn))
                return xn, xn_history, yn_history, der_history
            
            # If a zero derivative is reached, terminate the algorithm and
            # return None
            if der == 0:
                print('Reached zero derivative. No solution found.')
                return None

            # Apply the Newton's method update to xn
            xn = xn - yn/der

        # If no solution is found within 'max_iters', tell the user
        print('No solution found after max iterations.')


    def animate_newtons(f, x0, epsilon=0.000001, max_iters=500, runtime=20): 
        '''
        Creates an animation of Newton's root-finding method for scalar
        functions by plotting the function and showing the tangent lines
        at which the gradient was evaluated.

        INPUTS
        ======
        f: str, required
            Input function on which to perform Newton's method.
        x0: int or float, required
            Starting x-value at which to initialize the algorithm.
        epsilon: int or float, optional
            Solution accuracy threshold.
        max_iters: int, optional
            The maximum number of times to run the algorithm.
        runtime: int, optional
            The length of time to run the animation.

        RETURNS
        =======
        A matplotlib animataion showing our Newton's method located
        a root of the input function.
        '''
        # Get the root and history of x-values, y-values, and derivatives 
        # for the input function using newtons_method
        root, x_vals, y_vals, grad_vals = Optimize.newtons_method(f, x0, epsilon, max_iters)

        # Create a DreamDiff object to access private methods
        x = ad(1.0)

        # Create a list of x-vals searched
        t_vals = np.arange(min(x_vals)-2, max(x_vals)+2, 0.1)
        
        # Parse the input function string
        f_parsed = x._parse_input(f)

        # Calculate the values of the input function in the range of x-vals searched
        f_vals = [x._evaluate_function(f_parsed, t).val[0] for t in t_vals]
      
        # Initialize the animation
        fig = plt.figure()
        fig.canvas.draw()
        fig.suptitle("Newton's Method for f(x)="+f)

        # Run the animation for the duration of 'runtime'
        while runtime > 0:

            # Plot the animation, where each frame shows the function and
            # tangent line at which the gradient was evaluated
            for i in range(len(x_vals)):
                grad = grad_vals[i]
                intercept = y_vals[i] - grad*x_vals[i]
                tangent_y = grad*t_vals + intercept
                plt.cla()
                plt.grid(True)
                plt.plot(t_vals, f_vals, label='Function')
                plt.plot(t_vals, tangent_y, '-r', label='Tangent')
                plt.xlim(t_vals[0], t_vals[-1])
                plt.ylim(min(f_vals), max(f_vals))
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                plt.pause(0.2)
                runtime -= 1


    def grad_descent(f, x0, epsilon=0.000001, max_iters=500, eta=0.1):
        '''
        Implements the gradient descent optimization algorithm.

        INPUTS
        ======
        f: str, required
            Input function on which to perform gradient descent.
        x0: int or float, required
            Starting x-value at which to initialize the algorithm.
        epsilon: int or float, optional
            Solution accuracy threshold.
        max_iters: int, optional
            The maximum number of times to run the algorithm.
        eta: int or float, optional
            The learning rate, which controls the algorithm step size.

        RETURNS
        =======
        If a minimum is found, the algorithm returns its x-value, a list
        containing the previous points at which the derivative was 
        evaluated, a list of the function's values at those points, and
        a list of the function's derivatives at those points.
        If no minimum is found, None is returned.
        '''
        xn = x0
        x = ad(1.0)
        fn_parsed = x._parse_input(fn)
        xn_history = []
        
        for i in range(max_iters):
            xn_history.append(xn)
            new_fxn = x._evaluate_function(fn_parsed, xn)
            Dfxn = new_fxn.der[0] 
            new_xn = xn - eta*Dfxn
            if abs(new_xn - xn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(new_xn))
                return new_xn
            xn = new_xn
        print('No solution found after max iterations.')


    def nesterov_grad_descent(fn, x0, epsilon, max_iters, eta):
        '''
        WORKING
        '''
        xn = x0
        x = ad(1.0)
        t = 1.0
        yn = x0
        fn_parsed = x._parse_input(fn)
        xn_history = []
        for i in range(max_iters):
            xn_history.append(xn)
            new_fxn = x._evaluate_function(fn_parsed, yn)
            Dfxn = new_fxn.der[0]
            # Calculate the new gradient
            new_t = 0.5*(1 + np.sqrt(1 + 4*t**2))
            new_xn = yn - eta*Dfxn
            new_yn = new_xn + (t - 1.0)/(new_t)*(new_xn - xn)
            if abs(new_xn - xn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(new_xn))
                return new_xn
            if np.dot(yn - new_xn, new_xn - xn) > 0:
                new_yn = new_xn
                new_t = 1
            xn = new_xn
            yn = new_yn
            t = new_t
        print('No solution found after max iterations.')
        

    def stochastic_grad_descent(fn, x0, epsilon, max_iters, eta, train_size=1, batch_size=1):
        '''
        CURRENTLY NOT WORKING
        '''
        xn = x0
        x = ad(1.0)
        fn_parsed = x._parse_input(fn)
        xn_history = []
        for i in range(max_iters):
            xn_history.append(xn)
            # Calculate a gradient estimate
            random_points = np.random.choice(train_size, batch_size)
            grad_list = []
            random_points = [0.1, 0.2, 0.3, 0.4, 0.5]
            for point in random_points:
                new_fxn = x._evaluate_function(fn_parsed, point)
                grad_list.append(new_fxn.der[0])
            batch_grad = np.mean(grad_list)
            # Update the gradient
            new_xn = xn - (eta / (i+1))*batch_grad
            if abs(new_xn - xn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(new_xn))
                return new_xn
            xn = new_xn
            print(xn)
        print('No solution found after max iterations.')




f1 = 'x^5 - 5*x^2 + 2*x - 1'
f2 = 'x^2'
f3 = 'x^3 - 3*x^2 + 4'
#root, history = Optimize.newtons_method(f1, 2, 0.00001, max_iters=500)
Optimize.animate_newtons(f3, 0.3, 0.000001, max_iters=500, runtime=20)
#print(root, history)

#Optimize.grad_descent(f1, 0, 0.00001, max_iters=500, eta=0.5)
#Optimize.nesterov_grad_descent(f1, 0, 0.00001, max_iters=500, eta=0.5)
