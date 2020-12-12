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
        a list of the function's derivatives at those points in a 4-tuple.
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
                return (xn, xn_history, yn_history, der_history)
            
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
        A matplotlib animataion showing how Newton's method located
        a root of the input function if a root was found, otherwise
        None.
        '''
        # Get the root and history of x-values, y-values, and derivatives 
        # for the input function using newtons_method
        results = Optimize.newtons_method(f, x0, epsilon, max_iters)
        
        # Check if newtons_method returned None
        if results == None:
            print("Newton's method did not find a root.")
            return None
        else:
            root, x_vals, y_vals, grad_vals = results[0], results[1], results[2], results[3]

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
        a list of the function's derivatives at those points in a 4-tuple.
        If no minimum is found, None is returned.
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
            
            # Apply the gradient descent update (scaled by learning rate)
            new_xn = xn - eta*der

            # If threshold is met, terminate the algorithm and return results
            if abs(new_xn - xn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(new_xn))
                return (new_xn, xn_history, yn_history, der_history)

            # Update xn
            xn = new_xn

        # If no solution is found within 'max_iters', tell the user and return None
        print('No solution found after max iterations.')
        return None


    def nesterov_grad_descent(f, x0, epsilon, max_iters, eta):
        '''
        Implements Nesterov's accelerated gradient descent 
        optimization algorithm, which uses a momentum parameter 't'.

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
        a list of the function's derivatives at those points in a 4-tuple.
        If no minimum is found, None is returned.
        '''
        # Initialize xn at the starting point
        xn = x0

        # Create a DreamDiff object to access private methods
        x = ad(1.0)

        # Initialize the momentum coefficient
        t = 1.0

        # Initialize the y-value
        y = x0

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
            new_f = x._evaluate_function(f_parsed, y)
            yn = new_f.val[0]
            der = new_f.der[0]
            yn_history.append(yn)
            der_history.append(der)

            # Apply Nesterov's momentum update
            new_t = 0.5*(1 + np.sqrt(1 + 4*t**2))

            # Calculate the new gradient (scaled by learning rate)
            new_xn = y - eta*der
            new_y = new_xn + (t - 1.0)/(new_t)*(new_xn - xn)

            # If threshold is met, terminate the algorithm and return results
            if abs(new_xn - xn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(new_xn))
                return (new_xn, xn_history, yn_history, der_history)

            # Reset conditions
            if np.dot(y - new_xn, new_xn - xn) > 0:
                new_y = new_xn
                new_t = 1

            # Update xn, y, t
            xn = new_xn
            y = new_y
            t = new_t

        # If no solution is found within 'max_iters', tell the user and return None
        print('No solution found after max iterations.')
        return None
    

    def animate_grad_desc(f, x0, epsilon=0.000001, max_iters=500, eta=0.1, method='grad', runtime=20): 
        '''
        Creates an animation of the gradient descent method for scalar
        functions by plotting the function and showing the tangent lines
        at which the gradient was evaluated.

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
        method: str, optional
            The gradient descent method to use: input 'grad' to use
            standard gradient descent and 'nesterov' to use
            Nesterov's accelerated gradient descent.
        runtime: int, optional
            The length of time to run the animation.

        RETURNS
        =======
        A matplotlib animataion showing how gradient descent located
        a minimum of the input function, depending on the descent method
        used.
        '''
        # Get the minimum and history of x-values, y-values, and derivatives 
        # for the input function using grad_descent
        if method == 'grad':
            results = Optimize.grad_descent(f, x0, epsilon, max_iters, eta)
        elif method == 'nesterov':
            results = Optimize.nesterov_grad_descent(f, x0, epsilon, max_iters, eta)
        else:
            raise Exception(f'invalid gradient descent method: {method}')

        # Check if grad_descent returned None
        if results == None:
            print('Gradient descent did not find a minimum.')
            return None
        else:
            minimum, x_vals, y_vals, grad_vals = results[0], results[1], results[2], results[3]

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
            fig.suptitle("Gradient Descent for f(x)="+f)

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


    def plot_coeffs(coeffs, x_array, y_array):
        '''
        Plots coefficients along with original points

        INPUTS
        ======
        coeffs: array of length 3n, required
            The x values for user's given points, must be same length as y_array

        x_array: array of numeric values of length n+1, required
            The x values for user's given points, must be same length as y_array
        y_array: array of numeric values of length n+1, required
            The y values for user's given points, must be same length as x_array

        RETURNS
        =======
        plot: A matplotlib.pyplot object
            A plot with quadratic splines and points plotted.
        '''
        # check x and y for all points
        if not (len(x_array) == len(y_array)):
            raise Exception('Number of x values for data points do not match number of y values')

        num_eq = int(len(coeffs)/3)
        
        # plot each equation
        for i in range(num_eq):
            # define quadratic function for plotting
            def f(x, coeffs, i):
                return coeffs[i*3]*(x**2) + coeffs[(i*3)+1]*x + coeffs[(i*3)+2]
            
            temp_x = []
            # plot individual quadratic function
            temp_arange = np.arange(start = x_array[i], stop=x_array[i+1], step=0.01)
            for j in temp_arange:
                temp_x.append(f(j, coeffs, i))
            plt.plot(temp_arange, temp_x)

        # plot points
        plt.plot(x_array, y_array, 'ro')
        plt.title('Quadratic Splines')
        plt.show()
        return plt
    

    def quadratic_spline(x_array, y_array, plot = False):
        '''
        Runs Quadratic Spline interpolation on a set of points.

        INPUTS
        ======
        x_array: array of numeric values, required
            The x values for user's given points, must be same length as y_array
        y_array: array of numeric values, required
            The x values for user's given points, must be same length as x_array
        plot: bool, optional, default = False
            If plot is True, plots the piecewise quadratic splines using matplotlib.pyplot

        RETURNS
        =======
        Coeffs: list of floats
            The list of coefficients to the quadratic spline equations in list as [a_1, b_1, c_1,... a_n, b_n, c_n]

        Prints piecewise function using coefficients.
        
        If plot==True, displays quadratic splines using matplotlib.pyplot
        '''

        # ensure all points have x and y values
        if not (len(x_array) == len(y_array)):
            raise Exception('number of x values for data points do not match number of y values')

        # ensure at least 3 data points for quadratic splines
        if len(x_array)<3:
            raise Exception('Too few data points for quadratic spline interpolation')
        
        n = len(x_array) - 1
        sys_of_equations = np.zeros((3*n, 3*n))

        # define AutoDiff objects a,b,c
        def a(ad_val):
            return ad(ad_val**2)
        def b(ad_val):
            return ad(ad_val)
        def c(ad_val):
            return ad(1)

        # define 3n linear equations
        for i in range(n):
            # Equations to connect quadratic between point i, i+1
            sys_of_equations[3*i][(3*i)] = a(x_array[i]).val
            sys_of_equations[3*i][((3*i)+1)] = b(x_array[i]).val
            sys_of_equations[3*i][((3*i)+2)] = c(x_array[i]).val
            sys_of_equations[(3*i)+ 1][(3*i)] = a(x_array[i+1]).val
            sys_of_equations[(3*i)+ 1][((3*i)+1)] = b(x_array[i+1]).val
            sys_of_equations[(3*i)+ 1][((3*i)+2)] = c(x_array[i+1]).val

            if i!=n-1:
                # Constraints so derivative matches at all interior points
                sys_of_equations[(3*i)+ 2][(3*i)] = a(x_array[i+1]).der
                sys_of_equations[(3*i)+ 2][((3*i)+1)] = b(x_array[i+1]).der
                sys_of_equations[(3*i)+ 2][((3*i)+2)] = c(x_array[i+1]).der
                sys_of_equations[(3*i)+ 2][(3*(i+1))] = -(a(x_array[i+1]).der)
                sys_of_equations[(3*i)+ 2][((3*(i+1))+1)] = -(b(x_array[i+1]).der)
                sys_of_equations[(3*i)+ 2][((3*(i+1))+2)] = -(c(x_array[i+1]).der)

        # Last of 3n equations
        sys_of_equations[3*n -1][0] = 1

        # construct y array for linear equations
        y = np.zeros(3*n)
        for i in range(n):
            y[(3*i)] = y_array[i]
            y[(3*i)+1] = y_array[i+1]
            y[(3*i)+2] = 0

        # solve for coeffs
        coeffs_all = np.linalg.lstsq(sys_of_equations, y, rcond=None)
        coeffs = coeffs_all[0]

        # truncate coeffs
        for i in range(len(coeffs)):
            coeffs[i] = float(f"{coeffs[i]:.4f}")

        print("Quadratic Spline Equations: ")
        for index in range(n):
            print("S_{index}(x) = {a}*x^2 + {b}*x + {c} for x ∈ [{i_0}, {i_1}]".format(
                index=index, a=coeffs[index*3], b=coeffs[index*3+1], c=coeffs[index*3+2], i_0=x_array[index], i_1=x_array[index+1]
                ))
        # plot
        if plot == True:
            spline_plot = Optimize.plot_coeffs(coeffs, x_array, y_array)

        return coeffs


