from AutoDiff import AutoDiff as ad
from AutoDiff import Function as fun
import numpy as np


class Optimize:

    def newtons_method(fn, x0, epsilon, max_iters):
        '''
        WORKING
        '''
        xn = x0
        x = ad(1.0) # dummy ad object
        fn_parsed = x._parse_input(fn)
        for i in range(max_iters):
            new_fxn = x._evaluate_function(fn_parsed, xn)
            fxn = new_fxn.val[0]
            Dfxn = new_fxn.der[0]
            if abs(fxn) < epsilon:
                print('Found solution after {} iterations.'.format(i))
                print('Solution is: {}'.format(xn))
                return xn
            if Dfxn == 0:
                print('Reached zero derivative. No solution found.')
                return None
            xn = xn - fxn/Dfxn
        print('No solution found after max iterations.')


    def grad_descent(fn, x0, epsilon, max_iters, eta):
        '''
        WORKING
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



f1 = 'x^2 - 5*x + 1'
Optimize.newtons_method(f1, 0, 0.00001, max_iters=500)
Optimize.grad_descent(f1, 0, 0.00001, max_iters=500, eta=0.5)
Optimize.nesterov_grad_descent(f1, 0, 0.00001, max_iters=500, eta=0.5)
