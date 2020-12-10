#Gradient Descent Method
#GRADIENT DESCENT 
#FOR ONE-VARIABLE

from AutoDiff_Carlos import AutoDiff as ad
import numpy as np

class optimize():

    def graddescent(starting_point, funct):
    
        x_start= starting_point
        x_final = -99999

        deriv_val_start = 3*x_start**2
        returned_val = ad(x_start, input_function=funct)
        deriv_val_start = returned_val.der
        #deriv_val_start = 
        deriv_val_final = 10000
        epsilon = 1e-6
        step = 0.01
        deriv_difference = abs(deriv_val_final - deriv_val_start)
        num_iterations = 0
    
        while (deriv_difference >= epsilon and num_iterations <= 10000):
        
            print(x_start,x_final,deriv_val_start,deriv_val_final)
        
            x_final = x_start - step*deriv_val_start

            returned_val = ad(x_final, input_function=funct)
        
            deriv_val_final = returned_val.der 
        
            print(x_start,x_final,deriv_val_start,deriv_val_final)
        
            deriv_difference = abs(deriv_val_final - deriv_val_start)
            deriv_val_start = deriv_val_final
            x_start = x_final
            num_iterations = num_iterations + 1

        if (deriv_difference <= epsilon):
            print("optimum solution")
            print(round(x_start, 3))
            return round(x_start, 3)
        elif (deriv_difference > epsilon and num_iterations > 10000):
            print("No solution found")
            return None


returned_val = optimize.graddescent(1.5, 'x^2')

print(returned_val)
