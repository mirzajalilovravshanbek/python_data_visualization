# regression_handler.py
from scipy.optimize import curve_fit

class RegressionHandler:
    def least_squares_fit(self, x, y):
        def func(x, *params):
            return sum(p * x ** i for i, p in enumerate(params))

        # Adjust the initial guess based on the number of parameters needed
        num_params = 4  # Number of parameters for each function
        initial_guess = [1.0] * num_params
        
        try:
            params, _ = curve_fit(func, x, y, p0=initial_guess)
        except TypeError as e:
            print(f"Error: {e}")
            params = None

        return params
