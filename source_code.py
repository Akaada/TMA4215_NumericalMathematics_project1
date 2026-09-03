import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_values, y_values, x):
    """
    Perform Lagrange interpolation for a given set of points.

    Parameters:
    x_values (list/ndarray): A list of x-coordinates of the data points.
    y_values (list/ndarray): A list of y-coordinates of the data points.
    x (ndarray[float]): The x-values at which to evaluate the interpolating polynomial.

    Returns:
    float: The interpolated y-value at the given x.
    """

    n = len(x_values) # get the number of data points to make code more readable
    result = 0.0 # initialize the result as a float 

    for i in range(n): # loop over data points
        term = y_values[i] # get the weight (y-value) for the current data point
        for j in range(n): # loop over the data points
            if j != i: # avoid the index you are interpolating at
                term *= (x - x_values[j]) / (x_values[i] - x_values[j]) # add the term for current data point
        result += term # add Li to the result

    return result

def equidistant_points(start, end, num_points):
    """
    Generate equidistant points between a start and end value.

    Parameters:
    start (float): The starting value of the range.
    end (float): The ending value of the range.
    num_points (int): The number of equidistant points to generate.

    Returns:
    list: A list of equidistant points.
    """

    step = (end - start) / (num_points - 1) # calculate the step size
    return [start + i * step for i in range(num_points)] # generate the points

def chebishev_points(start, end, num_points):
    """
    Generate Chebyshev points between a start and end value.

    Parameters:
    start (float): The starting value of the range.
    end (float): The ending value of the range.
    num_points (int): The number of Chebyshev points to generate.

    Returns:
    list: A list of Chebyshev points.
    """

    integers = np.arange(num_points) # create an array of integers from 0 to num_points-1
    cheb_untransformed = np.cos((integers + 0.5)*np.pi/num_points) # calculate the Chebyshev points in the range [-1, 1]
    return (start + end)/2 +  cheb_untransformed*(end - start)/2 # transform the points to the range [start, end]


def runge_function(x):
    """
    Evaluate the Runge function at a given x. Used for testing interpolation methods.

    Parameters:
    x (ndarray[float]): The x-values at which to evaluate the Runge function.

    Returns:
    ndarray[float]: The values of the Runge function at x.
    """
    return 1 / (1 + x**2) # return the value of the Runge function

def func1(x):
    """
    Evaluate the function f(x) = cos(2πx) at a given x. Used for testing interpolation methods.

    Parameters:
    x (ndarray[float]): The x-values at which to evaluate the function.

    Returns:
    ndarray[float]: The values of the function at x.
    """
    return np.cos(2*np.pi*x) # return the value of the function

def func2(x):
    """
    Evaluate the function f(x) = e^(3x) * sin(2x) at a given x. Used for testing interpolation methods.

    Parameters:
    x (ndarray[float]): The x-values at which to evaluate the function.

    Returns:
    ndarray[float]: The values of the function at x.
    """
    return np.exp(3*x)*np.sin(2*x) # return the value of the function

def plot_interpolation(start,end,function, num_points=10, point_type= 'equidistant'):
    """
    Plot the interpolation of a given function using Lagrange interpolation with specific point types.

    Parameters:
    start (float): The starting value of the range.
    end (float): The ending value of the range.
    function (callable): The function to interpolate.
    num_points (int): The number of points to use for interpolation.
    point_type (str): The type of points to use ('equidistant' or 'chebyshev').

    Returns:
    None
    """

    if point_type == 'equidistant':
        x_values = equidistant_points(start, end, num_points)
    elif point_type == 'chebyshev':
        x_values = chebishev_points(start, end, num_points)
    else:
        raise ValueError("point_type must be 'equidistant' or 'chebyshev'")

    y_values = function(np.array(x_values)) # evaluate the function at the chosen points

    x_plot = np.linspace(start, end, 1000) # create a dense set of points for plotting
    y_plot = lagrange_interpolation(x_values, y_values, x_plot) # perform interpolation

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label='Lagrange Interpolation', color='blue')
    plt.plot(x_plot, function(x_plot), label='Original Function', color='green', linestyle='dashed')
    plt.scatter(x_values, y_values, color='red', label='Data Points')
    plt.title(f'Lagrange Interpolation using {point_type.capitalize()} Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()



class LagrangeInterpolator:
    """
    A class to perform Lagrange interpolation for a given set of points.
    """

    def __init__(self,start,end, **kwargs):
        """
        Initialize the LagrangeInterpolator with x and y values.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
        """
        self.start = start
        self.end = end
        self.num_points = kwargs.get('num_points', 10)
        self.func_name = kwargs.get('function', 'runge_function')
        self.func = getattr(self, self.func_name)
        self.point_function = getattr(self,f"{kwargs.get('point_type', 'equidistant')}_points")
        self.x_values = self.point_function()
        self.y_values = self.func(np.array(self.x_values))




    # functions to generate the different types of points for interpolation
    def equidistant_points(self):
        """
        Generate equidistant points between the start and end values. Stores the points in self.x_values.
        """
        step = (self.end - self.start) / (self.num_points - 1)
        return [self.start + i * step for i in range(self.num_points)]

    def chebyshev_points(self):
        """
        Generate Chebyshev points between the start and end values. Stores the points in self.x_values.
        """
        integers = np.arange(self.num_points)
        cheb_untransformed = np.cos((integers + 0.5)*np.pi/self.num_points)
        return (self.start + self.end)/2 +  cheb_untransformed*(self.end - self.start)/2






    # functions to evaluate the different functions for interpolation
    def runge_function(self, x):
        """
        Evaluate the Runge function at a given x.

        Parameters:
        x (ndarray[float]): The x-values at which to evaluate the Runge function.

        Returns:
        ndarray[float]: The values of the Runge function at x.
        """
        return 1 / (1 + x**2)

    def func1(self, x):
        """
        Evaluate the function f(x) = cos(2πx) at a given x.

        Parameters:
        x (ndarray[float]): The x-values at which to evaluate the function.

        Returns:
        ndarray[float]: The values of the function at x.
        """
        return np.cos(2*np.pi*x)

    def func2(self, x):
        """
        Evaluate the function f(x) = e^(3x) * sin(2x) at a given x.

        Parameters:
        x (ndarray[float]): The x-values at which to evaluate the function.

        Returns:
        ndarray[float]: The values of the function at x.
        """
        return np.exp(3*x)*np.sin(2*x)

    def lagrange_interpolation(self, x):
        """
        Perform Lagrange interpolation for the stored x and y values.

        parameters:
        x (ndarray[float]): The x-values at which to evaluate the interpolating polynomial.

        Returns:
        ndarray[float]: The interpolated y-value at the given x.

        """
        n = len(self.x_values)
        result = np.zeros_like(x, dtype=float)  # Initialize result as an array of zeros with the same shape as x

        for i in range(n):
            term = self.y_values[i]
            for j in range(n):
                if j != i:
                    term *= (x - self.x_values[j]) / (self.x_values[i] - self.x_values[j])
            result += term

        return result


    def plot_interpolation(self,points=1000):
        """
        Plot the interpolation of the stored function using Lagrange interpolation with specific point types.

        Returns:
        None
        """
        x_plot = np.linspace(self.start, self.end, points)
        y_plot = self.lagrange_interpolation(x_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label='Lagrange Interpolation', color='blue')
        plt.plot(x_plot, self.func(x_plot), label='Original Function', color='green', linestyle='dashed')
        plt.scatter(self.x_values, self.y_values, color='red', label='Data Points')
        plt.title(f'Lagrange Interpolation using {self.point_function.__name__.replace("_points", "").capitalize()} Points')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()