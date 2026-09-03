import numpy as np
import matplotlib.pyplot as plt

class RBF:
    def __init__(self,start,end,**kwargs):
        """
        Initialize the RBF class with optional parameters.

        Parameters:
        **kwargs: Arbitrary keyword arguments for future extensions.
        """
        self.start = start
        self.end = end
        self.func_name = kwargs.get('function', 'runge_function')
        self.func = getattr(self, self.func_name)
        if kwargs.get('x_interpolate') == None:
            num_points = kwargs.get('num_points', 10)
            self.x_interpolate = np.linspace(start, end, num_points)

        self.epsilon = kwargs.get('epsilon', 1.0)
    def phi(self, r):
        """
        Evaluate the radial basis function (RBF) at a given distance r.

        Parameters:
        r (ndarray[float]): The distances at which to evaluate the RBF.

        Returns:
        ndarray[float]: The values of the RBF at r.
        """
        return np.exp(-(self.epsilon*r)**2)


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


    def find_weights(self, x):
        """
        Find the weights for the RBF interpolation given input data points.

        Parameters:
        x (ndarray[float]): The input data points.

        Returns:
        ndarray[float]: The weights for the RBF interpolation.
        """
        y = self.func(x)  

        # Compute the distance matrix
        distance_matrix = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        M = self.phi(distance_matrix)

        weights = np.linalg.solve(M, y)
        return weights

    def RBF_interpolation(self, x):
        """
        Perform RBF interpolation at given points using the computed weights.

        Parameters:
        x (ndarray[float]): The points at which to evaluate the RBF interpolation.
        weights (ndarray[float]): The weights for the RBF interpolation.

        Returns:
        ndarray[float]: The interpolated values at x.
        """
        weights = self.find_weights(self.x_interpolate)
        phis = self.phi(x[:, np.newaxis] - self.x_interpolate[np.newaxis, :])
        return phis @ weights

    def plot(self,n = 1000):
        """
        Plot the original function and its RBF interpolation.

        Parameters:
        n (int): The number of points to use for plotting.
        """

        x_plot = np.linspace(self.start, self.end, n)
        y_plot = self.func(x_plot)
        y_interp = self.RBF_interpolation(x_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label='Original Function', color='blue')
        plt.plot(x_plot, y_interp, label='RBF Interpolation', color='red', linestyle='--')
        plt.scatter(self.x_interpolate, self.func(self.x_interpolate), color='green', label='Interpolation Points')
        plt.title('RBF Interpolation')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()