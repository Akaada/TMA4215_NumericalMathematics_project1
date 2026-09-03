import numpy as np
import matplotlib.pyplot as plt


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
        plt.title(f'Lagrange Interpolation using {self.num_points} {self.point_function.__name__.replace("_points", "").capitalize()} Points')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()