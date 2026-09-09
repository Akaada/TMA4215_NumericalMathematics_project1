import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")


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
        self.piecewise = kwargs.get("piecewise", False)

        # if not piecewise, generate points and evaluate function at those points
        if not self.piecewise:
            self.x_values = self.point_function()
            self.y_values = self.func(np.array(self.x_values))
        else:
            self._initialize_piecewise(kwargs.get("K", 1))


    # a helper function to initialize piecewise interpolation so to not clutter __init__
    def _initialize_piecewise(self, K):
        """
        Initialize the piecewise interpolation by dividing the interval into K subintervals and generating points for each subinterval.

        Parameters:
        K (int): The number of subintervals to divide the interval into.
        
        Returns:
        None
        """
        self.subintervals = np.linspace(self.start,self.end,K + 1)

        # create lists to hold the x and y values for each subinterval
        self.x_values_piecewise = []
        self.y_values_piecewise = []

        # loop through each subinterval and generate points and evaluate function at those points
        for i in range(K):
            sub_start = self.subintervals[i]
            sub_end = self.subintervals[i + 1]

            # get the x and y values for the current subinterval
            x_values = self.point_function(
                n=self.num_points,
                start=sub_start,
                end=sub_end
            )
            # evaluate the function at those x values
            y_values = self.func(np.array(x_values))

            # append the x and y values to the lists for piecewise interpolation
            self.x_values_piecewise.append(np.array(x_values))
            self.y_values_piecewise.append(np.array(y_values))



    # functions to generate the different types of points for interpolation
    def equidistant_points(self, n=None, start=None, end=None):
        """
        Generate equidistant points between start and end.

        Parameters:
        n (int, optional): The number of points to generate. Defaults to self.num_points
        start (float, optional): The starting point of the interval. Defaults to self.start.
        end (float, optional): The ending point of the interval. Defaults to self.end.
        
        Returns:
        ndarray[float]: The equidistant points between start and end.
        """

        # added flexibility to specify n, start, and end for piecewise interpolation
        if n is None:
            n = self.num_points
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        # Generate n equidistant points between start and end
        return np.linspace(start, end, n)


    def chebyshev_points(self, n=None, start=None, end=None):
        """
        Generate Chebyshev points between start and end.

        Parameters:
        n (int, optional): The number of points to generate. Defaults to self.num_points        
        start (float, optional): The starting point of the interval. Defaults to self.start.
        end (float, optional): The ending point of the interval. Defaults to self.end.

        Returns:
        ndarray[float]: The Chebyshev points between start and end.
        """

        # added flexibility to specify n, start, and end for piecewise interpolation
        if n is None:
            n = self.num_points
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        # Generate n Chebyshev points between start and end
        integers = np.arange(n)
        #cheb_untransformed = np.cos((integers + 0.5) * np.pi / n)
        # since in the piecewise interpolation we want the endpoints to be included, we use the following formula for Chebyshev points
        cheb_untransformed = np.cos(integers * np.pi / (n - 1))  # Chebyshev points in [-1, 1]

        # transform the Chebyshev points from [-1, 1] to [start, end]
        return (start + end) / 2 + cheb_untransformed * (end - start) / 2






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




    # interpolation methods
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



    def piecewise_interpolation(self, x):
        """
        Evaluate the piecewise Lagrange interpolant at x.

        Parameters:
        x (ndarray[float]): The x-values at which to evaluate the piecewise interpolant.

        Returns:
        ndarray[float]: The interpolated y-values at the given x.
        """
        x_eval = np.asarray(x, dtype=float) # to ensure that x is a numpy array for consistent behavior
        result = np.zeros_like(x_eval) # Initialize result as an array of zeros with the same shape as x_eval

        # Loop through each subinterval
        for i in range(len(self.subintervals) - 1):

            # get the start and end of the current subinterval
            sub_start = self.subintervals[i]
            sub_end = self.subintervals[i + 1]


            # start by finding the indices of x_eval that are within the current subinterval
            # use a Boolean mask to identify the points in the current subinterval
            if i == len(self.subintervals) - 2:
                mask = (x_eval >= sub_start) & (x_eval <= sub_end)
            else:
                mask = (x_eval >= sub_start) & (x_eval < sub_end)

            # if no points are in the current subinterval, skip to the next one
            if not np.any(mask):
                continue

            # get the local x and y values for the current subinterval, necessary for the Lagrange basis polynomial calculation
            local_x = self.x_values_piecewise[i]
            local_y = self.y_values_piecewise[i]

            # loop over interpolation points in the current subinterval
            # note that this is similar to the lagrange_interpolation method
            # it could be reused, but due to how we have implemented the function to figure out which subinterval the x_eval points are in, it is more efficient to do it this way
            for j in range(self.num_points):

                # creates a basis polynomial for the j-th point in the current subinterval
                # this is done to avoid having to initialize the first term of the product as 1 and then multiply it by each factor in the Lagrange basis polynomial
                basis = np.ones(np.sum(mask))

                # loop over all points in the current subinterval to construct the Lagrange basis polynomial
                for k in range(self.num_points):
                    if j != k:
                        basis *= (
                            (x_eval[mask] - local_x[k])
                            / (local_x[j] - local_x[k])
                        )

                result[mask] += local_y[j] * basis


        # if the input x was a scalar, return a scalar instead of an array
        if x_eval.ndim == 0:
            return result[0]

        return result



    
    def plot_interpolation(self,points=1000):
        """
        Plot the interpolation of the stored function using Lagrange interpolation with specific point types.

        Parameters:
        points (int): The number of points to use for plotting the interpolation. Defaults to 1000.

        Returns:
        None
        """
        x_plot = np.linspace(self.start, self.end, points)
        if self.piecewise:
            y_plot = self.piecewise_interpolation(x_plot)
        else:
            y_plot = self.lagrange_interpolation(x_plot)


        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label='Lagrange Interpolation', color='blue')
        plt.plot(x_plot, self.func(x_plot), label='Original Function', color='green', linestyle='dashed')
        if not self.piecewise:
            plt.scatter(self.x_values, self.y_values, color='red', label='Data Points')
            plt.title(f'Lagrange Interpolation using {self.num_points} {self.point_function.__name__.replace("_points", "").capitalize()} Points')
        else:
            for i in range(len(self.subintervals) - 1):
                plt.scatter(
                    self.x_values_piecewise[i],
                    self.y_values_piecewise[i],
                    color=f'C{i % 10}',
                    label=f'Data Points Subinterval {i+1}',
                    zorder=3
                )       

            plt.title(f'Piecewise Lagrange Interpolation using {self.num_points} points and {len(self.subintervals)-1} Subintervals on {self.func_name} using {self.point_function.__name__.replace("_points", "").capitalize()} Points')

        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()




    def error_aprox(self,norm = 'max'):
        """
        Calculate the approximation error of the interpolation.

        Parameters:
        norm (str): The norm to use for the error calculation. Defaults to 'max'. Other options include 'l2' for L2 norm.

        Returns:
        float: The approximation error.
        """
        num_points = self.num_points *100
        x_error_eval = np.linspace(self.x_values[0], self.x_values[-1], num_points)
        y_error_eval = self.func(x_error_eval)
        if not self.piecewise:
            interpolated_values = self.lagrange_interpolation(x_error_eval)
        else:
            interpolated_values = self.piecewise_interpolation(x_error_eval)
        error = np.abs(y_error_eval - interpolated_values)
        if norm == 'max':
            return np.max(error)
        elif norm == 'l2':
            return np.sqrt(np.sum(error**2)) * np.sqrt((self.end - self.start) / num_points)
