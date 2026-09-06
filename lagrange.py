
from email import errors

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")


# functions to interpolate, define and name them for plotting and ease of use
def runge(x):
    """f(x) = 1 / (1 + x^2)."""
    return 1.0 / (1.0 + x**2)
runge.__name__ = "Runge function"

def f_cos(x):
    """f(x) = cos(2*pi*x), used on [0, 1]."""
    return np.cos(2 * np.pi * x)
f_cos.__name__ = "cos(2*pi*x)"

def f_expsin(x):
    """f(x) = exp(3x) * sin(2x), used on [0, pi/4]."""
    return np.exp(3 * x) * np.sin(2 * x)
f_expsin.__name__ = "exp(3x) * sin(2x)"



# nodes generation functions
def equidistant_nodes(a, b, n):
    """
    Generate n+1 equidistant nodes on [a, b].

    parameters:
    a (float): left endpoint of the interval
    b (float): right endpoint of the interval
    n (int): number of intervals

    returns:
    np.ndarray: array of n+1 equidistant nodes
    """
    return np.linspace(a, b, n + 1)


def chebyshev_nodes(a, b, n):
    """
    Generate n+1 Chebyshev nodes on [a, b].

    parameters:
    a (float): left endpoint of the interval
    b (float): right endpoint of the interval
    n (int): number of intervals

    returns:
    np.ndarray: array of n+1 Chebyshev nodes
    """
    if n == 0:
        return np.array([0.5 * (a + b)])
    t = np.cos(np.arange(n + 1) * np.pi / n)[::-1]   # Chebyshev nodes on [-1, 1]
    return 0.5 * (b - a) * t + 0.5 * (b + a)         # Chebyshev nodes on [a, b]



def get_nodes(nodes, a, b, n):
    """
    Get the nodes for interpolation.

    parameters:
    nodes (str): type of nodes to generate ("equidistant" or "chebyshev")
    a (float): left endpoint of the interval
    b (float): right endpoint of the interval
    n (int): number of intervals

    returns:
    np.ndarray: array of n+1 nodes
    """
    if nodes == "equidistant":
        return equidistant_nodes(a, b, n)
    elif nodes == "chebyshev":
        return chebyshev_nodes(a, b, n)
    raise ValueError("nodes must be 'equidistant' or 'chebyshev'")


def lagrange_eval(x_nodes, y_nodes, x_eval):
    """
    Evaluate the Lagrange interpolating polynomial at given points.

    Parameters
    x_nodes (ndarray): The x-coordinates of the interpolation nodes.
    y_nodes (ndarray): The y-coordinates of the interpolation nodes.
    x_eval (ndarray): The x-coordinates at which to evaluate the interpolating polynomial.

    Returns
    ndarray: The y-coordinates of the interpolating polynomial at the evaluation points.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x = np.asarray(x_eval, dtype=float)

    result = np.zeros(x.shape, dtype=float)
    for i in range(x_nodes.size):
        basis = np.ones(x.shape, dtype=float)      # l_i evaluated at x
        for j in range(x_nodes.size):
            if j != i:
                basis *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * basis

    return float(result) if result.ndim == 0 else result


def make_lagrange(f, a, b, n, nodes="equidistant"):
    """
    Create a Lagrange interpolating function for a given function f on the interval [a, b] using n intervals and specified node type.
    
    
    parameters:
    f (callable): The function to interpolate.
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    n (int): The number of intervals.
    nodes (str): The type of nodes to use ("equidistant" or "chebyshev").

    returns:
    callable: The Lagrange interpolating function.
    """
    x_nodes = get_nodes(nodes, a, b, n)
    y_nodes = np.asarray(f(x_nodes), dtype=float)

    def g(x):
        return lagrange_eval(x_nodes, y_nodes, x)

    g.nodes = x_nodes
    g.label = f"n = {n} ({nodes})"
    g.funcion = f
    return g

def make_piecewise(f, a, b, K, n, nodes="equidistant"):
    """
    Create a piecewise Lagrange interpolating function for a given function f on the interval [a, b] using K subintervals and n intervals per subinterval with specified node type.

    Parameters
    f (callable): The function to interpolate.
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    K (int): The number of subintervals.
    n (int): The number of intervals per subinterval.
    nodes (str): The type of nodes to use ("equidistant" or "chebyshev").

    Returns
    callable: The piecewise Lagrange interpolating function.
    """
    v = np.linspace(a, b, K + 1)
    xs = [get_nodes(nodes, v[k], v[k + 1], n) for k in range(K)]
    ys = [np.asarray(f(xk), dtype=float) for xk in xs]

    def g(x):
        out = np.zeros_like(x, dtype=float)
        for k in range(K):
            upper = (x <= v[k + 1]) if k == K - 1 else (x < v[k + 1])
            mask = (x >= v[k]) & upper
            if np.any(mask):
                out[mask] = lagrange_eval(xs[k], ys[k], x[mask])
        return out 

    g.nodes = np.unique(np.concatenate(xs))
    g.label = f"K = {K}, n = {n} ({nodes})"
    g.funcion = f
    return g

def plot_interpolant(a, b, *interps, n_plot=1000):
    """
    Plot the original function and its interpolants.
    
    Parameters:
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    *interps: The interpolant functions to plot.
    n_plot (int): The number of points to use for plotting.

    returns:
    None
    """
    f = interps[0].funcion
    x = np.linspace(a, b, n_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x, f(x), color="black", linestyle="dashed", label="f(x)")
    for i, g in enumerate(interps):
        plt.plot(x, g(x), color=f"C{i}", label=g.label)
        plt.scatter(g.nodes, f(g.nodes), color=f"C{i}", s=25, zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Interpolation of {f.__name__} on [{a}, {b}]")
    plt.legend()
    plt.grid(True)
    plt.show()


def error(g):
    """
    Compute the maximum and L2 error of the interpolant g compared to the original function f.
    
    Parameters
    g (callable): The interpolant function.

    Returns:
    tuple: A tuple containing the maximum error and the L2 error.
    """

    f = g.funcion
    N = len(g.nodes) * 100
    x = np.linspace(g.nodes[0], g.nodes[-1], N)
    a = g.nodes[0]
    b = g.nodes[-1]

    error = np.abs(f(x) - g(x))
    return (np.max(error), np.sqrt(np.sum(error**2) *(b - a)/N))



def convergence(f,a,b,n_vals,nodes="equidistant",N=1000):
    """
    Compute the convergence errors for a given function f on the interval [a, b] using Lagrange interpolation with specified node type.
    
    Parameters:
    f (callable): The function to interpolate.
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    n_vals (array-like): The values of n to use for interpolation.
    nodes (str): The type of nodes to use for interpolation.
    N (int): The number of points to use for error computation.

    Returns:
    tuple: A tuple containing the maximum errors and the L2 errors.
    """
    max_error_list = []
    l2_error_list = []
    for n in n_vals:
        g = make_lagrange(f, a, b, n, nodes=nodes)
        max_error, l2_error = error(g)
        max_error_list.append(max_error)
        l2_error_list.append(l2_error)
    return (np.array(max_error_list), np.array(l2_error_list))

def convergence_piecewise(f, a, b, K_vals, n, nodes="equidistant", N=1000):
    """ 
    Compute the convergence errors for a given function f on the interval [a, b] using piecewise Lagrange interpolation with specified node type.

    Parameters:
    f (callable): The function to interpolate.
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    K_vals (array-like): The values of K to use for piecewise interpolation.
    n (int): The number of intervals per subinterval.
    nodes (str): The type of nodes to use for interpolation.
    N (int): The number of points to use for error computation.

    Returns:
    tuple: A tuple containing the maximum errors and the L2 errors.
    """

    max_error_list = []
    l2_error_list = []
    for K in K_vals:
        g = make_piecewise(f, a, b, K, n, nodes=nodes)
        max_error, l2_error = error(g)
        max_error_list.append(max_error)
        l2_error_list.append(l2_error)
    return (np.array(max_error_list), np.array(l2_error_list))

# original plot_convergence function, kept for reference
"""
def plot_convergence(f,a,b,n0,n_end,n_step=1, nodes="equidistant",N=1000):
    n_vals = np.arange(n0, n_end + 1, n_step)
    max_error, l2_error = convergence(f,a,b,n_vals,nodes=nodes,N=N)

    plt.figure(figsize=(10, 6))
    plt.semilogy(n_vals, max_error, label="Max error", marker='o')
    plt.semilogy(n_vals, l2_error, label="L2 error", marker='o')
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.title(f"Convergence of Lagrange interpolation for {f.__name__} on [{a}, {b}] using {nodes} nodes")
    plt.legend()
    plt.grid(True)
    plt.show()
"""

def plot_convergence(functions, n0, n_end, n_step=1,nodes = ["equidistant", "chebyshev"], N=1000):
    """
    Plot convergence errors for multiple functions and node types.

    Parameters:
    functions (list): A list of tuples containing the function, left endpoint, and right endpoint. (f, a, b)
    n0 (int): The starting value of n for interpolation.
    n_end (int): The ending value of n for interpolation.
    n_step (int): The step size for the range of n values.
    nodes (list): A list of strings representing the types of nodes to use for interpolation.
    N (int): The number of points to use for error computation.
    """
    n_vals = np.arange(n0, n_end + 1, n_step)

    fig, axes = plt.subplots(
        len(functions),
        len(nodes),
        figsize=(6 * len(nodes), 4 * len(functions)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row, (f, a, b) in enumerate(functions):
        for col, node_type in enumerate(nodes):
            max_error, l2_error = convergence(
                f, a, b, n_vals, nodes=node_type, N=N
            )

            ax = axes[row, col]
            ax.semilogy(n_vals, max_error, label="Max error", marker="o")
            ax.semilogy(n_vals, l2_error, label="L2 error", marker="o")

            ax.set_title(
                f"{f.__name__} on [{a}, {b}]\n{node_type} nodes"
            )
            ax.set_xlabel("n")
            ax.set_ylabel("Error")
            ax.grid(True)
            ax.legend()

    fig.tight_layout()
    plt.show()





def plot_piecewise_convergence(functions, K0, K_end,n = 3, K_step=1,nodes = ["equidistant", "chebyshev"], N=1000):
    """
    Plot convergence errors for piecewise Lagrange interpolation for multiple functions and node types.

    Parameters:
    functions (list): A list of tuples containing the function, left endpoint, and right endpoint. (f, a, b)
    K0 (int): The starting value of K for interpolation.
    K_end (int): The ending value of K for interpolation.
    n (int): The number of intervals per subinterval.
    K_step (int): The step size for the range of K values.
    nodes (list): A list of strings representing the types of nodes to use for interpolation.
    N (int): The number of points to use for error computation.

    Returns:
    None
    """
    K_vals = np.arange(K0, K_end + 1, K_step)

    fig, axes = plt.subplots(
        len(functions),
        len(nodes),
        figsize=(6 * len(nodes), 4 * len(functions)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row, (f, a, b) in enumerate(functions):
        for col, node_type in enumerate(nodes):
            max_error, l2_error = convergence_piecewise(
                f, a, b, K_vals, n, nodes=node_type, N=N
            )

            ax = axes[row, col]
            ax.semilogy(K_vals, max_error, label="Max error", marker="o")
            ax.semilogy(K_vals, l2_error, label="L2 error", marker="o")

            ax.set_title(
                f"{f.__name__} on [{a}, {b}]\n"
                f"piecewise n = {n}, {node_type} nodes"
            )
            ax.set_xlabel("K")
            ax.set_ylabel("Error")
            ax.grid(True)
            ax.legend()

    fig.tight_layout()
    plt.show()