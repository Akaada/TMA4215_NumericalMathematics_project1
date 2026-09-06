import numpy as np
import matplotlib.pyplot as plt
import torch

plt.style.use("bmh")


# functions to interpolate, define and name them for plotting and ease of use
# created new functions for torch, since the original ones were not compatible with torch tensors
# instead of having to rewrite the entire interpolation code, I created new functions that are compatible with torch tensors
def runge_torch(x):
    """f(x) = 1 / (1 + x^2)."""
    return 1.0 / (1.0 + x**2)
runge_torch.__name__ = "Runge function"

def f_cos_torch(x):
    """f(x) = cos(2*pi*x)"""
    return torch.cos(2 * torch.pi * x)
f_cos_torch.__name__ = "cos(2*pi*x)"

def f_expsin_torch(x):
    """f(x) = exp(3x) * sin(2x)"""
    return torch.exp(3 * x) * torch.sin(2 * x)
f_expsin_torch.__name__ = "exp(3x) * sin(2x)"

def phi(r,epsilon):
    return torch.exp(-(epsilon * r) ** 2)

def M_create(x, epsilon):
    distances = torch.abs(x[:, None] - x[None, :])
    return phi(distances, epsilon)

def weights_create(x,f, epsilon):
    y = f(x)
    M = M_create(x, epsilon)
    return torch.linalg.solve(M, y)


def RBF_interpolation(x, x_nodes, weights, epsilon):
    # NOTE: This function does not work for scalar x
    distances = torch.abs(x[:, None] - x_nodes[None, :])
    return torch.matmul(phi(distances, epsilon), weights)

def RBF_interpolation_task(x,x_nodes,y,epsilon):
    # NOTE: This function is only created for the task. 
    # NOTE: It is more efficient to create the weights once and then use them for multiple evaluations of the interpolation function.
    weights = weights_create(x_nodes, y, epsilon)
    return RBF_interpolation(x, x_nodes, weights, epsilon)


def create_RBF_interpolator(x_nodes, f, epsilon):
    weights = weights_create(x_nodes, f, epsilon)
    def g(x):
        return RBF_interpolation(x, x_nodes, weights, epsilon)

    g.function = f
    g.nodes = x_nodes
    g.epsilon = epsilon
    return g


def plot_RBF_interpolator(f,a,b,epsilon,n,N=1000):
    x_nodes = torch.linspace(a, b, n+1)
    g = create_RBF_interpolator(x_nodes, f, epsilon)

    x_plot = torch.linspace(a, b, N)
    y_plot = g(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.numpy(), y_plot.numpy(), label=f"RBF Interpolation (n={n}, ε={epsilon})")
    plt.plot(x_plot.numpy(), f(x_plot).numpy(), label=f"Original Function: {f.__name__}")
    plt.scatter(x_nodes.numpy(), f(x_nodes).numpy(), color='red', zorder=5, label="Interpolation Nodes")
    plt.title(f"RBF Interpolation of {f.__name__} with n={n} and ε={epsilon}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


def condition_number_RBF(x_nodes, epsilon):
    M = M_create(x_nodes, epsilon)
    return torch.linalg.cond(M).item()


def error(g):
    """
    Compute the maximum and L2 error of the interpolant g compared to the original function f.
    
    Parameters
    g (callable): The interpolant function.

    Returns:
    tuple: A tuple containing the maximum error and the L2 error.
    """
    f = g.function
    N = len(g.nodes) * 100
    x = torch.linspace(g.nodes[0], g.nodes[-1], N)
    a = g.nodes[0]
    b = g.nodes[-1]

    error = torch.abs(f(x) - g(x))
    return (torch.max(error).item(), torch.sqrt(torch.sum(error**2) *(b - a)/N).item())


def plot_RBF_convergence_condition(function,a,b,epsilon0,epsilon_end,epsilon_number,n=10,N=1000):
    """
    Plot convergence errors and condition numbers for RBF interpolation for a given function and range of parameters.
    
    Parameters:
    function (callable): The function to interpolate.
    a (float): The start of the interval.
    b (float): The end of the interval.
    epsilon0 (float): The initial value of epsilon.
    epsilon_end (float): The final value of epsilon.
    epsilon_number (int): The number of epsilon values to test.
    n (int, optional): The number of nodes to use for interpolation. Default is 10.
    N (int, optional): Number of points for plotting. Default is 1000.
    """

    epsilon_values = torch.linspace(epsilon0, epsilon_end, epsilon_number)
    max_errors = []
    l2_errors = []
    cond_numbers = []
    x_nodes = torch.linspace(a, b, n+1)
    for epsilon in epsilon_values:
        g = create_RBF_interpolator(x_nodes, function, epsilon)

        max_error, l2_error = error(g)
        cond_number = condition_number_RBF(x_nodes, epsilon)

        max_errors.append(max_error)
        l2_errors.append(l2_error)
        cond_numbers.append(cond_number)

    epsilon_values = epsilon_values.numpy()

    fig, error_axis = plt.subplots(figsize=(10, 6))
    condition_axis = error_axis.twinx()

    error_axis.semilogy(
        epsilon_values, max_errors, "o-", label="Max error"
    )
    error_axis.semilogy(
        epsilon_values, l2_errors, "s-", label="L2 error"
    )

    condition_axis.semilogy(
        epsilon_values, cond_numbers, "^-", color="red",
        label="Condition number"
    )

    error_axis.set_xlabel("epsilon")
    error_axis.set_ylabel("Interpolation error")
    condition_axis.set_ylabel("Condition number")

    error_axis.set_title(
        f"RBF errors and condition number for {function.__name__}"
    )
    error_axis.grid(True)

    lines1, labels1 = error_axis.get_legend_handles_labels()
    lines2, labels2 = condition_axis.get_legend_handles_labels()
    error_axis.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    plt.show()

