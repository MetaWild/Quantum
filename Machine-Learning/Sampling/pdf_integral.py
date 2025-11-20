import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap
from uniform import standard_uniform
import inspect


def multidimensional_monte_carlo(f, a, b, num_transistors, num_dimensions, num_samples, key):
    subkeys = random.split(key, num_dimensions)
    samples = vmap(lambda subkey, start, end: start + (end - start) * standard_uniform(num_transistors, subkey, num_samples))(subkeys, a, b).T
    function_values = vmap(lambda args: f(*args))(samples)
    return jnp.mean(function_values, axis=0) * (b[0] - a[0])**num_dimensions

def gauss_int_jax(coeffs, a, b):
    def u_function(t):
        return jnp.polyval(coeffs, ((b - a) * t + b + a) / 2)
    x_0 = -jnp.sqrt(3/5)
    x_1 = 0.0
    x_2 = jnp.sqrt(3/5)
    w_0 = 5/9
    w_1 = 8/9
    w_2 = 5/9
    integral = ((b - a) / 2) * (
        w_0 * u_function(x_0) + w_1 * u_function(x_1) + w_2 * u_function(x_2)
    )
    return integral

def gauss_int(f, a, b):
    def u_function(t):
        return f(((b - a) * t + b + a) / 2)
    x_0 = -np.sqrt(3/5)
    x_1 = 0.0
    x_2 = np.sqrt(3/5)
    w_0 = 5/9
    w_1 = 8/9
    w_2 = 5/9
    integral = ((b - a) / 2) * (
        w_0 * u_function(x_0) + w_1 * u_function(x_1) + w_2 * u_function(x_2)
    )
    return integral

def comp_gauss_int(f, a, b, n):
    h = (b - a) / n
    integral = 0.0
    for i in range(n):
        x_i = a + i * h
        x_i1 = a + (i + 1) * h
        integral += gauss_int(f, x_i, x_i1)
    return integral

def get_num_arguments(func):
    sig = inspect.signature(func)
    return len(sig.parameters)

def standard_normal_2d(x, y):
    return (1 / (2 * jnp.pi)) * jnp.exp(-0.5 * (x**2 + y**2))

# a = -1
# b = 1
# num_transistors = 10
# num_samples = 100000
# key = random.PRNGKey(0)

# result = multidimensional_monte_carlo(standard_normal_2d, a, b, num_transistors, num_samples, key)
# print(f"The integral of the 2D standard normal distribution from {a} to {b} is approximately {result}")

# def f(x):
#     return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# result = comp_gauss_int(f, -1, 1, 10)
# print(f"The integral of f from -1 to 1 is approximately {result}")