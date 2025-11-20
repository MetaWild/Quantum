import jax
import jax.numpy as jnp
from jax.numpy.linalg import lstsq
import numpy as np
from uniform import standard_uniform
from pdf_integral import gauss_int
from pdf_integral import gauss_int_jax
import matplotlib.pyplot as plt

def sample_from(f, bin_indices, xi_grid, xi1_grid, num_qubits, key):
    x_i = xi_grid[bin_indices, :]
    x_i1 = xi1_grid[bin_indices, :]
    subkeys = jax.random.split(key, x_i.shape[0])
    samples = jax.vmap(lambda xi, xi1, subkey: sample_between(xi, xi1, subkey, num_qubits))(x_i, x_i1, subkeys)
    # function_values = jax.vmap(lambda args: f(*args))(samples)
    return samples

def sample_between(xi, xi1, key, num_qubits):
    subkeys = jax.random.split(key, xi.shape[0])
    random_values = jax.vmap(lambda xi, xi1, subkey: xi + (xi1 - xi) * standard_uniform(num_qubits, subkey, 1)[0])(xi, xi1, subkeys)
    return random_values

# def sample_from(f, bin_indices, h, a, num_transistors, key):
#     x_i = bin_indices * h + a
#     x_i1 = (bin_indices + 1) * h + a
#     x_points = jax.vmap(lambda xi, xi1: jnp.linspace(xi, xi1, 4))(x_i, x_i1)
#     y_points = f(x_points)
#     normalized_coefficients = normalized_polynomial_coefficients(x_points, y_points)
#     subkeys = jax.random.split(key, normalized_coefficients.shape[0])
#     return jax.vmap(lambda coeffs, xi, xi1, subkey: polynomials_sample(coeffs, 4, xi, xi1, num_transistors, subkey))(normalized_coefficients, x_i, x_i1, subkeys)

# def normalized_polynomial_coefficients(x_points, y_points):
#     def polynomial_fit(x_points, y_points):
#         coefficients = polyfit(x_points, y_points, 3)

#         x1, x4 = x_points[0], x_points[-1]

#         area = integrate_polynomial(coefficients, x1, x4)
#         normalized_coefficients = coefficients / area
#         return normalized_coefficients
    
#     return jax.vmap(polynomial_fit)(x_points, y_points)

# def polyfit(x, y, degree):
#     X = jnp.vander(x, N=degree + 1)
#     coeffs, residuals, rank, s = lstsq(X, y, rcond=None)
#     return coeffs

# def integrate_polynomial(coefficients, a, b):
#     n = len(coefficients)
#     powers = jnp.arange(n - 1, -1, -1)
#     antiderivative_coeffs = coefficients / (powers + 1)
#     antiderivative_coeffs = jnp.append(antiderivative_coeffs, 0)

#     integral_b = jnp.polyval(antiderivative_coeffs, b)
#     integral_a = jnp.polyval(antiderivative_coeffs, a)

#     return integral_b - integral_a


# def polynomial_function(coefficients):
#     def poly(x):
#         return jnp.polyval(coefficients, x)
#     return poly


# def polynomials_sample(coefficients, B, a , b, num_transistors, key):
#     h = (b - a) / B
#     x_i = jnp.arange(B) * h + a
#     x_i1 = (jnp.arange(B) + 1) * h + a
#     probabilities = jax.vmap(lambda xi, xi1: gauss_int_jax(coefficients,xi, xi1))(x_i, x_i1)
#     probabilities /= jnp.sum(probabilities)
#     cumulative_probabilities = jnp.cumsum(probabilities)
#     random_numbers = standard_uniform(num_transistors, key, 2)
#     random_number = random_numbers[0]
#     bin_index = jnp.searchsorted(cumulative_probabilities, random_number, side='right')
#     bin_start = bin_index * h + a
#     bin_end = (bin_index + 1) * h + a
#     sample_random_number = random_numbers[1]
#     return sample_random_number * (bin_end - bin_start) + bin_start

# def sample_from(f, i, h, a):
#     x_i = i * h + a
#     x_i1 = (i + 1) * h + a
#     x_points = np.linspace(x_i, x_i1, 4)
#     y_points = f(x_points)
#     polynomial = normalized_polynomial(x_points, y_points)
#     return polynomial_sample(polynomial, 4, x_i, x_i1)


# def normalized_polynomial(x_points, y_points):
#     coefficients = np.polyfit(x_points, y_points, 3)
#     polynomial = np.poly1d(coefficients)

#     integral_poly = np.polyint(polynomial)

#     x1, x4 = x_points[0], x_points[-1]

#     area = integral_poly(x4) - integral_poly(x1)

#     normalized_coefficients = coefficients / area
#     normalized_polynomial = np.poly1d(normalized_coefficients)

#     return normalized_polynomial

# def polynomial_sample(polynomial, B, a , b):
#     probabilities = np.zeros(B)
#     h = (b - a) / B
#     for i in range(B):
#         x_i = i * h + a
#         x_i1 = (i + 1) * h + a
#         probabilities[i] = gauss_int(polynomial, x_i, x_i1)
#     probabilities /= sum(probabilities)
#     cumulative_probabilities = np.zeros(B)
#     for i in range(B):
#         cumulative_probabilities[i] = np.sum(probabilities[:i + 1])
    
#     random_number = standard_uniform(10)
#     for i in range(B):
#         if cumulative_probabilities[i] >= random_number:
#             x_i = i * h + a
#             x_i1 = (i + 1) * h + a
#             return standard_uniform(10) * (x_i1 - x_i) + x_i



# x_points = np.array([0, 0.3333, 0.6666, 1])
# y_points = np.array([1, 0.5, 1, 0.5])

# norm_poly = normalized_polynomial(x_points, y_points)

# print("Normalized Polynomial Equation:", norm_poly)

# x_value = 0.55
# y_value = norm_poly(x_value)
# print(f"Value at x = {x_value}: y = {y_value}")

# x1, x4 = x_points[0], x_points[-1]
# integral_poly = np.polyint(norm_poly)
# computed_area = integral_poly(x4) - integral_poly(x1)
# print(f"Verified that the integral from {x1} to {x4} is approximately: {computed_area}")

# num_samples = 10000
# random_samples = [polynomial_sample(norm_poly, 4, x1, x4) for _ in range(num_samples)]

# plt.hist(random_samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
# plt.title('Histogram of Generated Random Numbers')
# plt.xlabel('Random Number')
# plt.ylabel('Density')
# plt.show()


