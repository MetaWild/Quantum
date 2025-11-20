import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import matplotlib.pyplot as plt

def probabilistic_transistors(subkey, probability, num_transistors):
    return random.bernoulli(subkey, p=probability, shape=(num_transistors,))

def standard_uniform(num_transistors, key, num_samples):
    subkeys = random.split(key, num_samples)
    bits = vmap(lambda sk: probabilistic_transistors(sk, 0.5, num_transistors))(subkeys)
    bits = bits.astype(float)
    exponents = jnp.arange(1,num_transistors+1)
    exponents = exponents.astype(float)
    fractions = bits * (2 ** -exponents)
    return jnp.sum(fractions, axis=1)

# def probabilistic_transistor():
#     return np.random.randint(0, 2)

# def standard_uniform(num_transistors=10):
#     lower_bound = 0.0
#     upper_bound = 1.0
    
#     for _ in range(num_transistors):
#         midpoint = (lower_bound + upper_bound) / 2.0
#         output = probabilistic_transistor()
        
#         if output == 0:
#             upper_bound = midpoint
#         else:
#             lower_bound = midpoint
    
#     random_number = (lower_bound + upper_bound) / 2.0
#     return random_number

# num_samples = 1000000
# key = random.PRNGKey(0)
# num_transistors = 10
# random_numbers = standard_uniform(num_transistors, key, num_samples)

# plt.hist(random_numbers, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
# plt.title('Histogram of Generated Random Numbers')
# plt.xlabel('Random Number')
# plt.ylabel('Density')
# plt.show()