# Sampling

## Description

This folder contains code for generating random samples from an arbitrary probability density function (PDF) using either probabilistic transistor (p‑bit) sampling or qubit-based amplitude sampling. The code is designed to:

- discretize a continuous PDF into bins,
- compute bin probabilities (via numerical integration / Monte Carlo),
- draw random bin indices from the resulting discrete distribution, and
- generate uniform samples inside the chosen bin to return continuous samples.

Files and how they are used:

- `pdf_sampler.py` — Overall top level file which builds the bin grid over [a,b], computes or requests bin probabilities (via `pdf_integral.py`), converts those probabilities into either a qubit amplitude distribution (for sampling with Qiskit) or uses p‑bit sampling, and finally calls `bin_sampling.sample_from` to turn chosen bins into continuous samples.
- `bin_sampling.py` — given bin indices and bin edge information, this module samples uniformly inside each selected bin. 
- `pdf_integral.py` — numerical integration helpers. It provides Gaussian quadrature for short integrals, a composite quadrature helper (`comp_gauss_int`) and a Monte Carlo integrator (`multidimensional_monte_carlo`) used to estimate bin probabilities for multi‑dimensional PDFs.
- `uniform.py` — provides a pseudo‑random generator built from probabilistic transistor-style sampling: `standard_uniform` constructs uniform numbers from many parallel Bernoulli draws (useful when simulating p‑bits). This is also used to provide randomness for the Monte Carlo integrators and internal sampling.

At a high level you can pick between two sampling paths in `pdf_sampler.py`:

- `distribution_qubit_samples(...)` — constructs amplitudes from the bin probabilities, initializes a Qiskit state, measures the qubit register to draw bin indices, then uses `bin_sampling.sample_from` to get continuous samples.
- `distribution_pbit_samples(...)` — uses Monte Carlo / probabilistic transistor sampling to estimate bin probabilities and draws samples using `standard_uniform` without constructing a quantum state.

Both routes return an array of samples shaped (N, D) where D is the number of dimensions of the PDF `f`.

## To start

Open `pdf_sampler.py` and inspect the sample test at the bottom (the `if __name__ == "__main__"` block). That section shows example settings (N, B, a, b, num_qubits/num_pbits, key) and a simple plotting flow. Set `random_samples = distribution_qubit_samples(...)` or `distribution_pbit_samples(...)` depending on which backend you want to exercise.

## Settings (quick reference)

- N — number of samples to generate. Larger N → more time and better statistics.
- B — number of bins along each dimension. Larger B → finer discretization and higher cost when computing bin probabilities.
- a, b — lower/upper bounds for the sampling range. PDFs with infinite support must be truncated to [a,b] for these simulations.
- num_transistors / num_pbits / num_qubits — controls internal parallelism or the size of the quantum register used for amplitude sampling.
- key — JAX PRNG key used for reproducibility.

Notes:

- The qubit-based path uses Qiskit/Aer; make sure Qiskit and qiskit-aer are installed if you plan to run `distribution_qubit_samples`.
- The Monte Carlo integral (`multidimensional_monte_carlo`) can be expensive for high dimensions; tune `num_transistors` and the internal `num_samples` in that function for performance vs accuracy.

## Example parameter tips

For heavier 3D or 2D sampling you'll likely need many more samples and bins; the existing test harness in `pdf_sampler.py` shows reasonable defaults for quick experiments.

## Run

Run the top-level sampler directly from the folder:

```bash
python pdf_sampler.py
```

This will execute the demonstration harness and show plots when appropriate.

## Other probability density functions available in `pdf_sampler.py` (examples)

- `standard_exponential_distribution = create_exponential_function(1)`
- `test_normal_distribution = create_normal_function(1, 0.5)`
- `test_normal_distribution_2d = create_normal_function_2d(1, 0.5)`
