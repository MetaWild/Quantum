# Error Correction

## About this folder

This directory collects projects, reference implementations, and notes about Quantum Error Correction (QEC). QEC is pivotal for making quantum computation practical: qubits are fragile and interact with their environment, causing errors that accumulate and quickly destroy computation unless corrected.

One of the central challenges of QEC is that quantum error correction must fix an unknown error on a qubit without measuring and thereby collapsing its quantum state. Naively measuring a qubit reveals its state and destroys quantum superposition. QEC resolves this apparent contradiction by encoding logical qubits into entangled states of multiple physical qubits and using ancillary qubits and syndrome measurements that reveal only error information (which qubit errored and what kind of error) while leaving the logical quantum information intact.

Key ideas used in these projects:

- Encoding: a single logical qubit is represented by an entangled state across several physical qubits (e.g., the 3-qubit repetition code or Shor's 9-qubit code).
- Syndrome measurement: measure carefully chosen operators (stabilizers) whose outcomes indicate the presence and type of errors but commute with the logical operators, so they don't reveal the encoded quantum information.
- Ancilla qubits: temporary helper qubits are entangled with the data qubits and measured; only the ancilla are measured, so the data qubits' superposition survives.
- Recovery: based on syndrome outcomes, apply corrective gates to restore the logical state.

Why this is hard in practice:

- Errors are continuous: unlike classical bits, quantum errors can be arbitrary rotations; QEC discretizes them into a set of correctable error types (bit-flip, phase-flip, or combinations) using code structure.
- Measurement and decoherence: syndrome extraction itself must be implemented fault-tolerantly so that the correction procedure does not introduce more errors than it fixes.

## Structure

- `3-Qubit-Bit-Flip/` — Minimal demonstration of the 3-qubit repetition (bit-flip) code. Includes an example Python script (`3-qubit-bit-flip.py`) and a project README explaining the circuit and how syndrome measurement and recovery work for single bit-flip errors.
- `Shor-code/` — Materials for Shor's 9-qubit code which protects against both bit-flip and phase-flip errors by concatenating repetition codes. Contains explanation, example circuits, and reference implementation(s).

For more details, open the README inside each subfolder — those files contain per-project explanations, usage examples, and instructions for running the included scripts.

---

