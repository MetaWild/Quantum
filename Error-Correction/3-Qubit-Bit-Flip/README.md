## 3-Qubit Bit-Flip Code

## Description

This project demonstrates the 3-qubit repetition (bit-flip) code: a single logical qubit is encoded across three physical qubits, a random single-qubit X (bit-flip) error is injected on one of the physical qubits, syndrome bits are measured using two ancilla qubits to detect which physical qubit (if any) flipped, and a corrective X is applied based on the syndrome. Finally the logical qubit is decoded and measured.

The included script (`3-qubit-bit-flip.py`) runs repeated trials where, for each trial, the circuit is built with a chosen input state (|0>, |1>, or a 'random' rotated state), a random X error may be introduced, and one shot is simulated. The script extracts the measured logical bit from the output bitstring and compares it with the expected logical value. A trial counts as a success when the final measured logical bit equals the expected input.

Success rate is computed as:

- successes = number of trials where the measured logical bit == expected logical bit
- success_rate = successes / num_trials

Notes:
- The script uses one shot per trial (shots=1) and repeats trials to build statistics.
- For the `random` input option the script applies and then undoes a random rotation around the prepared qubit so the measurement expects a particular outcome (the script assumes the rotation is undone before measuring the logical qubit).

### Parity checks and syndrome mapping

Parity (syndrome) extraction uses two ancilla qubits. Each ancilla records the parity (XOR) between a pair of data qubits: the first ancilla compares qubits 0 and 1, the second compares qubits 1 and 2. These comparisons are implemented with CNOTs from the data qubits into the ancilla and then measuring the ancilla.

Measured ancilla bits form a 2-bit syndrome that identifies which (if any) physical qubit experienced a bit-flip:

- syndrome = (1,0)  => error on qubit 0
- syndrome = (1,1)  => error on qubit 1
- syndrome = (0,1)  => error on qubit 2
- syndrome = (0,0)  => no detected single-bit X error

The script uses these syndrome patterns to apply a corrective X to the identified physical qubit before decoding. That corrective step is what allows the logical qubit to be recovered without ever measuring the logical information directly.
