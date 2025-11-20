# Shor's 9‑Qubit Code

## Description

This project contains an example implementation of Shor's 9‑qubit code. The code protects a single logical qubit against arbitrary single‑qubit Pauli errors by combining bit‑flip (X) repetition codes on three 3‑qubit GHZ blocks with a phase‑flip (Z) repetition across those blocks.

The included script (`9-qubit-shor-code`) builds a 17‑qubit circuit (9 data qubits + ancilla), prepares an input state (|0>, |1>, or a random rotated state), applies a random single‑qubit Pauli (I, X, Y, or Z) on one data qubit, performs syndrome extraction for both X (within each GHZ block) and Z (across blocks), applies corrective operations based on the measured syndromes, decodes the logical qubit back to a single qubit, and measures the logical result.

Success for a single trial is defined the same way as in the 3‑qubit example: after decoding, the measured logical bit is compared to the expected logical value (for `random` the script assumes the preparation rotation is undone before measurement). The script runs many trials (one shot per trial) and reports a success_rate = successes / num_trials.

## Parity / stabilizer extraction and syndrome mapping

Shor's code uses two layers of checks:

- X (bit‑flip) checks inside each 3‑qubit GHZ block. Each GHZ block (block 0: qubits [0,3,4], block 1: [1,5,6], block 2: [2,7,8]) is protected by a 3‑qubit repetition code. Two ancilla bits per block measure parity between pairs of the block's data qubits (implemented with CNOTs into ancilla and then measuring the ancilla). The resulting 2‑bit X‑syndrome for each block maps to which physical qubit in that block flipped (or no flip):

	- syndrome = (1,0) => X error on first qubit of the block (e.g., qubit 0 for block 0)
	- syndrome = (1,1) => X error on second qubit of the block (e.g., qubit 3 for block 0)
	- syndrome = (0,1) => X error on third qubit of the block (e.g., qubit 4 for block 0)
	- syndrome = (0,0) => no detected single‑bit X error in that block

	The script applies the corresponding corrective X to the identified data qubit in each block before proceeding.

- Z (phase‑flip) checks across blocks. After removing bit flips, two ancilla bits are used to measure parity across the three GHZ blocks (this is done by applying Hadamards and CNOTs to collect parity onto ancilla qubits, then measuring them). The resulting 2‑bit Z‑syndrome identifies which GHZ block (if any) suffered a logical Z error:

	- z‑syndrome = (1,0) => Z on block 0 (apply Z correction to representative qubit of block 0)
	- z‑syndrome = (1,1) => Z on block 1
	- z‑syndrome = (0,1) => Z on block 2
	- z‑syndrome = (0,0) => no detected block‑level Z error

	Because a Y error equals iXZ (combination of X and Z), the two layers together can detect and correct a Y as well by applying both corrections when indicated.

Together these checks form the stabilizer measurements for the Shor code: local X stabilizers inside blocks (detect bit flips) and block‑level Z stabilizers (detect phase flips). The key point is that only syndrome (error) information is extracted via ancilla measurements; the logical quantum state remains encoded and is not collapsed by these measurements.

## How results are taken / success rate

- Each trial builds the circuit, injects a random Pauli on a randomly chosen data qubit, runs one shot (shots=1), and records the classical registers containing:
	- the decoded logical measurement (the recovered logical bit)
	- the X syndromes for each block and the Z syndrome across blocks (used for logging/debug prints in low trial counts)
- A trial counts as a success when the decoded logical measurement equals the expected logical value (the code undoes any random preparation rotation for the `random` input option, so the expected is known).
- The script repeats this for `num_trials` and reports success_rate = successes / num_trials.

