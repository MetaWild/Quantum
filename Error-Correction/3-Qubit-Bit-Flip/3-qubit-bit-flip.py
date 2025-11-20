from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import random

def build_circuit(input_state = "0"):

    bit_flip = QuantumCircuit(5)           

    c_syndrome = ClassicalRegister(2, "syndrome")
    c_data = ClassicalRegister(1, "data")

    bit_flip.add_register(c_syndrome)
    bit_flip.add_register(c_data)

    # Prepare input state
    if input_state == "1":
        bit_flip.x(0)
    elif input_state == "random":
        theta = random.uniform(0, 2 * np.pi)
        phi   = random.uniform(0, 2 * np.pi)
        bit_flip.ry(theta, 0)
        bit_flip.rz(phi, 0)

    # Encode logical qubit into 3 physical qubits
    bit_flip.cx(0,1)
    bit_flip.cx(0,2)

    # Introduce a random bit flip error on one of the three qubits
    i = random.randrange(4)

    if i != 0:
        bit_flip.x(i - 1)
        print("Flipped Qubit", i - 1)

    #Parity Checks to detect bit flip
    bit_flip.cx(0,3)
    bit_flip.cx(1,3)
    bit_flip.cx(1,4)
    bit_flip.cx(2,4)

    # Measure ancilla bits
    bit_flip.measure([3,4],c_syndrome)

    # Correct bit flip based on syndrome bits
    with bit_flip.if_test((c_syndrome, 0b01)):
        bit_flip.x(0)

    with bit_flip.if_test((c_syndrome, 0b11)):
        bit_flip.x(1)

    with bit_flip.if_test((c_syndrome, 0b10)):
        bit_flip.x(2)

    # Decode state back to one qubit
    bit_flip.cx(0,2)
    bit_flip.cx(0,1)

    if input_state == "random":
        bit_flip.rz(-phi, 0)
        bit_flip.ry(-theta, 0)

    bit_flip.measure(0, c_data)

    return bit_flip


sim = AerSimulator()

def run_single_trial(input_state: str, num_trials: int) -> int:
    """Return 1 if logical output matches input, else 0."""
    qc = build_circuit(input_state) 

    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=1).result()
    counts = result.get_counts()

    bitstring = next(iter(counts.keys()))

    logical_bit = bitstring[0]
    x = bitstring[2:4]
    
    if num_trials <= 50:
        if x == "01":
            print(f"Corrected X on qubit {0}")
        elif x == "11":
            print(f"Corrected X on qubit {1}")
        elif x == "10":
            print(f"Corrected X on qubit {2}")
    
    if input_state == "0":
        expected = "0"
    elif input_state == "1":
        expected = "1"
    elif input_state == "random":
        expected = "0"  # assuming undo rotation before measurement

    return 1 if logical_bit == expected else 0

input_state_list = ["0","1","random"] # options for input state
input_state = input_state_list[2]
num_trials = 10
successes = 0

for _ in range(num_trials):
    successes += run_single_trial(input_state, num_trials)

success_rate = successes / num_trials
print(f"Success rate for logical |{input_state}> with random single-qubit X flip: {success_rate:.3f}")