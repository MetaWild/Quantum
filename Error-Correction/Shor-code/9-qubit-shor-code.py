from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import random

def build_shor_circuit(input_state = "0"):
    shor = QuantumCircuit(17)

    shor_x0_syndrome = ClassicalRegister(2, "x0_syndrome")
    shor_x1_syndrome = ClassicalRegister(2, "x1_syndrome")
    shor_x2_syndrome = ClassicalRegister(2, "x2_syndrome")
    shor_z_syndrome = ClassicalRegister(2, "z_syndrome")
    shor_data = ClassicalRegister(1, "shor_data")

    shor.add_register(
        shor_x0_syndrome,
        shor_x1_syndrome,
        shor_x2_syndrome,
        shor_z_syndrome,
        shor_data,
    )

    # Prepare input state
    if input_state == "1":
        shor.x(0)
    elif input_state == "random":
        theta = random.uniform(0, 2 * np.pi)
        phi   = random.uniform(0, 2 * np.pi)
        shor.ry(theta, 0)
        shor.rz(phi, 0)

    # Encode Shore Code
    shor.cx(0,1)
    shor.cx(0,2)
    shor.h(0)
    shor.h(1)
    shor.h(2)


    # Continue Encoding by creating GHZ blocks
    shor.cx(0,3)
    shor.cx(0,4)

    shor.cx(1,5)
    shor.cx(1,6)

    shor.cx(2,7)
    shor.cx(2,8)

    i = random.randrange(9)

    pauli = random.choice(['I', 'X', 'Y', 'Z'])

    if pauli == 'X':
        shor.x(i)
    elif pauli == 'Y':
        shor.y(i)
    elif pauli == 'Z':
        shor.z(i)

    print(f"Applied {pauli} on qubit {i}")

    # Apply 3-Qubit X Correction on first GHZ block which consists of qubits 0,3,4
    shor.cx(0,9)
    shor.cx(3,9)
    shor.cx(3,10)
    shor.cx(4,10)

    shor.measure([9,10],shor_x0_syndrome)

    with shor.if_test((shor_x0_syndrome, 0b01)):
        shor.x(0)

    with shor.if_test((shor_x0_syndrome, 0b11)):
        shor.x(3)

    with shor.if_test((shor_x0_syndrome, 0b10)):
        shor.x(4)

    # Apply 3-Qubit X Correction on first GHZ block which consists of qubits 1,5,6
    shor.cx(1,11)
    shor.cx(5,11)
    shor.cx(5,12)
    shor.cx(6,12)

    shor.measure([11,12],shor_x1_syndrome)

    with shor.if_test((shor_x1_syndrome, 0b01)):
        shor.x(1)

    with shor.if_test((shor_x1_syndrome, 0b11)):
        shor.x(5)

    with shor.if_test((shor_x1_syndrome, 0b10)):
        shor.x(6)


    # Apply 3-Qubit X Correction on first GHZ block which consists of qubits 2,7,8
    shor.cx(2,13)
    shor.cx(7,13)
    shor.cx(7,14)
    shor.cx(8,14)

    shor.measure([13,14],shor_x2_syndrome)

    with shor.if_test((shor_x2_syndrome, 0b01)):
        shor.x(2)

    with shor.if_test((shor_x2_syndrome, 0b11)):
        shor.x(7)

    with shor.if_test((shor_x2_syndrome, 0b10)):
        shor.x(8)

    # Correcting Z error on block level  
    shor.h(15)
    for q in [0,3,4,1,5,6]:
        shor.cx(15,q)
    shor.h(15)

    shor.h(16)
    for q in [1,5,6,2,7,8]:
        shor.cx(16,q)
    shor.h(16)

    shor.measure([15,16], shor_z_syndrome)

    with shor.if_test((shor_z_syndrome, 0b01)):
        shor.z(0)

    with shor.if_test((shor_z_syndrome, 0b11)):
        shor.z(1)

    with shor.if_test((shor_z_syndrome, 0b10)):
        shor.z(2)

    # Decode Shor code back to one qubit
    shor.cx(2,8)
    shor.cx(2,7)
    shor.cx(1,6)
    shor.cx(1,5)
    shor.cx(0,4)
    shor.cx(0,3)

    shor.h(2)
    shor.h(1)
    shor.h(0) 

    shor.cx(0,2)
    shor.cx(0,1)

    if input_state == "random":
        shor.rz(-phi, 0)
        shor.ry(-theta, 0)

    #[0,1,2,3,4,5,6,7,8]
    shor.measure(0, shor_data)

    return shor



sim = AerSimulator()

def run_single_trial(input_state: str, num_trials: int) -> int:
    """Return 1 if logical output matches input, else 0."""
    qc = build_shor_circuit(input_state) 

    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=1).result()
    counts = result.get_counts()

    bitstring = next(iter(counts.keys()))

    logical_bit = bitstring[0]
    z = bitstring[2:4]
    x2 = bitstring[5:7]
    x1 = bitstring[8:10]
    x0 = bitstring[11:13]
    
    if num_trials <= 50:
        if x0 == "01":
            print(f"Corrected X on qubit {0}")
        elif x0 == "11":
            print(f"Corrected X on qubit {3}")
        elif x0 == "10":
            print(f"Corrected X on qubit {4}")

        if x1 == "01":
            print(f"Corrected X on qubit {1}")
        elif x1 == "11":
            print(f"Corrected X on qubit {5}")
        elif x1 == "10":
            print(f"Corrected X on qubit {6}")

        if x2 == "01":
            print(f"Corrected X on qubit {2}")
        elif x2 == "11":
            print(f"Corrected X on qubit {7}")
        elif x2 == "10":
            print(f"Corrected X on qubit {8}")

        if z == "01":
            print(f"Corrected Z on block {0} with qubits 0,3,4")
        elif z == "11":
            print(f"Corrected Z on block {1} with qubits 1,5,6")
        elif z == "10":
            print(f"Corrected Z on block {2} with qubits 2,7,8")
    

    if input_state == "0":
        expected = "0"
    elif input_state == "1":
        expected = "1"
    elif input_state == "random":
        expected = "0"  # assuming undo rotation before measurement

    return 1 if logical_bit == expected else 0

input_state_list = ["0","1","random"] # options for input state
input_state = input_state_list[0]
num_trials = 100
successes = 0

for _ in range(num_trials):
    successes += run_single_trial(input_state, num_trials)

success_rate = successes / num_trials
print(f"Success rate for logical |{input_state}> with random single-qubit Pauli: {success_rate:.3f}")