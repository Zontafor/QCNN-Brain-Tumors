import pennylane as qml

from svd import SVD
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

n_qubits = 4
device = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(device)
def quantum_circuit(inputs, weights):
    qml.templates.AmplitudeEmbedding(
        inputs, wires=range(n_qubits), normalize=True
    )
    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



def main():
    input = "BraTS2020_training_data/content"
    output = "Outputs"

    svd = SVD(input, output, 64, 95)
    svd.svd()
    X_train, X_test, y_train, y_test = svd.get_dataset()


if __name__ == "__main__":
    main()
