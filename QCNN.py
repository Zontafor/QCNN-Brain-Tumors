import pennylane as qml

from svd import SVD
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


class QCNN:

    def __init__(self, total_images: int, num_components: int, n_qubits: int):
        self.total_images = total_images
        self.num_components = num_components
        self.n_qubits = 4
        self.device = qml.device("default.qubit", wires=n_qubits)
