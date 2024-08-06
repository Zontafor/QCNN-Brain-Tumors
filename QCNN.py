import pennylane as qml
import matplotlib.pyplot as plt

from svd import SVD
from tensorflow import py_function, map_fn, reshape, expand_dims
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
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


class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        print(f"QuantumLayer initialized with {self.n_qubits} qubits")

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            shape=(self.n_qubits,),
            initializer="random_normal",
            trainable=True,
            name="q_weights",
        )
        print(
            f"QuantumLayer build with input shape {input_shape} and weights shape {self.q_weights.shape}"
        )
        super(QuantumLayer, self).build(input_shape)

    def call(self, inputs):
        print(f"QuantumLayer call with inputs shape {inputs.shape}")
        inputs = reshape(inputs, [-1, 16])  # Ensure the input has length 16
        quantum_output = map_fn(
            lambda x: py_function(
                quantum_circuit, [x, self.q_weights], tf.float32
            ),
            inputs,
        )
        quantum_output.set_shape([None, self.n_qubits])
        return quantum_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


def qcnn_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    print(f"Model input shape: {inputs.shape}")

    # Reshape the input to 2D (batch_size, input_shape, 1)
    x = layers.Reshape((input_shape, 1))(inputs)

    # Classical convolutional layers
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)

    # Flatten before quantum layer
    x = layers.Flatten()(x)

    # Quantum layer
    x = QuantumLayer(n_qubits)(x)

    # Output layer
    outputs = layers.Dense(4, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    input = "BraTS2020_training_data/content"
    output = "Outputs"

    svd = SVD(input, output, 64, 95)
    svd.svd()
    X_train, X_test, y_train, y_test = svd.get_dataset()

    qcnn = qcnn_model(svd.num_components)
    qcnn.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Convert y_train and y_test back to one-hot encoding for QCNN training
    y_train_one_hot = to_categorical(y_train, 4)
    y_test_one_hot = to_categorical(y_test, 4)
    print(
        f"y_train_one_hot shape: {y_train_one_hot.shape}, y_test_one_hot shape: {y_test_one_hot.shape}"
    )

    X_train_expanded = expand_dims(X_train, axis=-1)
    X_test_expanded = expand_dims(X_test, axis=-1)
    print(
        f"X_train_expanded shape: {X_train_expanded.shape}, X_test_expanded shape: {X_test_expanded.shape}"
    )

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train the model with early stopping
    print("Starting model training")
    history = qcnn_model.fit(
        X_train_expanded,
        y_train_one_hot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    print("Model training completed")

    loss, accuracy = qcnn_model.evaluate(X_test_expanded, y_test_one_hot)
    print(f"Test accuracy: {accuracy}")

    # Save the model
    qcnn_model.save("qcnn_model.h5")
    print("Model saved")

    # Plotting the number of epochs vs accuracy
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
