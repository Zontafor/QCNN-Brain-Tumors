import os
import numpy as np
import glob
import h5py
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
import pickle
import pennylane as qml
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# # Create tumor, edema, and necrotic masks
# # Create a list of all .h5 files in the directory
# h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]
# print(f"Found {len(h5_files)} .h5 files:\nExample file names:{h5_files[:3]}")

# # Open the first .h5 file in the list to inspect its contents
# if h5_files:
#     file_path = os.path.join(directory, h5_files[25070])
#     with h5py.File(file_path, "r") as file:
#         print("\nKeys for each file:", list(file.keys()))
#         for key in file.keys():
#             print(f"\nData type of {key}:", type(file[key][()]))
#             print(f"Shape of {key}:", file[key].shape)
#             print(f"Array dtype: {file[key].dtype}")
#             print(f"Array max val: {np.max(file[key])}")
#             print(f"Array min val: {np.min(file[key])}")
# else:
#     print("No .h5 files found in the directory.")


# # Pre-processing with SVD (dimensionality reduction)
# # Function to load images
# def load_images(data_dir, batch_size=64):
#     images = []
#     total_images_loaded = 0
#     for patient_folder in os.listdir(data_dir):
#         patient_path = os.path.join(data_dir, patient_folder)
#         if os.path.isdir(patient_path):
#             for file in glob.glob(os.path.join(patient_path, "*.h5")):
#                 try:
#                     with h5py.File(file, "r") as f:
#                         image_data = f["image"][()]
#                         images.append(image_data)
#                         total_images_loaded += 1
#                         if len(images) >= batch_size:
#                             yield np.array(images, dtype=np.float32)
#                             images = []
#                 except Exception as e:
#                     print(f"Error loading image: {file} - {e}")
#     if images:
#         yield np.array(images, dtype=np.float32)
#     print(f"Total images loaded: {total_images_loaded}")


# data_dir = r"C:\Users\Michelle Wu\OneDrive\Desktop\UCR\MATH\MATH194\BraTS2020 Data\archive\BraTS2020_training_data\content"
# output_dir = (
#     r"C:\Users\Michelle Wu\OneDrive\Desktop\UCR\MATH\MATH194\Code Outputs"
# )

# if not os.path.exists(data_dir):
#     print(
#         f"The directory {data_dir} does not exist. Please provide the correct path."
#     )
#     sys.exit()

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Calculate the total number of images
# total_images_loaded = sum(
#     [len(files) for r, d, files in os.walk(data_dir) if files]
# )

# # Define parameters
# num_components = 95
# batch_size = 64

# index = 0
# batch_index = 0

# svd_transformers = []

# for batch in load_images(data_dir, batch_size):
#     # Flatten the batch of images
#     X_batch = batch.reshape(batch.shape[0], -1)
#     print(f"Original shape of X_batch: {X_batch.shape}")

#     # Apply SVD to the batch
#     current_num_components = min(num_components, X_batch.shape[1])
#     svd = TruncatedSVD(n_components=current_num_components)
#     X_reduced_batch = svd.fit_transform(X_batch)
#     print(f"Reduced shape of X_batch: {X_reduced_batch.shape}")

#     svd_transformers.append(svd)

#     # Simulate label data for demonstration purposes
#     y_batch = np.random.randint(4, size=batch.shape[0])
#     y_batch = tf.keras.utils.to_categorical(y_batch, 4)

#     # Save the reduced data and labels to disk as intermediate files
#     batch_file_X = os.path.join(
#         output_dir, f"X_reduced_batch_{batch_index}.npy"
#     )
#     batch_file_y = os.path.join(output_dir, f"y_batch_{batch_index}.npy")

#     np.save(batch_file_X, X_reduced_batch)
#     np.save(batch_file_y, y_batch)

#     print(f"Saved batch {batch_index} to disk.")
#     batch_index += 1

# # Combine the intermediate files into memory-mapped arrays
# X_reduced_mmap = np.memmap(
#     "X_reduced.dat",
#     dtype=np.float32,
#     mode="w+",
#     shape=(total_images_loaded, num_components),
# )
# y_mmap = np.memmap(
#     "y.dat", dtype=np.float32, mode="w+", shape=(total_images_loaded, 4)
# )

# index = 0
# for batch_index in range(batch_index):
#     batch_file_X = os.path.join(
#         output_dir, f"X_reduced_batch_{batch_index}.npy"
#     )
#     batch_file_y = os.path.join(output_dir, f"y_batch_{batch_index}.npy")

#     X_reduced_batch = np.load(batch_file_X)
#     y_batch = np.load(batch_file_y)

#     current_batch_size = X_reduced_batch.shape[0]

#     # Check if shapes match before saving to memory-mapped arrays
#     try:
#         X_reduced_mmap[
#             index : index + current_batch_size, : X_reduced_batch.shape[1]
#         ] = X_reduced_batch
#         y_mmap[index : index + current_batch_size] = y_batch
#         index += current_batch_size
#         print(f"Loaded batch {batch_index} from disk.")
#     except ValueError as e:
#         print(f"Error saving batch {batch_index}: {e}")
#         continue

# # Ensure all data is written to disk
# X_reduced_mmap.flush()
# y_mmap.flush()

# # Save the SVD transformers for future use
# with open("svd_transformers.pkl", "wb") as f:
#     pickle.dump(svd_transformers, f)

# print("Initial pre-processing and saving completed.")

# Create and train QCNN
# Constants
# total_images_loaded = 57195  # Update with the actual total number of images
# num_components = 64

# # Load memory-mapped arrays
# X_reduced_mmap = np.memmap(
#     "X_reduced.dat",
#     dtype=np.float32,
#     mode="r",
#     shape=(total_images_loaded, num_components),
# )
# y_mmap = np.memmap(
#     "y.dat", dtype=np.float32, mode="r", shape=(total_images_loaded, 4)
# )

# # Load the SVD transformers
# with open("svd_transformers.pkl", "rb") as f:
#     svd_transformers = pickle.load(f)

# # Prepare the reduced dataset
# X_reduced = []
# y_reduced = []

# for i in range(len(svd_transformers)):
#     start_index = i * 64
#     end_index = min((i + 1) * 64, total_images_loaded)
#     if start_index >= total_images_loaded:
#         break

#     X_reduced_batch = X_reduced_mmap[start_index:end_index]
#     y_batch = y_mmap[start_index:end_index]

#     X_reduced.append(X_reduced_batch)
#     y_reduced.append(y_batch)

# X_reduced = np.vstack(X_reduced)
# y_reduced = np.vstack(y_reduced)

# # Convert one-hot encoded labels to 1D array of class labels
# y_reduced_1d = np.argmax(y_reduced, axis=1)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X_reduced, y_reduced_1d, test_size=0.2, random_state=42
# )

# Define the quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# Define a simple quantum circuit
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AmplitudeEmbedding(
        inputs, wires=range(n_qubits), normalize=True
    )
    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# Define the layer that will use the quantum circuit
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
        inputs = tf.reshape(inputs, [-1, 16])  # Ensure the input has length 16
        quantum_output = tf.map_fn(
            lambda x: tf.py_function(
                quantum_circuit, [x, self.q_weights], tf.float32
            ),
            inputs,
        )
        quantum_output.set_shape([None, self.n_qubits])
        return quantum_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


# Define the QCNN model
def create_qcnn_model(input_shape):
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


# Create and train the QCNN model
input_shape = num_components  # Using reduced feature size for training
print(f"Creating model with input shape {input_shape}")
qcnn_model = create_qcnn_model(input_shape)

# Compile the model
qcnn_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
print("Model compiled")

# Convert y_train and y_test back to one-hot encoding for QCNN training
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 4)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 4)
print(
    f"y_train_one_hot shape: {y_train_one_hot.shape}, y_test_one_hot shape: {y_test_one_hot.shape}"
)

# Expand the dimensions of X_train and X_test
X_train_expanded = tf.expand_dims(X_train, axis=-1)
X_test_expanded = tf.expand_dims(X_test, axis=-1)
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

# Evaluate the model
print("Evaluating model")
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


# Introduce GBA (incomplete)
# Function to apply GBA to enhance images
def enhance_images(images, model):
    enhanced_images = []
    for img in images:
        img_flat = img.flatten().reshape(1, -1)
        enhanced_img_flat = model.predict(img_flat)
        enhanced_img = enhanced_img_flat.reshape(img.shape)
        enhanced_images.append(enhanced_img)
    return np.array(enhanced_images)


# Assume `gba_model` is already trained and available
# Perform the inverse transformation and save each chunk
chunk_size = 16  # Example chunk size
for i in range(len(svd_transformers)):
    for chunk_index in range(0, 64, chunk_size):
        start_index = i * 64 + chunk_index
        end_index = min(start_index + chunk_size, total_images_loaded)
        if start_index >= total_images_loaded:
            break

        X_reduced_chunk = X_reduced_mmap[start_index:end_index]
        svd = svd_transformers[i]

        try:
            X_reconstructed_chunk = svd.inverse_transform(X_reduced_chunk)
            X_reconstructed_chunk = X_reconstructed_chunk.reshape(
                -1, 240, 240, 4
            )  # Example shape, update as needed

            # Apply the GBA model to enhance the reconstructed images
            enhanced_images_chunk = enhance_images(
                X_reconstructed_chunk, gba_model
            )

            # Save the reconstructed and enhanced chunk to an HDF5 file
            with h5py.File(
                os.path.join(
                    output_dir, f"X_enhanced_chunk_{i}_{chunk_index}.h5"
                ),
                "w",
            ) as hf:
                hf.create_dataset("data", data=enhanced_images_chunk)
            print(f"Saved enhanced chunk {i}_{chunk_index} to disk.")
        except ValueError as e:
            print(
                f"Error in chunk {i}_{chunk_index}: {e}. Skipping this chunk."
            )
            continue
