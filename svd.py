import os
import pickle
import numpy as np

from preprocessing import Preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class SVD:

    def __init__(
        self, input: str, output: str, batch_size: int, num_components: int
    ):
        self.input = input
        self.output = output
        self.batch_size = batch_size
        self.processed = None
        self.num_components = num_components
        self.reduced_batch_size = None
        self.svds = None

    def _memap_dataset(self, mode: str):
        total_images = self.processed.total_images_loaded
        X_reduced_mmap = np.memmap(
            "X_reduced.dat",
            dtype=np.float32,
            mode=mode,
            shape=(total_images, self.num_components),
        )
        y_mmap = np.memmap(
            "y.dat",
            dtype=np.float32,
            mode=mode,
            shape=(total_images, 4),
        )
        return (X_reduced_mmap, y_mmap)

    def _save_memap(
        self,
    ):
        # Combine the intermediate files into memory-mapped array
        X_reduced_mmap, y_mmap = self._memap_dataset("w+")

        index = 0
        for batch_index in range(self.reduced_batch_size):
            batch_file_X = os.path.join(
                self.output, f"X_reduced_batch_{batch_index}.npy"
            )
            batch_file_y = os.path.join(
                self.output, f"y_batch_{batch_index}.npy"
            )

            X_reduced_batch = np.load(batch_file_X)
            y_batch = np.load(batch_file_y)

            current_batch_size = X_reduced_batch.shape[0]

            # Check if shapes match before saving to memory-mapped arrays
            try:
                X_reduced_mmap[
                    index : index + current_batch_size,
                    : X_reduced_batch.shape[1],
                ] = X_reduced_batch
                y_mmap[index : index + current_batch_size] = y_batch
                index += current_batch_size
                print(f"Loaded batch {batch_index} from disk.")
            except ValueError as e:
                raise ValueError(f"Error saving batch {batch_index}: {e}")

        # Ensure all data is written to disk
        X_reduced_mmap.flush()
        y_mmap.flush()

        if self.svds:
            # Save the SVD transformers for future use
            with open("svd_transformers.pkl", "wb") as f:
                pickle.dump(self.svds, f)

            print("Initial pre-processing and saving completed.")

    def svd(self):
        self.svds = []
        self.processed = Preprocessing(self.input, self.output, self.batch_size)
        for self.reduced_batch_size, batch in enumerate(
            self.processed.load_images()
        ):
            # Flatten the batch of images
            X_batch = batch.reshape(batch.shape[0], -1)

            # Apply SVD to the batch
            current_num_components = min(self.num_components, X_batch.shape[1])
            svd = TruncatedSVD(n_components=current_num_components)
            X_reduced_batch = svd.fit_transform(X_batch)

            self.svds.append(svd)

            # Simulate label data for demonstration purposes
            y_batch = np.random.randint(4, size=batch.shape[0])
            y_batch = to_categorical(y_batch, 4)

            # Save the reduced data and labels to disk as intermediate files
            batch_file_X = os.path.join(
                self.output, f"X_reduced_batch_{self.reduced_batch_size}.npy"
            )
            batch_file_y = os.path.join(
                self.output, f"y_batch_{self.reduced_batch_size}.npy"
            )

            np.save(batch_file_X, X_reduced_batch)
            np.save(batch_file_y, y_batch)

            print(f"Saved batch {self.reduced_batch_size} to disk.")

        self._save_memap()

    def get_dataset(self):
        X_reduced_mmap, y_mmap = self._memap_dataset("r")

        # Load the SVD transformers
        with open("svd_transformers.pkl", "rb") as f:
            svd_transformers = pickle.load(f)

        # Prepare the reduced dataset
        X_reduced = []
        y_reduced = []

        total_images_loaded = self.processed.total_images_loaded
        for i in range(len(svd_transformers)):
            start_index = i * 64
            end_index = min((i + 1) * 64, total_images_loaded)
            if start_index >= total_images_loaded:
                break

            X_reduced_batch = X_reduced_mmap[start_index:end_index]
            y_batch = y_mmap[start_index:end_index]

            X_reduced.append(X_reduced_batch)
            y_reduced.append(y_batch)

        X_reduced = np.vstack(X_reduced)
        y_reduced = np.vstack(y_reduced)

        # Convert one-hot encoded labels to 1D array of class labels
        y_reduced_1d = np.argmax(y_reduced, axis=1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y_reduced_1d, test_size=0.2, random_state=42
        )

        return (X_train, X_test, y_train, y_test)
