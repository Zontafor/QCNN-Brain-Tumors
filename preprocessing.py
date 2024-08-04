import os
import numpy as np
import h5py
import glob

from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.utils import to_categorical


class Preprocessing:

    def __init__(
        self, input: str, output: str, batch_size: int, num_components: int
    ):
        self.input = input
        self.output = output
        self.total_images_loaded = 0
        self.batch_size = batch_size
        self.num_components = num_components
        self.svds = None

    def _check_directories(self):
        if not os.path.exists(self.input):
            raise IOError(f"No such directory: {self.input}")
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def _check_image_count(self):
        _, _, images = next(os.walk(self.input))
        image_count = len(images)
        if image_count != self.total_images_loaded:
            raise Exception(
                "Not all images have been loaded from the input directory"
            )

    def _load_images(self):
        images = []
        self._check_directories()
        for folder in os.listdir(self.input):
            path = os.path.join(self.input, folder)
            if os.path.isdir(path):
                for file in glob.glob(os.path.join(path, "*.h5")):
                    try:
                        with h5py.File(file, "r") as f:
                            data = f["image"][()]
                            images.append(data)
                            self.total_images_loaded += 1
                            if len(images) >= self.batch_size:
                                yield np.array(images, dtype=np.float32)
                                images = []
                    except Exception as e:
                        raise Exception(f"Error loading image: {file} - {e}")
        if images:
            yield np.array(images, dtype=np.float32)
        self._check_image_count()

    def svd(self):
        self.svds = []
        for batch_index, batch in enumerate(self._load_images()):
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
                self.output, f"X_reduced_batch_{batch_index}.npy"
            )
            batch_file_y = os.path.join(
                self.output, f"y_batch_{batch_index}.npy"
            )

            np.save(batch_file_X, X_reduced_batch)
            np.save(batch_file_y, y_batch)

            print(f"Saved batch {batch_index} to disk.")
