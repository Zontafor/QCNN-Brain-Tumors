import os
import h5py
import glob
import numpy as np


class Preprocessing:

    def __init__(self, input: str, output: str, batch_size: int):
        self.input = input
        self.output = output
        self.batch_size = batch_size
        self.total_images_loaded = 0

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

    def load_images(self):
        images = []
        self._check_directories()
        if os.path.isdir(self.input):
            for file in glob.glob(os.path.join(self.input, "*.h5")):
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
