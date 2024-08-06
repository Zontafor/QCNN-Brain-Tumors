import os
import h5py
import numpy as np

class GBA:

    def __init__(self, output, total_images, X, y, svd, chunk_size, model):
        self.output = output
        self.total_images = total_images
        self.X = X
        self.y = y
        self.svd = svd
        self.chunk_size = chunk_size
        self.model = model

    # Introduce GBA (incomplete)
    # Function to apply GBA to enhance images
    def enhance_images(self, images, model):
        enhanced_images = []
        for img in images:
            img_flat = img.flatten().reshape(1, -1)
            enhanced_img_flat = model.predict(img_flat)
            enhanced_img = enhanced_img_flat.reshape(img.shape)
            enhanced_images.append(enhanced_img)
        return np.array(enhanced_images)

    def save_chunks(self):
        # Assume `gba_model` is already trained and available
        # Perform the inverse transformation and save each chunk
        for i in range(len(self.svd)):
            for chunk_index in range(0, 64, self.chunk_size):
                start_index = i * 64 + chunk_index
                end_index = min(
                    start_index + self.chunk_size, self.total_images
                )
                if start_index >= self.total_images:
                    break

                X_reduced_chunk = self.X[start_index:end_index]
                svd = self.svd[i]

                try:
                    X_reconstructed_chunk = svd.inverse_transform(
                        X_reduced_chunk
                    )
                    X_reconstructed_chunk = X_reconstructed_chunk.reshape(
                        -1, 240, 240, 4
                    )  # Example shape, update as needed

                    # Apply the GBA model to enhance the reconstructed images
                    enhanced_images_chunk = self.enhance_images(
                        X_reconstructed_chunk, self.model
                    )

                    # Save the reconstructed and enhanced chunk to an HDF5 file
                    with h5py.File(
                        os.path.join(
                            self.output,
                            f"X_enhanced_chunk_{i}_{chunk_index}.h5",
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
