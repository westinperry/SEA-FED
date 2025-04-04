# utils/logger.py

from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images. Expects images as [N, C, H, W] or [C, H, W] numpy arrays."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image
                self.writer.add_image(tag, images, step)
            elif images.ndim == 4:  # Batch of images
                for i, img in enumerate(images):
                    self.writer.add_image(f'{tag}/{i}', img, step)
            else:
                raise ValueError("Unsupported image shape")
        else:
            raise TypeError("Expected images as NumPy array")

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins=bins)

    def close(self):
        self.writer.close()
