# dehaze_model.py

import cv2
import numpy as np

class DehazeModel:
    def __init__(self, omega=0.95, size=15, t0=0.1):
        self.omega = omega
        self.size = size
        self.t0 = t0

    def get_dark_channel(self, image):
        """Calculate the dark channel of an image."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.size, self.size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def get_atmosphere(self, image, dark_channel, top_percent=0.001):
        """Estimate the atmospheric light."""
        total_pixels = dark_channel.size
        num_top_pixels = int(max(total_pixels * top_percent, 1))
        indices = np.argsort(dark_channel.ravel())[::-1][:num_top_pixels]
        brightest = np.unravel_index(indices, dark_channel.shape)
        atmosphere = np.mean(image[brightest], axis=0)
        return atmosphere

    def get_transmission(self, image, atmosphere):
        """Estimate the transmission map."""
        norm_image = image / atmosphere
        dark_channel = self.get_dark_channel(norm_image)
        transmission = 1 - self.omega * dark_channel
        return transmission

    def recover_image(self, image, transmission, atmosphere):
        """Recover the dehazed image using the estimated transmission and atmosphere."""
        transmission = np.clip(transmission, self.t0, 1)
        transmission = transmission[:, :, np.newaxis]
        recovered = (image - atmosphere) / transmission + atmosphere
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        return recovered
