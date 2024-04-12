import numpy as np
from detector.detector import Detector
import cv2
from matplotlib import pyplot as plt

class ColorDetector(Detector):
    # H: 0 a 179 (elección de color)
    # S: 0 a 255 (de blanco al color)
    # V: 0 a 255 (de negro a menos negro)
    def __init__(self):
        super().__init__()
        
        self.lower_blue_limit = [110, 200, 50]
        self.upper_blue_limit = [130, 255, 255]
        
        self.ideal_mask = np.ones((40, 80), dtype=np.float32)

    def show_ideal_mask(self):
        plt.imshow(self.ideal_mask, cmap='gray')
        plt.title('Máscara Ideal')
        plt.show()

    def filter(self, image):
        None