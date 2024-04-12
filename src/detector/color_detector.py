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
        
        self.mask = np.ones((40, 80), dtype=np.float32)

    def show_ideal_mask(self):
        plt.imshow(self.mask, cmap='gray')
        plt.title('Máscara Ideal')
        plt.show()

    # TODO: Cut zones looking the coord(x, y) 
    def cut_detected_zones(self, image):
        detected_subimages = []
        None

    # TODO: Get the resulted mask
    def apply_blue_filter(self, image):
        None
