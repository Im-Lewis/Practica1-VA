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
        
        self.lower_blue_limit = np.array([105, 150, 20])
        self.upper_blue_limit = np.array([130, 255, 255])
        self.ideal_mask = self.create_ideal_mask()
        self.ideal_blues = np.sum(self.ideal_mask)
        
    def create_ideal_mask(self):
        ideal_mask_image = self.preprocessor.load_image('./','ideal_mask.png')
        ideal_mask_image = cv2.resize(ideal_mask_image, (80,40))
        return self.apply_blue_filter([ideal_mask_image])[0]
        
        
    def show_ideal_mask(self):
        plt.imshow(self.ideal_mask, cmap='gray')
        plt.title('Máscara Ideal')
        plt.show()

    def cut_detected_zones(self, imagenes, coords):
        return self.preprocessor.cut_detected_zones(imagenes, coords)
    
    def apply_blue_filter(self, imagenes):
        masks = []
        for image in imagenes:
            img = self.preprocessor.convert_bgr_to_hsv(image)
            
            mask = cv2.inRange(img, self.lower_blue_limit, self.upper_blue_limit)
            masks.append(mask/255)
            
            self.show_blue_filter(image, mask)
            
        return masks
    def show_blue_filter(self, original, filtered):
        plt.subplot(1,2,1)
        self.preprocessor.show(original, 'original')
        plt.subplot(1,2,2)
        self.preprocessor.show(filtered, 'filtered')
        plt.show()
        
    def correlation(self, filtered):
        correlation_image = filtered * self.ideal_mask
        correlation = ((np.sum(correlation_image)) % self.ideal_blues) / self.ideal_blues
        
        plt.title(f'Score:{correlation}')
        
        plt.subplot(3,1,1)
        plt.imshow(filtered)
        
        plt.subplot(3,1,2)
        plt.imshow(self.ideal_mask)

        plt.subplot(3,1,3)
        plt.imshow(correlation_image)
        
        plt.show()
    
        
