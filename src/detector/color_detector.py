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

    def cut_detected_zones(self, imagenes, all_coords):
        sub_imagenes = []
        for imagen, coords in zip(imagenes, all_coords):
            sub_image = self.preprocessor.cut_detected_zones(imagen, coords)
            sub_imagenes.append(sub_image)
        return sub_imagenes
    
    
    def apply_all_blue_filter(self, detected_zones_list):
        masks_per_image = []
        for subpanels in detected_zones_list:
            masks = self.apply_blue_filter(subpanels)
            masks_per_image.append(masks)
        return masks_per_image
        
    
    def apply_blue_filter(self, imagenes):
        masks = []
        for image in imagenes:
            img = self.preprocessor.convert_bgr_to_hsv(image)
            mask = cv2.inRange(img, self.lower_blue_limit, self.upper_blue_limit)
            masks.append(mask/255)
            # TODO: Cuantos representar
            #self.show_blue_filter(image, mask)
            
        return masks
    # TODO
    def show_blue_filter(self, original, filtered):
        plt.subplot(1,2,1)
        self.preprocessor.show(original, 'original')
        plt.subplot(1,2,2)
        self.preprocessor.show(filtered, 'filtered')
        plt.show()
    
    def all_correlation(self, masks_list:list[list[np.array]], coords_list:list[list[tuple]]):
        scoreds_and_coords_list = []
        for masks, coords in zip(masks_list, coords_list):
            scoreds_coords = self.correlation(masks, coords)
            scoreds_and_coords_list.append(scoreds_coords)
        return scoreds_and_coords_list
        
    def correlation(self, masks, coords):
        scoreds_masks = []
        for mask, coord in zip(masks, coords):
            correlation_image = mask * self.ideal_mask
            correlation = ((np.sum(correlation_image)) % self.ideal_blues) / self.ideal_blues
            if correlation > 0.7:
                scored_mask = (coord, correlation)
                scoreds_masks.append(scored_mask)
        return scoreds_masks
        # TODO
            # plt.title(f'Score:{correlation}')
            
            # plt.subplot(3,1,1)
            # plt.imshow(mask)
            
            # plt.subplot(3,1,2)
            # plt.imshow(self.ideal_mask)

            # plt.subplot(3,1,3)
            # plt.imshow(correlation_image)
            
            # plt.show()
    
        
