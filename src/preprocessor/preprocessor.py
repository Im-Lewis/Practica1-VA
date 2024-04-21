from preprocessor.loader import Loader
from preprocessor.converter import Converter
from matplotlib import pyplot as plt
import cv2
import numpy as np

class Preprocessor(Loader, Converter):
    def __init__(self):
        None
        
    def cut_detected_zones(self, imagen, coords  = [(1300, 400, 270, 170)]):
        detected_subimages = []
        for coord in coords:
            im = self.cut_zone(imagen, coord)
            detected_subimages.append(im)
        return detected_subimages
    
    def cut_zone(self, image, coord):
        (x,y,w,h) = coord
        im = image[y:y+h, x:x+w]
        resized_image = cv2.resize(im, (80,40))
        return resized_image

    def extract_border(self, imagen):
        img = self.increase_saturation(imagen)
        img = self.apply_blue_filter([img])[0]
        return img
    
    def draw_a_rectangle_on_region(self, image, region):
        x, y, w, h = cv2.boundingRect(region)
        return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    def drawn_regions(self, polygons, image):
        hulls = []
        for p in polygons:
            hull = cv2.convexHull(p.reshape(-1, 1, 2))
            hulls.append(hull)
        
        # Dibujamos las regiones
        return cv2.polylines(image, hulls, 1, (0, 255, 0), 2)
    
    def increase_saturation(self, image, factor = 2):
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Aumentar la saturación
        hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 255).astype(np.uint8)

        # Convertir la imagen de vuelta a BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result
    
    def apply_blue_filter(self, imagenes):
        self.lower_blue_limit = np.array([105, 150, 20])
        self.upper_blue_limit = np.array([130, 255, 255])
        masks = []
        for image in imagenes:
            img = self.convert_bgr_to_hsv(image)
            mask = cv2.inRange(img, self.lower_blue_limit, self.upper_blue_limit)
            masks.append(mask)
            # TODO: Cuantos representar
            #self.show_blue_filter(image, mask)
            
        return masks