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

    def extract_border(self, gray_image):
        # Umbralizado adaptativo usando la media
        img_med = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21, 10)
        # Canny para obtener la imagen de bordes
        filtered_image = cv2.Canny(img_med, 100, 200)
        # Dilatamos los bordes de la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
        dilated_image = cv2.dilate(filtered_image, kernel)
        return dilated_image
    
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