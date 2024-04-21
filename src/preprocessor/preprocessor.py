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