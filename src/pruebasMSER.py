import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector
import os

preprocessor = Preprocessor()
color_detector = ColorDetector()    
imagen_list = None
gray_images = None

def image_upload():
    imagen_list = preprocessor.load_all_images()
    gray_images = preprocessor.convert_to_gray(imagen_list)
    preprocessor.show_first_and_last(imagen_list)
    preprocessor.show_first_and_last(gray_images)

if __name__ == "__main__":    
    imagen_list = preprocessor.load_all_images()