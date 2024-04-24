import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector
from detector.mser_detector import MSERDetector
from detector.nms import NMS
import shutil
import os

import numpy as np

dir = "imagenesTest"
preprocessor = Preprocessor()
color_detector = ColorDetector()    
mser_detector = MSERDetector()
nms_detector = NMS()
imagen_list = None
gray_images = None

def image_upload():
    global imagen_list
    global gray_images
    
    imagen_list = preprocessor.load_all_images(dir)
    gray_images = preprocessor.convert_to_gray(imagen_list)




if __name__ == "__main__":    

    input_folder = 'imagenesTest'
    output_folder = 'imagenesColor'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 1)
            
            if img is not None:
                
                img = increase_saturation(img)
                img = color_detector.apply_blue_filter([img])[0]
                
                output_img_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_img_path, img)
            else:
                print(f"Error al cargar la imagen {filename}")
        else:
            print(f"Omitido archivo no imagen: {filename}")


