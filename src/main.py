import cv2
from matplotlib import pyplot as plt
from preprocessor.preprocessor import Preprocessor
from detector.color_detector import ColorDetector

dir = "imagenesTest"
preprocessor = Preprocessor()
color_detector = ColorDetector()    
imagen_list = None
gray_images = None

def image_upload():
    imagen_list = preprocessor.load_all_images(dir)
    gray_images = preprocessor.convert_to_gray(imagen_list)
    preprocessor.show_first_and_last(imagen_list)
    preprocessor.show_first_and_last(gray_images)
    
def detection_with_correlation_masks():
    color_detector.show_ideal_mask()
    #######################################################
    # vvvv Mocke vvvvv
    # TODO: CONSEGUIR COORDENADAS MSER 
    imagenes = [cv2.imread('./imagenesTest/00003.png')] 
    coords = [[(1300, 400, 270, 170)]]
    #######################################################
    detected_zones = color_detector.cut_detected_zones(imagenes, coords)                                         
    masks = color_detector.apply_blue_filter(detected_zones)
    color_detector.correlation(masks[0])
    

if __name__ == "__main__":    
    #image_upload()
    detection_with_correlation_masks()
