import cv2
class Converter:
    def __init__(self):
        None
        
    def convert_to_gray(self, images):
        gray_images = []
        for image in images: 
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ_image = cv2.equalizeHist(gray_image)
            gray_images.append(equ_image)

        return gray_images
    
    def convert_bgr_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def convert_bgr_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def convert_hsv_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    def convert_hsv_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
