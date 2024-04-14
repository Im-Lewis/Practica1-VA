import cv2
import numpy as np


class MSERDetecor:
    def __init__(self):
            super().__init__()
    
    def mser(self, image_list, gray_list):
        list_images_regions = [] # Lista de imagenes con las regiones dibujadas
        list_of_masks = [] # Lista de listas de regiones detectadas en cada imagen 
        list_of_regions = [] # Lista de regiones de una imagen

        for act_image, img in enumerate(gray_list):  
            copy = image_list[act_image].copy()

            # Thresholding using the mean
            img_med = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21, 10)

            # Canny to get de borders
            filtered_image = cv2.Canny(img_med, 100, 200)

            # We dilate the image to make the edges thicker
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
            dilated_image = cv2.dilate(filtered_image, kernel)

            mser = cv2.MSER_create(delta=10, max_variation=0.1, min_area=1000, max_area=45000)
            polygons, _ = mser.detectRegions(dilated_image)
            list_of_regions.append(polygons)

            # Create a mask to store the detected regions
            mask = np.zeros((dilated_image.shape[0], img.shape[1]), dtype=np.uint8)
            list_of_masks.append(mask)

            hulls = []
            for p in polygons:
                hull = cv2.convexHull(p.reshape(-1, 1, 2))
                hulls.append(hull)
            
            # We draw the regions
            cv2.polylines(copy, hulls, 1, (0, 255, 0), 2)
            list_images_regions.append(copy)

        # Show the images 
        cv2.imshow('Detected Traffic Signs', list_images_regions[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Detected Traffic Signs', list_images_regions[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return list_images_regions, list_of_masks, list_of_regions
