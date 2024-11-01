import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class GreyWhiteMatterDetection:
    """Detects grey and white matter"""
  
    
    def segment_grey_white_matter(self, image):
    # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get the brightest regions as white matter
        _, white_matter_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

        # Convert data type to CV_8U
        image = image.astype(np.uint8)

        # Invert the binary mask to get grey matter mask
        grey_matter_mask = cv2.bitwise_not(white_matter_mask)

        # Apply the mask to the original image to extract white matter
        white_matter = cv2.bitwise_and(image, image, mask=white_matter_mask)
        grey_matter = cv2.bitwise_and(image, image, mask=grey_matter_mask)

        # cv2.imshow('white_matter Image', white_matter)
        # cv2.imshow('Grey_matter Image', grey_matter)
    
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return grey_matter, white_matter