import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class processICForSlicing:
    """Crops the brain slices from IC"""
  
    
    def automate_slicing(self, main_image, template_image):

        # Read images
        main_image = cv2.imread(main_image)
        template_image = cv2.imread(template_image)
        # Convert images to grayscale if colored
        if main_image.shape[-1] == 3:
            main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        if template_image.shape[-1] == 3:
            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Normalized cross-correlation to find template matches
        correlation_result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)

        # Threshold for correlation values to be considered as matches
        threshold = 0.7

        # Coordinates of matches above the threshold
        loc = np.where(correlation_result > threshold)
        rows, cols = loc

        # Number of rows and columns in the IC
        unique_rows = np.unique(rows)
        unique_cols = np.unique(cols)

        num_row = len(unique_rows) - 1  # -1 because we don't need topmost row of IC
        num_col = len(unique_cols)

        start_y = rows[0] 
        start_x = cols[0] 

        # Calculate the width and height between Rs
        if len(unique_cols) >= 2:
            # Sort the unique_cols in ascending order
            sorted_cols = np.sort(unique_cols)

            # Calculate the width between the first two Rs
            size_x = sorted_cols[1] - sorted_cols[0]
        else:
            size_x = 0

        if len(unique_rows) >= 2:
            # Sort the unique_rows in ascending order
            sorted_rows = np.sort(unique_rows)

            # Calculate the height between two vertically aligned Rs
            size_y = sorted_rows[1] - sorted_rows[0]
        else:
            size_y = 0

        return start_x, start_y, size_x, size_y, num_row, num_col
    
    def imcrop(self, image, startX, startY, sizeX, sizeY):
        return image[startY:startY+sizeY, startX:startX+sizeX]
    

    def process_images(self, main_image_path, template_image_path, output_directory='DeepLearning/code/SOZDetectionUsingLLMs/testIC/croppedScans'):
        processor = processICForSlicing()

        start_x, start_y, size_x, size_y, num_row, num_col = self.automate_slicing(main_image_path, template_image_path)

        # Read the image again
        image = cv2.imread(main_image_path)

        shift_value = 13
        shift_value_y = 13

        # Initialize variables for looping
        k = 0
        s_value = start_x  # Store the initial value of start_x

        # Loop through rows
        for ii in range(1, num_row + 1):
            for jj in range(1, num_col + 1):
                # Create an empty list for imContour (not used in your provided code)
                imContour_k = []
               
                blank_threshold = 1400000 
                # Crop the image
                imCropped_k = self.imcrop(image, start_x + shift_value, start_y + shift_value_y, size_x, size_y)
                sum = np.sum(imCropped_k)
                #print(f"sum of Image {k} is {sum}")
                 # Check if the image is almost blank
                if np.sum(imCropped_k) < blank_threshold:
                    print(f"Image {k} is almost blank. Skipping...")
                else:
                    # Save the cropped image
                    output_path = os.path.join(output_directory, f'Cropped_Image_{k}.png')
                    cv2.imwrite(output_path, imCropped_k)

                    # Display the cropped image (you may remove this line if not needed)
                    #cv2.imshow(f'Cropped Image {k}', imCropped_k)
                    #cv2.waitKey(200)  # Add a delay for displaying the images

                # Update start_x for the next iteration
                start_x += size_x

                # Increment k for the next iteration
                k += 1

            # Update start_y and reset start_x for the next row
            start_y += size_y
            start_x = s_value

        cv2.destroyAllWindows()
