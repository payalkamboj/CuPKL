import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import detectRedActivationsIC

class ActivationOverlapOnGreyWhiteMatter:
    """Detects red activation overlap with grey and white matter"""
  
    
    def calculate_overlap_percentage(self, left_grey_matter, left_white_matter, right_hemisphere_red_clusters,moreThanOneActivation = 0):
    
        # Resize the right hemisphere to match the dimensions of the left hemisphere
        right_hemisphere_resized = cv2.resize(right_hemisphere_red_clusters, (left_grey_matter.shape[1], left_grey_matter.shape[0]))

        # Convert the right hemisphere to HSV color space
        right_hsv = cv2.cvtColor(right_hemisphere_resized, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for red hues in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a binary mask for red regions
        right_red_mask = cv2.inRange(right_hsv, lower_red, upper_red)

        cv2.imshow('Thresholded Red Clusters', right_red_mask)
        cv2.waitKey(1000)
        # cv2.imshow('Left grey matter', left_grey_matter)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Find contours in the mask
        contours, _ = cv2.findContours(right_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if moreThanOneActivation == 0:
            # Calculate the area of red clusters in the right hemisphere
            right_red_cluster_area = sum(cv2.contourArea(cnt) for cnt in contours)
            if right_red_cluster_area != 0:
                #print("Red activation area is=", right_red_cluster_area)

                # Calculate the overlap with grey matter in the left hemisphere
                overlap_with_grey = cv2.bitwise_and(left_grey_matter, right_red_mask)
                overlap_grey_area = cv2.countNonZero(overlap_with_grey)

                # Calculate the overlap with white matter in the left hemisphere
                overlap_with_white = cv2.bitwise_and(left_white_matter, right_red_mask)
                overlap_white_area = cv2.countNonZero(overlap_with_white)

                # Calculate the percentage overlap
                total_left_grey_area = cv2.countNonZero(left_grey_matter)
                total_left_white_area = cv2.countNonZero(left_white_matter)

                # Calculate the percentage overlap with respect to the entire area of red clusters
                percentage_overlap_with_grey = (overlap_grey_area / right_red_cluster_area) * 100
                percentage_overlap_with_white = (overlap_white_area / right_red_cluster_area) * 100
                            
                # # Check for majority overlap
                # if percentage_overlap_with_grey > percentage_overlap_with_white + 2*percentage_overlap_with_white:
                #     print("Majority overlap on grey matter")
                # elif percentage_overlap_with_white > percentage_overlap_with_grey + 2*percentage_overlap_with_grey:
                #     print("Majority overlap on white matter")
                # else:
                #     print("Overlap is almost the same on gray and white matter")
                print("Big red activation has overlap on:")
                print(f"Percentage overlap with grey matter: {percentage_overlap_with_grey}%")
                print(f"Percentage overlap with white matter: {percentage_overlap_with_white}%")
                # Visualize the overlapping areas
            
                return percentage_overlap_with_grey, percentage_overlap_with_white
        
        else:
            
            # Filter out small contours (adjust the area threshold based on your needs)
            red_clusters = [cnt for cnt in contours if cv2.contourArea(cnt) > 40]
            results_list = []
            # Calculate the area of red clusters in the hemisphere and overlap for each cluster
            for cluster_idx, cluster in enumerate(red_clusters):
                # Calculate the area of the current red cluster
                cluster_area = cv2.contourArea(cluster)

                # Calculate the overlap with grey matter in the left hemisphere
                overlap_with_grey = cv2.bitwise_and(left_grey_matter, right_red_mask)
                overlap_grey_area = cv2.countNonZero(overlap_with_grey)

                # Calculate the overlap with white matter in the left hemisphere
                overlap_with_white = cv2.bitwise_and(left_white_matter, right_red_mask)
                overlap_white_area = cv2.countNonZero(overlap_with_white)

                # Calculate the percentage overlap with respect to the area of the current red cluster
                percentage_overlap_with_grey = (overlap_grey_area / cluster_area) * 100
                percentage_overlap_with_white = (overlap_white_area / cluster_area) * 100

                # Append the results to the list
                results_list.append({
                    'cluster_index': cluster_idx + 1,
                    'percentage_overlap_with_grey': percentage_overlap_with_grey,
                    'percentage_overlap_with_white': percentage_overlap_with_white
                })
                # print(f"Results for Activation number {cluster_idx + 1}:")

                # print(f"Percentage overlap with grey matter: {percentage_overlap_with_grey}%")
                # print(f"Percentage overlap with white matter: {percentage_overlap_with_white}%")

                # Visualize the overlapping areas

            return results_list

