import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class DetectRedActivations:
    """Crops the brain slices from IC"""
  
    
    def detect_red_clusters(self, image,min_contour_area, print_statement=True):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a binary mask for red regions
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (adjust the area threshold based on your needs)
        #min_contour_area = 100
        red_clusters = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        medium_red_clusters = [cnt for cnt in contours if 35 < cv2.contourArea(cnt) < 120]
        small_red_clusters = [cnt for cnt in contours if 5 < cv2.contourArea(cnt) < 35]
        smallClustersNum = len(small_red_clusters)
        clustersNum = len(red_clusters)
        mediumSizeClusterSize = len(medium_red_clusters)
       
        middle = 0
        
        if print_statement:
            print(f'{clustersNum} big red activation(s) detected!')

           
            if mediumSizeClusterSize!=0:
                 print(f'{mediumSizeClusterSize} medium red activation(s) detected!')

            if smallClustersNum!=0:
                 print(f'{smallClustersNum} small red activation(s) detected!')
                 
            # Find the largest contour in the brain slice
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

        #    # Calculate and print the areas
        #     total_red_cluster_area = sum(cv2.contourArea(cnt) for cnt in red_clusters)
        #     area_inside_largest_contour = cv2.contourArea(largest_contour)

        #     #print(f'Total red cluster area: {total_red_cluster_area}')
        #     #print(f'Area inside the largest contour: {area_inside_largest_contour}')

        #     # Check if the total area of red clusters is close enough to the area inside the largest contour
        #     if total_red_cluster_area > 1/4 * area_inside_largest_contour:  # Adjust the threshold as needed
        #         print('The total area of red clusters is close to the area inside the largest contour.')
        #     else:
        #         print('The total area of red clusters is not close to the area inside the largest contour.')

            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            brain_slice_center = (x + w // 2, y + h // 2)

            for clusternum, cluster in enumerate(red_clusters):
            # Calculate the centroid of the cluster
                M = cv2.moments(cluster)
                if M["m00"] != 0:
                    cluster_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # Calculate the distance between the cluster center and the brain slice center
                    distance_to_center = np.sqrt((cluster_center[0] - brain_slice_center[0]) ** 2 +
                                                (cluster_center[1] - brain_slice_center[1]) ** 2)

                    # You can adjust the threshold for considering a cluster to be in the middle
                    threshold_distance = 20  # Adjust as needed

                    if distance_to_center < threshold_distance:
                        print(f"Big red activation number {clusternum+1} is in the middle of the brain!")
                        middle = 1
                    else:
                        print(f"Big red activation number {clusternum+1} is not in the middle of the brain slice!")
                        # Do something with the cluster, if needed
                        middle = 0
            for clusternum, cluster in enumerate(medium_red_clusters):
            # Calculate the centroid of the cluster
                M = cv2.moments(cluster)
                if M["m00"] != 0:
                    cluster_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # Calculate the distance between the cluster center and the brain slice center
                    distance_to_center = np.sqrt((cluster_center[0] - brain_slice_center[0]) ** 2 +
                                                (cluster_center[1] - brain_slice_center[1]) ** 2)

                    # You can adjust the threshold for considering a cluster to be in the middle
                    threshold_distance = 20  # Adjust as needed
       
                    if  0 <= clustersNum > 1: #if big red cluster is present then do not print medium middle information!
                        if distance_to_center < threshold_distance:
                            print(f"Medium size red activation number {clusternum+1} is in the middle of the brain!")
                        
                        else:
                            print(f"Medium size red activation number {clusternum+1} is not in the middle of the brain!")
                            # Do something with the cluster, if needed
                        
        return red_clusters, medium_red_clusters, middle
    
    def check_symmetry_single_big_activation(self, contour1, contour2):
        # Compare the shapes using cv2.matchShapes
        shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)

        # You can set a threshold for similarity and decide if the shapes are symmetrical
        threshold = 0.1  

        return shape_similarity < threshold

    def detect_red_clusters_hemishpere(self, image,min_contour_area, hemisphereType, printStatement = True):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a binary mask for red regions
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (adjust the area threshold based on your needs)
        #min_contour_area = 100
        red_clusters = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        clustersNum = len(red_clusters)
        # if printStatement == True:
        #     print(f'{clustersNum} activation(s) detected in {hemisphereType} hemisphere!')

        return red_clusters