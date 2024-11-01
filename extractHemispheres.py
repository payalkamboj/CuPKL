import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import detectRedActivationsIC
import greyWhiteMatterDetection
import CalculateActivationOverlapGWMatter

class GetHemispheres:
    """Detects right and left hemispheres"""
  
    
    def extractRightAndLeftHemisphere(self, cropped_image):
       # Convert the image to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment the brain
        _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour corresponds to the brain boundary
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Extract the region of interest (ROI) from the cropped image based on the bounding rectangle
        roi_brain_slice = cropped_image[y:y+h, x:x+w]

        # Split the brain image into left and right hemispheres using the bounding rectangle
        left_hemisphere = cropped_image[y:y+h, x:x+w//2, :]
        right_hemisphere = cropped_image[y:y+h, x+w//2:x+w, :]

        # # Display the results (you can modify this part based on your needs)
        # cv2.imshow('Original Image', cropped_image)
        # cv2.imshow('Left Hemisphere', left_hemisphere)
        # cv2.imshow('Right Hemisphere', right_hemisphere)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return left_hemisphere, right_hemisphere
    
    def process_hemisphere(self, hemisphere, min_contour_area, flip_status):
        # Detect red clusters in the hemisphere
        detectActivationsIC = detectRedActivationsIC.DetectRedActivations()
        big_red_clusters = detectActivationsIC.detect_red_clusters_hemishpere(hemisphere, min_contour_area, hemisphereType='a', printStatement = False)

        # Check if red clusters are present
        if big_red_clusters:
            # Flip the hemisphere horizontally (180 degrees)
            flipped_hemisphere = cv2.flip(hemisphere, 1)

            # Display the original and flipped images
            #cv2.imshow('Original Hemisphere', hemisphere)
            #cv2.imshow('Flipped Hemisphere', flipped_hemisphere)
            flip_status = True
            return flip_status, flipped_hemisphere
        else:
            # Display the original hemisphere if no red clusters are detected
            #cv2.imshow('Original Hemisphere', hemisphere)
            flip_status = False
            return flip_status, hemisphere
    
    def compare_clusters_in_hemispheres(self, left_hemisphere, right_hemisphere, min_contour_area=35):
       
        detectActivationsIC = detectRedActivationsIC.DetectRedActivations()
        left_clusters = detectActivationsIC.detect_red_clusters_hemishpere(left_hemisphere, min_contour_area, hemisphereType='left', printStatement = False)
        right_clusters = detectActivationsIC.detect_red_clusters_hemishpere(right_hemisphere, min_contour_area, hemisphereType='right', printStatement = False)
        
        similar_clusters = []
        non_similar_clusters_left = []
        non_similar_clusters_right = []
       # Check if both hemispheres have clusters
        if left_clusters and right_clusters:
            # Flip one of the hemispheres
            flipped_right_image = cv2.flip(right_hemisphere, 1)  # 1 indicates horizontal flip
            
            # Detect clusters in the flipped hemisphere
           
            flipped_right_clusters = detectActivationsIC.detect_red_clusters_hemishpere(flipped_right_image, min_contour_area,hemisphereType='right', printStatement = False)

            for left_idx, left_cluster in enumerate(left_clusters):
               
                                # Check overlap with flipped hemisphere
                                for flipped_right_idx, flipped_right_cluster in enumerate(flipped_right_clusters):
                                    overlap_score = cv2.matchShapes(left_cluster, flipped_right_cluster, cv2.CONTOURS_MATCH_I1, 0.0)
                                    left_activation_number = left_idx + 1
                                    right_activation_number = flipped_right_idx + 1
                                    if overlap_score < 0.2:  # Adjust the threshold based on your needs
                                        #print("Similar shapes of activations in both hemispheres")
                                        print(f"Left Hemisphere Activation/Cluster {left_activation_number} has similar shape with Right Hemisphere Activation/Cluster {right_activation_number}")
            
           
           
            # Get the centroids of the left clusters
            left_centroids = [np.mean(cluster, axis=0) for cluster in left_clusters]

            # Flip the right image
            flipped_right_image = cv2.flip(right_hemisphere, 1)  # 1 indicates horizontal flip

            # Detect clusters in the flipped right hemisphere
            flipped_right_clusters = detectActivationsIC.detect_red_clusters_hemishpere(flipped_right_image, min_contour_area, hemisphereType='right', printStatement=False)

            # Get the centroids of the flipped right clusters
            flipped_right_centroids = [np.mean(cluster, axis=0) for cluster in flipped_right_clusters]

            # Check if the centroids of left clusters are similar to centroids of flipped right clusters
            for left_idx, left_centroid in enumerate(left_centroids):
                matched_indices = []
                for right_idx, flipped_right_centroid in enumerate(flipped_right_centroids):
                    distance = np.linalg.norm(left_centroid - flipped_right_centroid)
                    if distance < 10:  # Adjust the threshold based on your needs
                         similar_clusters.append((left_idx, right_idx))
                         matched_indices.append(right_idx)
                         print(f"Symmetry in activations found at hemisphere level: Left hemisphere activation number {left_idx + 1} with Right hemisphere activation number {right_idx + 1}")
                
                if not matched_indices:
                    print(f"No symmetry found for Left Hemisphere Activation number {left_idx + 1}")
            unmatched_left_indices = set(range(len(left_centroids))) - set([pair[0] for pair in similar_clusters])

            greyWhiteMatterDetect = greyWhiteMatterDetection.GreyWhiteMatterDetection()
            CalculateActivationOverlapOnGreyWhiteMatter = CalculateActivationOverlapGWMatter.ActivationOverlapOnGreyWhiteMatter()
            for left_idx in unmatched_left_indices:
                print(f"No symmetry found for Left Hemisphere Activation number {left_idx + 1}")
                unmatched_left_activation = left_clusters[left_idx]
                flipped_left_hemisphere = cv2.flip(left_hemisphere, 1)
                grey_matter, white_matter = greyWhiteMatterDetect.segment_grey_white_matter(right_hemisphere)
                results_list= CalculateActivationOverlapOnGreyWhiteMatter.calculate_overlap_percentage(grey_matter, white_matter, flipped_left_hemisphere, moreThanOneActivation = 1)
                # Iterate through the results list
                for result in results_list:
                    cluster_index = result['cluster_index']
                    overlap_with_grey = result['percentage_overlap_with_grey']
                    overlap_with_white = result['percentage_overlap_with_white']
                    if cluster_index == left_idx + 1:
                        print(f"Percentage overlap of activation {cluster_index} with grey matter: {overlap_with_grey}%")
                        print(f"Percentage overlap of activation {cluster_index} with white matter: {overlap_with_white}%")

            unmatched_right_indices = set(range(len(flipped_right_centroids))) - set(matched_indices)
            for right_idx in unmatched_right_indices:
                print(f"No symmetry found for Right Hemisphere Activation number {right_idx + 1}")
                unmatched_right_activation = flipped_right_clusters[right_idx]
                flipped_right_hemisphere = cv2.flip(right_hemisphere, 1)
                grey_matter, white_matter = greyWhiteMatterDetect.segment_grey_white_matter(left_hemisphere)
                results_list = CalculateActivationOverlapOnGreyWhiteMatter.calculate_overlap_percentage(grey_matter, white_matter, flipped_right_hemisphere, moreThanOneActivation = 1)
                
                # Iterate through the results list
                for result in results_list:
                    cluster_index = result['cluster_index']
                    overlap_with_grey = result['percentage_overlap_with_grey']
                    overlap_with_white = result['percentage_overlap_with_white']

                    if cluster_index == right_idx + 1:
                        print(f"Percentage overlap of activation {cluster_index} with grey matter: {overlap_with_grey}%")
                        print(f"Percentage overlap of activation {cluster_index} with white matter: {overlap_with_white}%")


            # print(f"Number of clusters with symmetry: {len(similar_clusters)*2}")
            # print(f"Number of clusters without symmetry in left hemisphere: {len(unmatched_left_indices)}")
            # print(f"Number of clusters without symmetry in right hemisphere: {len(unmatched_right_indices)}")

        else:
            
             print("All clusters are in one hemisphere only! ")

        return similar_clusters