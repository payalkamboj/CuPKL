import cv2
import numpy as np

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def automate_slicing(main_image, template_image):

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


def imcrop(image, startX, startY, sizeX, sizeY):
    return image[startY:startY+sizeY, startX:startX+sizeX]

runAutomateSLicing = False

if runAutomateSLicing ==True:
    start_x, start_y, size_x, size_y, num_row, num_col = automate_slicing('IC_9_thresh.png', 'R.png')
    print(start_x, start_y, size_x, size_y, num_row, num_col)
    # Read the image again
    image_path = 'IC_2_thresh.png'
    image = cv2.imread(image_path)

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

            # Crop the image
            imCropped_k = imcrop(image, start_x+shift_value, start_y+shift_value_y, size_x, size_y)
            output_directory = '.'
            # Save the cropped image
            output_path = os.path.join(output_directory, f'Cropped_Image_{k}.png')
            cv2.imwrite(output_path, imCropped_k)

            # Display the cropped image (you may remove this line if not needed)
            cv2.imshow(f'Cropped Image {k}', imCropped_k)
            cv2.waitKey(200)  # Add a delay for displaying the images

        
            # Update start_x for the next iteration
            start_x += size_x

            # Increment k for the next iteration
            k += 1

        # Update start_y and reset start_x for the next row
        start_y += size_y
        start_x = s_value

    cv2.destroyAllWindows()

def detect_red_clusters(image,min_contour_area):
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
     # Check if each red cluster is in the middle of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (width/2, height/2)

    for cluster in red_clusters:
        # Calculate the centroid of the cluster
        M = cv2.moments(cluster)
        if M["m00"] != 0:
            cluster_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Calculate the distance between the cluster center and the image center
            distance_to_center = np.sqrt((cluster_center[0] - image_center[0]) ** 2 + (cluster_center[1] - image_center[1]) ** 2)

            # You can adjust the threshold for considering a cluster to be in the middle
            threshold_distance = 30  # Adjust as needed

            if distance_to_center < 20:
                print("Red cluster is in the middle!")
            else:
                print("Red cluster is not in center")
                # Do something with the cluster in the middle, if needed
    return red_clusters

def calculate_overlap_percentage(red_cluster, white_matter, grey_matter):
  # Create masks for each contour
  # Initialize counters for overlap with grey matter and white matter
    overlap_with_grey = 0
    overlap_with_white = 0

    # Iterate over each red cluster
    for red_cluster in red_clusters:
        # Create a mask for the current red cluster
        red_cluster_mask = np.zeros_like(grey_matter)
        cv2.drawContours(red_cluster_mask, [red_cluster], -1, 255, thickness=cv2.FILLED)

        # Calculate the overlap with grey matter
        overlap_with_grey += np.sum(np.logical_and(red_cluster_mask, grey_matter))

        # Calculate the overlap with white matter
        overlap_with_white += np.sum(np.logical_and(red_cluster_mask, white_matter))

    # Calculate the total area of the red clusters
    total_red_area = sum(cv2.contourArea(red_cluster) for red_cluster in red_clusters)

    # Calculate the overlap percentages
    overlap_percentage_grey = (overlap_with_grey / total_red_area) * 100
    overlap_percentage_white = (overlap_with_white / total_red_area) * 100


    return overlap_percentage_grey, overlap_percentage_white

def segment_grey_white_matter(image):
   # Convert the image to grayscale
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

   # Adaptive thresholding to segment grey matter
   grey_matter = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
   
   # Invert the grey matter to get white matter
   white_matter = cv2.bitwise_not(grey_matter)
   

   cv2.imshow('Grey Matter', grey_matter)
   cv2.imshow('white_matter', white_matter)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   return grey_matter, white_matter


# Load your cropped image
cropped_image_path = 'Cropped_Image_19.png'
cropped_image = cv2.imread(cropped_image_path)

# Segmentation of grey and white matter
grey_matter, white_matter = segment_grey_white_matter(cropped_image)

# Detect red clusters
red_clusters = detect_red_clusters(cropped_image,100)


if len(red_clusters) == 0:
    print("No red clusters detected.")
else:
    for i, red_cluster in enumerate(red_clusters):
        white_overlap_percentage, grey_overlap_percentage = calculate_overlap_percentage(
            red_cluster, white_matter, grey_matter)

        print(f'Red Cluster {i + 1} Overlap with White Matter: {white_overlap_percentage:.2f}%')
        print(f'Red Cluster {i + 1} Overlap with Grey Matter: {grey_overlap_percentage:.2f}%')

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

# Split the brain image into left and right hemispheres using the bounding rectangle
left_hemisphere = cropped_image[y:y+h, x:x+w//2, :]
right_hemisphere = cropped_image[y:y+h, x+w//2:x+w, :]

# # Display the results (you can modify this part based on your needs)
cv2.imshow('Original Image', cropped_image)
cv2.imshow('Left Hemisphere', left_hemisphere)
cv2.imshow('Right Hemisphere', right_hemisphere)
cv2.waitKey(0)
cv2.destroyAllWindows()

def check_symmetry(contour1, contour2):
    # Compare the shapes using cv2.matchShapes
    shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)

    # You can set a threshold for similarity and decide if the shapes are symmetrical
    threshold = 0.1  # Adjust this threshold based on your requirements

    return shape_similarity < threshold

# Assuming you have the left_hemisphere and right_hemisphere from the previous code

# Detect red clusters in the left and right hemispheres
left_red_clusters = detect_red_clusters(left_hemisphere,40)
right_red_clusters = detect_red_clusters(right_hemisphere,40)

if len(red_clusters) == 0:
    print("No red clusters detected, so no symmetry to check")
elif len(left_red_clusters) == 0 and len(right_red_clusters) == 0:
    print("No red clusters detected in either hemisphere, so no symmetry to check")
elif len(left_red_clusters) == 0 and len(right_red_clusters) !=0:
    print("No red clusters detected in left hemisphere, there is cluster in right, no symmetry to check")
elif len(right_red_clusters) == 0 and len(left_red_clusters) !=0:
    print("No red_clusters detected in right hemisphere, there is cluster in left, no symmetry to check")
elif len(right_red_clusters) == 0:
     print("No right_red_clusters detected, cluster is detected in left hemisphere, so no symmetry to check")
else:
    # Assuming the largest contours correspond to the red clusters
    left_largest_contour = max(left_red_clusters, key=cv2.contourArea)
    right_largest_contour = max(right_red_clusters, key=cv2.contourArea)

    # Check if the shapes are symmetrical
    is_symmetrical = check_symmetry(left_largest_contour, right_largest_contour)

    # Print the result
    if is_symmetrical:
        print("The red clusters have symmetrical shapes.")
    else:
        print("The red clusters do not have symmetrical shapes.")

    # Visualize the largest contours on the left and right hemispheres
    left_visualization = left_hemisphere.copy()
    right_visualization = right_hemisphere.copy()

    cv2.drawContours(left_visualization, [left_largest_contour], 0, (0, 255, 0), 2)
    cv2.drawContours(right_visualization, [right_largest_contour], 0, (0, 255, 0), 2)

    # Resize the images to the same height for concatenation
    common_height = 400  # Choose a common height for all images

    cropped_image_resized = cv2.resize(cropped_image, (int(cropped_image.shape[1] * common_height / cropped_image.shape[0]), common_height))
    left_visualization_resized = cv2.resize(left_visualization, (int(left_visualization.shape[1] * common_height / left_visualization.shape[0]), common_height))
    right_visualization_resized = cv2.resize(right_visualization, (int(right_visualization.shape[1] * common_height / right_visualization.shape[0]), common_height))

    # Create a blank image with three sections and some space in between
    combined_image_width = cropped_image_resized.shape[1] + 20  # 20 is the space between images
    combined_image = np.zeros((common_height, 3 * combined_image_width, 3), dtype=np.uint8)

    # Copy the resized images to their respective sections with space
    combined_image[:, :cropped_image_resized.shape[1], :] = cropped_image_resized
    combined_image[:, combined_image_width:combined_image_width + left_visualization_resized.shape[1], :] = left_visualization_resized
    combined_image[:, 2 * combined_image_width + 20:2 * combined_image_width + 20 + right_visualization_resized.shape[1], :] = right_visualization_resized

    # Display the result
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()