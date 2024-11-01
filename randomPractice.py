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
    # Find the largest contour in the brain slice
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    brain_slice_center = (x + w // 2, y + h // 2)

    for cluster in red_clusters:
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
                print("Red cluster is in the middle of the brain slice!")
            else:
                print("Red cluster is not in the middle of the brain slice")
                # Do something with the cluster, if needed
    return red_clusters
def segment_grey_white_matter(image):
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

    cv2.imshow('white_matter Image', white_matter)
    cv2.imshow('Grey_matter Image', grey_matter)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grey_matter, white_matter

def calculate_overlap_percentage(left_grey_matter, left_white_matter, right_hemisphere_red_clusters):
    
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
    cv2.imshow('Left grey matter', left_grey_matter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours in the mask
    contours, _ = cv2.findContours(right_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area of red clusters in the right hemisphere
    right_red_cluster_area = sum(cv2.contourArea(cnt) for cnt in contours)

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
   
    print(f"Percentage overlap with grey matter: {percentage_overlap_with_grey}%")
    print(f"Percentage overlap with white matter: {percentage_overlap_with_white}%")
     # Visualize the overlapping areas
   
    return percentage_overlap_with_grey, percentage_overlap_with_white

# Load your cropped image
cropped_image_path = 'Cropped_Image_19.png'
cropped_image = cv2.imread(cropped_image_path)

# Segmentation of grey and white matter
grey_matter, white_matter = segment_grey_white_matter(cropped_image)

# Detect red clusters
red_clusters = detect_red_clusters(cropped_image,100)


# if len(red_clusters) == 0:
#     print("No red clusters detected.")
# else:
#     for i, red_cluster in enumerate(red_clusters):
#         white_overlap_percentage, grey_overlap_percentage = calculate_overlap_percentage(
#             red_cluster, white_matter, grey_matter)

#         print(f'Red Cluster {i + 1} Overlap with White Matter: {white_overlap_percentage:.2f}%')
#         print(f'Red Cluster {i + 1} Overlap with Grey Matter: {grey_overlap_percentage:.2f}%')

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

#min_contour_area = 100  # Adjust as needed


#red_clusters = detect_red_clusters(cropped_image, min_contour_area)

# If you want to visualize the ROI, you can draw the bounding rectangle on the cropped_image
#cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the ROI
#cv2.imshow('Image with ROI', cropped_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Split the brain image into left and right hemispheres using the bounding rectangle
left_hemisphere = cropped_image[y:y+h, x:x+w//2, :]
right_hemisphere = cropped_image[y:y+h, x+w//2:x+w, :]

# # Display the results (you can modify this part based on your needs)
cv2.imshow('Original Image', cropped_image)
cv2.imshow('Left Hemisphere', left_hemisphere)
cv2.imshow('Right Hemisphere', right_hemisphere)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect red clusters in the left and right hemispheres
left_red_clusters = detect_red_clusters(left_hemisphere,40)
right_red_clusters = detect_red_clusters(right_hemisphere,40)

def check_symmetry(contour1, contour2):
    # Compare the shapes using cv2.matchShapes
    shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)

    # You can set a threshold for similarity and decide if the shapes are symmetrical
    threshold = 0.1  

    return shape_similarity < threshold


def process_hemisphere(hemisphere, min_contour_area, flip_status):
    # Detect red clusters in the hemisphere
    red_clusters = detect_red_clusters(hemisphere, min_contour_area)

    # Check if red clusters are present
    if red_clusters:
        # Flip the hemisphere horizontally (180 degrees)
        flipped_hemisphere = cv2.flip(hemisphere, 1)

        # Display the original and flipped images
        cv2.imshow('Original Hemisphere', hemisphere)
        cv2.imshow('Flipped Hemisphere', flipped_hemisphere)
        flip_status = True
        return flip_status, flipped_hemisphere
    else:
        # Display the original hemisphere if no red clusters are detected
        cv2.imshow('Original Hemisphere', hemisphere)
        flip_status = False
        return flip_status, hemisphere

flip_status = False
# Detect and process red clusters in left and right hemispheres
flip_status_left_hemisphere, left_hemisphere = process_hemisphere(left_hemisphere, 40, flip_status)
flip_status_right_hemishpere, right_hemisphere = process_hemisphere(right_hemisphere, 40, flip_status)



if len(red_clusters) == 1:
    if flip_status_left_hemisphere == False:
        grey_matter, white_matter = segment_grey_white_matter(left_hemisphere)
        calculate_overlap_percentage(grey_matter, white_matter, right_hemisphere)

    if flip_status_right_hemishpere == False:
        grey_matter, white_matter = segment_grey_white_matter(right_hemisphere)
        calculate_overlap_percentage(grey_matter, white_matter, left_hemisphere)

cv2.waitKey(10)
cv2.destroyAllWindows()

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

#     # Visualize the largest contours on the left and right hemispheres
#     left_visualization = left_hemisphere.copy()
#     right_visualization = right_hemisphere.copy()

#     cv2.drawContours(left_visualization, [left_largest_contour], 0, (0, 255, 0), 2)
#     cv2.drawContours(right_visualization, [right_largest_contour], 0, (0, 255, 0), 2)

#     # Resize the images to the same height for concatenation
#     common_height = 400  # Choose a common height for all images

#     cropped_image_resized = cv2.resize(cropped_image, (int(cropped_image.shape[1] * common_height / cropped_image.shape[0]), common_height))
#     left_visualization_resized = cv2.resize(left_visualization, (int(left_visualization.shape[1] * common_height / left_visualization.shape[0]), common_height))
#     right_visualization_resized = cv2.resize(right_visualization, (int(right_visualization.shape[1] * common_height / right_visualization.shape[0]), common_height))

#     # Create a blank image with three sections and some space in between
#     combined_image_width = cropped_image_resized.shape[1] + 20  # 20 is the space between images
#     combined_image = np.zeros((common_height, 3 * combined_image_width, 3), dtype=np.uint8)

#     # Copy the resized images to their respective sections with space
#     combined_image[:, :cropped_image_resized.shape[1], :] = cropped_image_resized
#     combined_image[:, combined_image_width:combined_image_width + left_visualization_resized.shape[1], :] = left_visualization_resized
#     combined_image[:, 2 * combined_image_width + 20:2 * combined_image_width + 20 + right_visualization_resized.shape[1], :] = right_visualization_resized

#     # Display the result
#     cv2.imshow('Combined Image', combined_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()