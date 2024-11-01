import processICForLLM
import detectRedActivationsIC
import greyWhiteMatterDetection
import extractHemispheres
import cv2
import CalculateActivationOverlapGWMatter
import GeminiLLM 
import ChatGPT_LLM
import sys
from io import StringIO
import os
import re
import pandas as pd
import glob
import time
# Create an empty list to store the information


if __name__ == '__main__':
    
    #to sort files names numerically
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    from pathlib import Path
    # Redirect standard output to a variable
    subjects = 'fmridatasets/' + 'ASUAI_0*'
    
    counter = 1
    #counterI = "All"
    for subject in sorted(glob.glob(subjects),key=numericalSort):
        data = []
        if (counter>82): # to start from nth patients 
            print("Inside the subject loop!")
            # Specify the directory
            filename = subject
            print("Filename is ", filename)
            excel_filename_result = filename.split("/")[-1]
            print("Filename after split ", excel_filename_result)
            #directory = 'testIC/report/'
            directory = filename + '/MO/report/'

            # Get a list of all files in the directory
            files = os.listdir(directory)

            # Filter files ending with 'thresh.png'
            image_files = [file for file in files if file.endswith('thresh.png')]
            
            # Sort the image files numerically
            image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

            # Process each image
            for image_file in image_files:
                # Construct the full path to the image
                print("Processing file:",image_file)
                image_path = os.path.join(directory, image_file)


                ProcessICSlicing = processICForLLM.processICForSlicing()
                ProcessICSlicing.process_images(image_path, 'DeepLearning/code/SOZDetectionUsingLLMs/R.png')
            
                # Specify the directory
                directoryCroppedSlices = 'DeepLearning/code/SOZDetectionUsingLLMs/testIC/croppedScans/'

                # Get a list of all files in the directory
                filesCroppedScans = os.listdir(directoryCroppedSlices)

                # Filter files ending with 'thresh.png'
                cropped_image_files = [file for file in filesCroppedScans if file.endswith('.png')]
                
                # Process each image
                for cropped_image in cropped_image_files:
                #  Load your cropped image
                #cropped_image_path = 'testIC/croppedScans/cropped_Image_20.png'
                    #print("Processing cropped slices", cropped_image)
                    cropped_image_name = cropped_image
                    print("croppedimage name", cropped_image_name)
                    cropped_image_path = os.path.join(directoryCroppedSlices, cropped_image)
                    # Add a delay of, say, 5 seconds before processing the image
                    #time.sleep(20)
                    cropped_image = cv2.imread(cropped_image_path)
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                
                    detectActivationsIC = detectRedActivationsIC.DetectRedActivations()
                    # Detect red clusters
                    big_red_clusters, medium_red_clusters, is_Big_Middle = detectActivationsIC.detect_red_clusters(cropped_image,120)
                    greyWhiteMatterDetect = greyWhiteMatterDetection.GreyWhiteMatterDetection()
                    grey_matter, white_matter = greyWhiteMatterDetect.segment_grey_white_matter(cropped_image)
                    CalculateActivationOverlapOnGreyWhiteMatter = CalculateActivationOverlapGWMatter.ActivationOverlapOnGreyWhiteMatter()

                
                    HemisphereExtraction = extractHemispheres.GetHemispheres()
                    left_hemisphere, right_hemisphere = HemisphereExtraction.extractRightAndLeftHemisphere(cropped_image)

                    #print("Detecting activations in both hemispheres....")
                    left_red_clusters = detectActivationsIC.detect_red_clusters_hemishpere(left_hemisphere,35, hemisphereType='left', printStatement = True)
                    right_red_clusters = detectActivationsIC.detect_red_clusters_hemishpere(right_hemisphere,35, hemisphereType='right', printStatement = True)

                    if is_Big_Middle == 0:
                        flip_status_left_hemisphere, left_hemisphere = HemisphereExtraction.process_hemisphere(left_hemisphere, 35, flip_status=False)
                        flip_status_right_hemishpere, right_hemisphere = HemisphereExtraction.process_hemisphere(right_hemisphere, 35, flip_status=False)
                        #print("Activation found in either hemisphere", flip_status_left_hemisphere,flip_status_right_hemishpere )
                    
                        if len(big_red_clusters) == 1 or len(medium_red_clusters)==1:
                            if flip_status_left_hemisphere == False:
                                grey_matter, white_matter = greyWhiteMatterDetect.segment_grey_white_matter(left_hemisphere)
                                CalculateActivationOverlapOnGreyWhiteMatter.calculate_overlap_percentage(grey_matter, white_matter, right_hemisphere, moreThanOneActivation = 0)

                            if flip_status_right_hemishpere == False:
                                grey_matter, white_matter = greyWhiteMatterDetect.segment_grey_white_matter(right_hemisphere)
                                CalculateActivationOverlapOnGreyWhiteMatter.calculate_overlap_percentage(grey_matter, white_matter, left_hemisphere,moreThanOneActivation = 0)
                            

                    if len(big_red_clusters) == 0 and len(medium_red_clusters) == 0:
                        print("No big or medium red activations detected")
                    elif len(left_red_clusters) == 0 and len(right_red_clusters) == 0:
                        print("No significant red activations detected in either hemisphere, so no symmetry to check")
                    elif len(left_red_clusters) == 0 and len(right_red_clusters) !=0:
                        print("No significant red activations detected in left hemisphere, there is activation in right hemisphere, no symmetry!")
                    elif len(right_red_clusters) == 0 and len(left_red_clusters) !=0:
                        print("No significant activations detected in right hemisphere, there is activation in left, no symmetry!")
                    elif len(big_red_clusters) == 1 and len(medium_red_clusters)<2: #big activation in both hemispheres
                        #print("Big red activation has overlap on:")
                        CalculateActivationOverlapOnGreyWhiteMatter.calculate_overlap_percentage(grey_matter, white_matter, cropped_image, moreThanOneActivation = 0)
                        # Assuming the largest contours correspond to the red clusters
                        left_largest_contour = max(left_red_clusters, key=cv2.contourArea)
                        right_largest_contour = max(right_red_clusters, key=cv2.contourArea)

                        # Check if the shapes are symmetrical
                        is_symmetrical = detectActivationsIC.check_symmetry_single_big_activation(left_largest_contour, right_largest_contour)

                        # Print the result
                        if is_symmetrical:
                            print("The big red activation has symmetrical shapes.")
                        else:
                            print("The big red activation has asymmetrical shapes.")
                    
                    elif 2 <= len(big_red_clusters) <= 3 or 2 <= len(medium_red_clusters) <= 3:
                        HemisphereExtraction.compare_clusters_in_hemispheres(left_hemisphere,right_hemisphere, min_contour_area=35)
                    
                    elif  len(big_red_clusters) > 3 or len(medium_red_clusters) > 3:
                        print("Too many big and medium activations!")

                    


                        
                    #model = genai.GenerativeModel(model_name='gemini-pro')
                    prompt = """Rules of seizure onset zone (SOZ) are as follows:
                    
                    1. Detection of 1 big red activation in the middle.
                    2. Big red activation has NO similar shape.
                    3. Big red activation has no symmetry detected in either left-right hemisphere.


                    

                    Rules of being not SOZ:
                1. Detection of 1 big red activation is not in the middle of the brain and majority overlap in gray matter.
                2. Presence of few big or medium sized red activations in gray matter.
                3. Symmetry in activations in both right and left hemispheres.
                4. Lots of small activations.

                Based on these rules, just answer in YES/NO if the below promts belong to SOZ or not. If unable to decide then say NO.

                """
                
            
                    # Get the value of the redirected output
                    IC_output = sys.stdout.getvalue()
                    #print("________________")
                    # Reset standard output to its original value
                    sys.stdout = old_stdout
                    prompt =  prompt + IC_output
                    #print ("Prompt: ", prompt)
                    #print("________________")
                    llmResponse = ChatGPT_LLM.get_LLM_answer_GPT_four(prompt)
                    #llmResponse = GeminiLLM.get_LLM_answer(prompt)
                    #print(llmResponse)

                    data.append({'IC_number': image_file, 'cropped_image': cropped_image_name, 'IC_prompt': IC_output, 'LLM_Response': llmResponse})

                
                # Specify the directory
                directoryCroppedSlices = 'DeepLearning/code/SOZDetectionUsingLLMs/testIC/croppedScans/'
                for cropped_image in cropped_image_files:
                    image_path = os.path.join(directoryCroppedSlices, cropped_image)
                    os.remove(image_path)

            # Create a DataFrame from the list of dictionaries
            df = pd.DataFrame(data)   

            # Export the DataFrame to an Excel file
            #excel_filename = 'Detailed_output_patient_1_results_shortPrompts_Trial4_tempchange_0.5.xlsx'
            excel_filename = excel_filename_result + '.xlsx'

            df.to_excel(excel_filename, index=False)
            print(f'Excel file "{excel_filename}" generated with information.')
        counter=counter+1
                