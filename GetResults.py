import re
import pandas as pd
import glob
from pathlib import Path

if __name__ == '__main__':
    
    # To sort file names numerically
    numbers = re.compile(r'(\d+)')
    
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    # Directory where the files are located
    subjects = 'DeepXSOZ 2022/Results/LLM Results/LLama3/8B-instruct/' + 'ASUAI_0*'
  
    # List to hold data for the final CSV
    csv_data = []

    # Iterate over each file matching the pattern
    for subject in sorted(glob.glob(subjects), key=numericalSort):
        # Initialize lists for the current subject
        result_list = []
        TP = []  # List to store true positive IC numbers

        # Read the Excel file using pandas
        excel_file = pd.read_excel(subject)
        subject_name = Path(subject).stem
        
        # Print the filename for reference
        print(f"Processing file: {subject}")
        
        # Check if the necessary columns exist
        if 'IC_number' in excel_file.columns and 'LLM_Response' in excel_file.columns:
            
            # Group by 'IC_number'
            grouped = excel_file.groupby('IC_number')
            
            # Iterate over each group (IC_number)
            for ic_name, group in grouped:
                # Extract the number from the 'IC_number' string
                match = re.match(r'IC_(\d+)_thresh\.png', ic_name)
                if match:
                    ic_number = int(match.group(1))  # Extract the number (e.g., 101)

                    # Convert 'LLM_Response' column to lowercase for case-insensitive comparison
                    if (group['LLM_Response'].str.lower() == 'yes').any():
                        result_list.append(ic_number)  # Append the extracted number if "Yes" is found
        else:
            print(f"Required columns not found in {subject}")
        
        # Match with ground truth labels
        ground_truth_path = f'PCHData/fmridatasets/labels/{subject_name}.csv'
        # Read the CSV file
        try:
            ground_truth_df = pd.read_csv(ground_truth_path)
            print(f"Processing ground truth labels for: {subject_name}")
            print(ground_truth_df.head())  # Print first few rows of the ground truth file for verification
            
            # Check for matching ICs and Labels
            for index, row in ground_truth_df.iterrows():
                ic_label = int(row['IC'])  # Directly convert IC to an integer
                if ic_label in result_list and row['Label'] == 3:
                    TP.append(ic_label)  # Append to TP list if Label is 3
            
        except FileNotFoundError:
            print(f"Ground truth file not found for: {subject_name}")
        
        # Calculate False Positives (FP)
        FP = list(set(result_list) - set(TP))  # FP = Result - TP

        # Print the result list, TP list, and FP list
        print("Result list:", result_list)
        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)
        
        # Collect data for the CSV
        csv_data.append([
            subject_name,
            ', '.join(map(str, result_list)),
            ', '.join(map(str, TP)),
            ', '.join(map(str, FP)),
            len(result_list),
            len(TP),
            len(FP)
        ])
    
    # Create a DataFrame and save it to CSV
    csv_df = pd.DataFrame(csv_data, columns=['Subject_name', 'Result_list', 'TP_List', 'FP_List', 
                                              'Machine_marked_count', 'TP_count', 'FP_count'])
    
    # Specify the output CSV file path
    output_csv_path = 'subject_results.csv'
    csv_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")
