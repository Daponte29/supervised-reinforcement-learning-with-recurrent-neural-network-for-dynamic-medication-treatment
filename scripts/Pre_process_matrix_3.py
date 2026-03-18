# -*- coding: utf-8 -*- #Map
"""
Created on Fri Mar 29 11:20:25 2024

@author: nolot
"""
#Import Libraries Needed
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#LOAD DATA
final_filtered_df_method2_IMPUTED = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_filtered_df_method2_IMPUTED.csv")
# Get unique HADM_ID from final_filtered_df_method2_IMPUTED
unique_hadm_ids = final_filtered_df_method2_IMPUTED['HADM_ID'].unique()
df_final_chosen_ATC = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\unique_chosen_atc.csv")
df_final_chosen_ATC_NDC = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_NDC_ATC_df.csv")
top_2000_Diag = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\top_2000_diagnoses.csv")
Diagnosis_Filtered_by_top_2000_Diag = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Diagnosis_Table_filtered_Top2000_ICD9.csv")
#Filter Diagnosis further by HADM_ID in final_filtered_df_method2_IMPUTED
Filtered_Diagnosis = Diagnosis_Filtered_by_top_2000_Diag[Diagnosis_Filtered_by_top_2000_Diag['HADM_ID'].isin(unique_hadm_ids)]
#df_original = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Big Data Healthcare\PROJECT\Query_Tables\Static_var_not_height_weight.csv")
Medication_Filtered_by_chosen_ATC = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Perscription_Table_filtered_Chosen_ATC.csv")
#filter Medication further by HADM_ID in final_filtered_df_method2_IMPUTED
Filtered_Medication = Medication_Filtered_by_chosen_ATC[Medication_Filtered_by_chosen_ATC["HADM_ID"].isin(unique_hadm_ids)]
#----------------------------------------------------------------------------------








'''
Normalization is done in main Jupyter Notebook script after splitting data into Train,Validation,and Test sets to avoid Data Leakage in this timeseries problem
*We will move on to one-hot encoding and Imputing the categorical features in this script
'''



#IMPUTE REMAINING CATEGORICAL FEATURES ----USE MODE OR ONE HOT ENCODE AND THEN IMPUTE USING KNN? FOR NOW MODE OK ROUGH DRAFT
'''
Using Mode to Impute Categorical features
'''
# Replace missing values in religion, marital status, and language columns with mode
for col in ['RELIGION', 'MARITAL_STATUS', 'LANGUAGE']:
    final_filtered_df_method2_IMPUTED[col].fillna(final_filtered_df_method2_IMPUTED[col].mode()[0], inplace=True)
    
null_counts = final_filtered_df_method2_IMPUTED.isnull().sum()  #only DEATHTIME is NULL indicating no Death Date  

#Make LABELS FOR DEAD OR ALIVE FOR THAT ADMISSION TREATMENT PLAN----------------------------------------

# Initialize an empty dictionary to store Labels for Dead or Alive
labels = {}

# Group the DataFrame by HADM_ID
grouped = final_filtered_df_method2_IMPUTED.groupby('HADM_ID')


# Iterate over each group
for hadm_id, group_df in grouped:
    # Check if any value in the DEATHTIME column is not null
    if not group_df['DEATHTIME'].isnull().all():
        # If at least one value is not null, label as dead (1)
        labels[hadm_id] = 1 #Alive during admission treatment
    else:
        # If all values are null, label as alive (0)
        labels[hadm_id] = 0 #Dead during admission treatment
    
#df_imputed_normalized = df 




#Build One-Hot_Encode dictionary for categorial features

# Columns for which dictionaries will be created
target_columns = ['ETHNICITY', 'GENDER', 'LANGUAGE', 'MARITAL_STATUS', 'RELIGION']

# Initialize an empty dictionary to store mappings for each column
column_mappings = {}

# Iterate over each target column
for col in target_columns:
    # Get unique values in the column
    unique_values = final_filtered_df_method2_IMPUTED[col].unique()
    
    # Create a dictionary to map each unique value to its index in the list of unique values
    label_mapping = {value: i for i, value in enumerate(unique_values)}
    
    # Store the mapping for the column
    column_mappings[col] = label_mapping

# Apply codemap to dataframe
for column_name in target_columns:
    # Map each unique value to its index in the label mapping
    label_mapping = column_mappings[column_name]
    final_filtered_df_method2_IMPUTED[column_name] = final_filtered_df_method2_IMPUTED[column_name].map(label_mapping)

IMPUTED_FINAL = final_filtered_df_method2_IMPUTED
    

# Save the df_imputed_normalized DataFrame to a CSV file
IMPUTED_FINAL.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\IMPUTED_FINAL.csv", index=False)    




'''
Create Matrices for ML Model---------------------------------------------------------------------------------
'''

#Static Variable Matrix----------------------------------------------------
# Initialize the static matrix map
static_matrix_map = {}

# Assign an index to each static variable
index = 0
for column in ['GENDER', 'RELIGION', 'ETHNICITY', 'MARITAL_STATUS', 'LANGUAGE', 'age_in_years', 'avg_weight', 'avg_height']:
    static_matrix_map[column] = index
    index += 1


# Now static_matrix_map looks like this:
# {'GENDER': 0, 'RELIGION': 1, 'ETHNICITY': 2, 'MARITAL_STATUS': 3, 'LANGUAGE': 4, 'age_in_years': 5}

# Initialize a dictionary to store matrices for each HADM_ID
static_matrices = {}

# Iterate over each HADM_ID
for hadm_id, group_df in IMPUTED_FINAL.groupby('HADM_ID'):
    # Initialize a list to store static variables for the current HADM_ID
    static_variables = []
    
    # Iterate over static variable columns
    for column in ['GENDER', 'RELIGION', 'ETHNICITY', 'MARITAL_STATUS', 'LANGUAGE', 'age_in_years']:
        # Get the index of the static variable in the matrix
        index = static_matrix_map[column]
        
        # Get the value of the static variable for the current HADM_ID
        value = group_df[column].iloc[0]  # Assuming static variable value is the same for all rows
        
        # Append the normalized value to the list of static variables
        static_variables.append(value)
    
    # Convert the list of static variables to a numpy array
    static_matrix = np.array(static_variables)
    
    # Store the static matrix in the dictionary with HADM_ID as the key
    static_matrices[hadm_id] = static_matrix

    
#Time_Series_Matrix------------------------------------------------------------
# Initialize the time series map
time_series_map = {}

# Assign an index to each time series variable
index = 0
for column in ['heartrate', 'sysbp', 'diasbp', 'meanbp', 
               'resprate', 'tempc', 'glucose', 'urineoutput', 'pH_blood', 'pH_other', 
               'pH_urine', 'gcs', 'fio2']:
    time_series_map[column] = index
    index += 1


# Initialize a dictionary to store matrices for each HADM_ID
time_series_matrices = {}

# Iterate over each HADM_ID
for debug_hadm_id, group_df in IMPUTED_FINAL.groupby('HADM_ID'):
    # Sort the rows by time_stamp
    group_df_sorted = group_df.sort_values(by='time_stamp')

    # Find the minimum and maximum time_stamp values for this HADM_ID
    min_time_stamp = int(group_df_sorted['time_stamp'].min())
    max_time_stamp = int(group_df_sorted['time_stamp'].max())

    # Initialize the time series matrix
    num_rows = max_time_stamp + 1  # Account for padding up to 0
    num_cols = len(time_series_map)
    time_series_matrix = np.zeros((num_rows, num_cols))

    # Get the row where the time_stamp is equal to min_time_stamp
    first_row = group_df_sorted[group_df_sorted['time_stamp'] == min_time_stamp].iloc[0]

    # Fill the first rows of the matrix with the values of the first available time_stamp row
    for column in time_series_map:
        index = time_series_map[column]
        time_series_matrix[:min_time_stamp + 1, index] = first_row[column]

    # Iterate over rows of the sorted DataFrame
    for i, (_, row) in enumerate(group_df_sorted.iterrows()):
        # Fill the matrix row with time series values
        for column in time_series_map:
            index = time_series_map[column]
            time_series_matrix[int(row['time_stamp']), index] = row[column]

    # Store the time series matrix in the dictionary with HADM_ID as the key
    time_series_matrices[debug_hadm_id] = time_series_matrix
#Save as Pickle to load into Notebook later
# Folder path to save the pickle file
folder_path = r'C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB'

# Ensure the folder exists, create it if it doesn't
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Path to save the pickle file within the folder
pickle_file_path = os.path.join(folder_path, 'time_series_matrices.pkl')

# Save the dictionary as a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(time_series_matrices, pickle_file)
    
    
    
    

a=3





 




#FIX this so that its filtered by to get 29,000 uniqye admission n   #taking tiooooooo ooooooo lomg FILTER!!!!!!!!!!!! first this dffffffffffffff
unique_hadm_ids = list(static_matrices.keys())
final_filtered_df_method2_IMPUTED = final_filtered_df_method2_IMPUTED[final_filtered_df_method2_IMPUTED['HADM_ID'].isin(unique_hadm_ids)]

df2 = final_filtered_df_method2_IMPUTED[['HADM_ID', 'ADMITTIME']]

df = pd.merge(Filtered_Medication, df2, on='HADM_ID', how='inner')
# Filter out rows with null or NaN values
df = df.dropna()
df['STARTDATE'] = pd.to_datetime(df['STARTDATE'])
df['ENDDATE'] = pd.to_datetime(df['ENDDATE'])
df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['start_timestamp'] = (df['STARTDATE'] - df['ADMITTIME']).dt.total_seconds() // (24 * 3600)
df['end_timestamp'] = (df['ENDDATE'] - df['ADMITTIME']).dt.total_seconds() // (24 * 3600)
rows = []
for _, row in df.iterrows():
    for timestamp in range(int(row['start_timestamp']), int(row['end_timestamp']) + 1):
        rows.append({**row.to_dict(), 'timestamp': timestamp})
expanded_df = pd.DataFrame(rows)
final_df = expanded_df[['HADM_ID', 'Chosen_ATC', 'STARTDATE', 'ENDDATE', 'ADMITTIME', 'timestamp']]
# Replace values less than 0 in the timestamp column with 0
final_df.loc[final_df['timestamp'] < 0, 'timestamp'] = 0

unique_hadm_ids_final_df = final_df['HADM_ID'].unique()

num_unique_chosen_atc = final_df['Chosen_ATC'].nunique()

#SO I NEED TO MAKE ANOTHER MATRIX WHERE IT HAD NUM COLUMNS OF ATC_CODE and row is the timestamp of HADM_ID
num_features_perscription_matrix = len(df_final_chosen_ATC) # top ATC THIRD LVL CODES TOTAL used for ACTION SPACE

# Create a dictionary to store code map
ATC_code_map = {}

# Enumerate over Chosen_ATC codes in third_lvl_atc_codes_df
for index, row in df_final_chosen_ATC.iterrows():
    # Extract Chosen_ATC code
    chosen_atc = row['Chosen_ATC']

    # Assign an index to the Chosen_ATC code
    ATC_code_map[chosen_atc] = index


# Perscription treatment matrix------------------------------------------------------------
# Initialize an empty dictionary to store treatment matrices
treatment_matrices = {}

# Group final_df by HADM_ID
grouped_final_df = final_df.groupby('HADM_ID')

# Iterate over each group
for hadm_id, group_df in grouped_final_df:
    # Get unique timestamps for this HADM_ID
    timestamps = group_df['timestamp'].unique()
    
    # Initialize an empty list to store treatment matrix for this HADM_ID
    treatment_matrix_list = []
    
    # Iterate over each timestamp
    for timestamp in timestamps:
        # Initialize a numpy array for this timestamp with zeros
        timestamp_matrix = np.zeros(num_features_perscription_matrix)
        
        # Get the Chosen_ATC codes for this HADM_ID and timestamp
        chosen_atc_codes = group_df[group_df['timestamp'] == timestamp]['Chosen_ATC']
        
        # Iterate over Chosen_ATC codes for this timestamp
        for chosen_atc in chosen_atc_codes:
            # Get the index of the Chosen_ATC code from ATC_code_map
            index = ATC_code_map[chosen_atc]
            
            # Set the corresponding index in the timestamp_matrix to 1
            timestamp_matrix[index] = 1
        
        # Append the timestamp_matrix to the treatment matrix list
        treatment_matrix_list.append(timestamp_matrix)
    
    # Convert the list of matrices to a numpy array
    treatment_matrix_array = np.array(treatment_matrix_list)
    
    # Store the treatment matrix array in the dictionary with HADM_ID as the key
    treatment_matrices[hadm_id] = treatment_matrix_array
# Folder path to save the pickle file
folder_path = r'C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB'

# Ensure the folder exists, create it if it doesn't
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Path to save the pickle file within the folder
pickle_file_path = os.path.join(folder_path, 'treatment_matrices.pkl')

# Save the dictionary as a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(treatment_matrices, pickle_file)







#Diagnosis Matrix--------------------------------------------------------------------------------------------
# Get the unique ICD9_CODE values


# Create the code map dictionary
Diagnosis_codemap = {}

# Assign indices to unique ICD9_CODE values
for index, icd9_code in enumerate(top_2000_Diag['ICD9_CODE']):
    Diagnosis_codemap[icd9_code] = index

# Ensure that the dictionary contains mappings for all unique ICD9_CODE values
assert len(Diagnosis_codemap) == len(Diagnosis_codemap), "Missing mappings for some ICD9_CODE values"

# Check the length of the code map dictionary (should be equal to 2000)
print("Length of code map dictionary:", len(Diagnosis_codemap))


diagnosis_matrices = {}

# Group Filtered_Diagnosis dataframe by HADM_ID
grouped_diagnosis = Filtered_Diagnosis.groupby('HADM_ID')

# Iterate over each group
for hadm_id, group_df in grouped_diagnosis:
    # Initialize a binary matrix with 1 row and 2000 columns, filled with zeros
    matrix = np.zeros((1, 2000))
    
    # Get the ICD9_CODE values for this HADM_ID
    icd9_codes = group_df['ICD9_CODE'].values
    
    # Iterate over each ICD9_CODE value
    for icd9_code in icd9_codes:
        # Check if the ICD9_CODE exists in the codemap
        if icd9_code in Diagnosis_codemap:
            # Get the index from the codemap
            index = Diagnosis_codemap[icd9_code]
            # Set the corresponding index in the matrix to 1
            matrix[0, index] = 1
    
    # Store the matrix in the dictionary with HADM_ID as the key
    diagnosis_matrices[hadm_id] = matrix

#Difference in unique HADM_ID  despite query checks so filter remaining Matrices and Labels by the Diagnosis_Matrices since it is smaller and closer to papers value
unique_hadm_ids = Filtered_Diagnosis['HADM_ID'].unique()

static_matrices = {hadm_id: matrix for hadm_id, matrix in static_matrices.items() if hadm_id in unique_hadm_ids}
time_series_matrices = {hadm_id: matrix for hadm_id, matrix in time_series_matrices.items() if hadm_id in unique_hadm_ids}
#load treatment matrices pickle and then filter by HADM_ID 
# Path to the treatment_matrices.pickle file
pickle_file_path = r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\treatment_matrices.pickle"





# Load the data from the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    treatment_matrices = pickle.load(pickle_file)
labels = {hadm_id: matrix for hadm_id, matrix in labels.items() if hadm_id in unique_hadm_ids}




#Save as Pickle ----------------------------------------------------
# Folder path to save the pickle file
folder_path = r'C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB'

# Ensure the folder exists, create it if it doesn't
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Path to save the pickle file within the folder
pickle_file_path = os.path.join(folder_path, 'static_matrices.pkl')

# Save the dictionary as a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(static_matrices, pickle_file)



# Path to save the labels CSV file
# Convert the labels dictionary to a DataFrame and save it to a CSV file
labels_df = pd.DataFrame.from_dict(labels, orient='index', columns=['LABEL'])
labels_df.index.name = 'HADM_ID'
labels_df.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Final_Labels.csv")




#{} LAST DAY--------------------------GET DONE--------------------------
#{} all matrices same HADM_ID when you split to train for example for randomly to split static split static variavle or split lab variable: make sure same 
#make sure same HADM_ID when you split
#OUTPUT: dynamic ATC_CODE for each timestamp of HADM_ID --probablity of alive by our treatment --first result is treatment reccomendation for each HADM_ID

#[] 
#what is output and input into model_  --- READ ACTOR NAD CRITIC then SRL_RNN 




    
    
    
    
    
    
    
    
    
    
    
    