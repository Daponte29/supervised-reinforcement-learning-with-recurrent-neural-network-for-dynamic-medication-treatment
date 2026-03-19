# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:19:19 2024

@author: nolot
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
#----------Load Data-------------------------------------------------------------


df_static_height_weight = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Big Data Healthcare\PROJECT\Query_Tables\Static_var_height_weight.csv")
df_static_other = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Static_var_not_height_weight.csv")
df_static_other = df_static_other.drop_duplicates(subset=['HADM_ID'])
df_time_series = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Time_series_variables.csv")
#------------------------------------------------------------------------------------

#checks---
# Group the DataFrame by HADM_ID
grouped_time_series = df_time_series.groupby('HADM_ID')


# List to store HADM_IDs with non-zero values in all time_stamp columns
non_zero_time_stamp_hadm_ids = []

# Iterate over each group
for hadm_id, group_df in grouped_time_series:
    # Check if all values in the time_stamp column are non-zero
    if (group_df['time_stamp'] != 0).all():
        non_zero_time_stamp_hadm_ids.append(hadm_id)

# Count unique HADM_ID values in the df_time_series DataFrame
unique_hadm_ids_other = df_static_other['HADM_ID'].nunique()
unique_hadm_ids_height_weight = df_static_height_weight['HADM_ID'].nunique()
icustay_count_per_hadm = df_static_height_weight.groupby('HADM_ID')['icustay_id'].size().reset_index(name='icustay_count')
#--End checks------




#----Preprocess static and time series dataframes----------------------------------------------------------------------------------
#make sure both static variable dfs have same HADM_ID
unique_hadm_ids_other = df_static_other['HADM_ID'].unique()
filtered_df_static_height_weight = df_static_height_weight[df_static_height_weight['HADM_ID'].isin(unique_hadm_ids_other)]
filtered_df_static_height_weight = filtered_df_static_height_weight.drop_duplicates(subset=['HADM_ID'])

# Remove rows where height_first or weight_first is NaN
# Calculate averages grouped by hadm_id, keeping NaN values
averages_df = df_static_height_weight.groupby('HADM_ID').agg({
    'weight_first': lambda x: x.mean(skipna=True),
    'height_first': lambda x: x.mean(skipna=True)
}).reset_index()
# Rename the columns
averages_df = averages_df.rename(columns={'weight_first': 'avg_weight', 'height_first': 'avg_height'})

unique_HADM_ID_list = df_static_other['HADM_ID'].unique().tolist()

filtered_averages_df = averages_df[averages_df['HADM_ID'].isin(unique_HADM_ID_list)]

# Perform left join between df_static_other and filtered_averages_df
merged_df = pd.merge(df_static_other, filtered_averages_df, on='HADM_ID', how='left')



# Count the number of null values in the merged dataframe for HADM_ID column from df_static_other
unmatched_hadm_ids_count = merged_df['HADM_ID'].isnull().sum()
#Now merge the static(both dataframes static) with time series dataframe
final_merged_df = pd.merge(merged_df, df_time_series, on='HADM_ID', how='inner')
unique_HADM_ID_FINAL_MERGED = final_merged_df['HADM_ID'].nunique()


#Remove Admissions/HADM_ID that have more than 5 missing variables per process in literature-----------------------
#exclude columns that wont be in calculatetion of missing variables per literature
exclude_columns = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'treatment_plan_days', 'DEATHTIME', 'time_stamp']
#METHOD 1 -----------------------------------------------
'''Method 1 removes only the timestamp that has more than 5 missing variables within the HADM_ID admission
'''

# # Initialize a list to store indices of rows to remove
# rows_to_remove = []

# # Iterate through each row in the DataFrame
# for index, row in final_merged_df.iterrows():
#     # Exclude specified columns from the row
#     filtered_row = row.drop(exclude_columns)
    
#     # Count missing values in the filtered row
#     missing_values_count = filtered_row.isnull().sum()
    
#     # Check if the row has more than 10 missing values
#     if missing_values_count > 5:
#         # Add the index of the row to the list for removal
#         rows_to_remove.append(index)

# # Remove rows with more than 10 missing values
# final_filtered_df = final_merged_df.drop(rows_to_remove)
# unique_HADM_ID = final_filtered_df["HADM_ID"].nunique()


#METHOD 2------------------------------------------------ 
'''
Method 2 removes the entire HADM_ID admission if any timestamps within the HADM_ID have more than 5 missing variables 
'''
# Initialize a list to store HADM_IDs to remove
hadm_ids_to_remove = []

# Group final_merged_df by HADM_ID
grouped_final_merged_df = final_merged_df.groupby('HADM_ID')

# Iterate over each group
for hadm_id, group_df in grouped_final_merged_df:
    # Initialize a flag to track if any row in the group has more than 10 missing values
    remove_hadm_id = False
    
    # Iterate over each row in the group
    for index, row in group_df.iterrows():
        # Exclude specified columns from the row
        filtered_row = row.drop(exclude_columns)
        
        # Count missing values in the filtered row
        missing_values_count = filtered_row.isnull().sum()
        
        # Check if the row has more than 10 missing values
        if missing_values_count > 5:
            # Mark the HADM_ID for removal
            remove_hadm_id = True
            break  # No need to continue checking other rows once marked for removal
    
    # If any row in the group has more than 10 missing values, mark the HADM_ID for removal
    if remove_hadm_id:
        hadm_ids_to_remove.append(hadm_id)

# Remove rows with HADM_IDs marked for removal
final_filtered_df_method2 = final_merged_df[~final_merged_df['HADM_ID'].isin(hadm_ids_to_remove)]
unique_HADM_ID_method2 = final_filtered_df_method2['HADM_ID'].nunique()

# Path to save the CSV file in the Downloads directory
output_csv_file_path = r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_filtered_df_method2.csv"

# Save the DataFrame to a CSV file

final_filtered_df_method2.to_csv(output_csv_file_path, index=False)




#NOTES
#Do this after impute missing values 
#[] static tensor just 1 row with normalized values for numeric and onehot encoded categorical
#[] normalize the time series variables too since all numeric 









