# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 07:59:57 2024

@author: nolot
"""
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import time
#import the final dataframe from method 2 of removing HADM_ID with more than 5 missing variables

#Load Data-------------------------------------------------
final_filtered_df_method2 = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_filtered_df_method2.csv")
unique_hadm_ids = final_filtered_df_method2['HADM_ID'].nunique()

# Find the index of the DEATHTIME column
death_column_index = final_filtered_df_method2.columns.get_loc("DEATHTIME")

# Move the age_in_years column to the position after the DEATHTIME column
df_columns = list(final_filtered_df_method2.columns)
age_index = df_columns.index("age_in_years")
df_columns.pop(age_index)
df_columns.insert(death_column_index + 1, "age_in_years")
final_filtered_df_method2 = final_filtered_df_method2[df_columns]


# Extract the age_in_years column and the last 16 columns of the DataFrame
columns_to_impute = final_filtered_df_method2.iloc[:, -17:]

# Specify the number of neighbors for the KNN imputer
n_neighbors = 5  # You can adjust this parameter based on your data and requirements

# Initialize the KNNImputer object with the specified number of neighbors
imputer = KNNImputer(n_neighbors=n_neighbors)

# Perform KNN imputation on the extracted columns
imputed_columns = imputer.fit_transform(columns_to_impute)

# Convert the imputed numpy array back to a DataFrame
imputed_df_columns = pd.DataFrame(imputed_columns, columns=columns_to_impute.columns)

# Concatenate the imputed DataFrame columns with the original DataFrame
imputed_df = pd.concat([final_filtered_df_method2.iloc[:, :-17], imputed_df_columns], axis=1)
final_filtered_df_method2_IMPUTED = imputed_df #rename for clarity

final_filtered_df_method2_IMPUTED.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_filtered_df_method2_IMPUTED.csv", index=False)
#Imputed all numerical values --> next will one hot encode categorical and impute the rest.





