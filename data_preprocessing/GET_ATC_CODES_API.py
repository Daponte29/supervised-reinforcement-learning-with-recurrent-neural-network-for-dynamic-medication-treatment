# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:06:23 2024

@author: nolot
"""

import numpy as np
import requests
import pandas as pd
import json
import xml.etree.ElementTree as ET
import ast
#Load Raw Data"
top_medications_df = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Top-medication.csv" )
top_medication_drug_names_df = pd.read_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\Top-medication-drug-names.csv")#the perscription table in MIMIC filtered by NDC in top 1000 medication  for Drug Names to use in API
file_path2=r"C:\Users\nolot\Downloads\empty_ndc_perscription.csv"




# Function to extract ATC codes from API response------API 1--------------------
def extract_atc_codes(rxcui):
    # Define base URL
    base_url = "https://rxnav.nlm.nih.gov"
    # Define API endpoint for getting ATC codes
    endpoint_atc = f"/REST/rxclass/class/byRxcui?rxcui={rxcui}&relaSource=ATC"
    # Construct full URL
    url_atc = f"{base_url}{endpoint_atc}"
    # Make GET request
    response_atc = requests.get(url_atc)
    # Check if request was successful (status code 200)
    if response_atc.status_code == 200:
        # Parse XML response
        root = ET.fromstring(response_atc.text)
        # Extract classId elements and get first 4 characters
        atc_codes = list(set([child.find('classId').text[:4] for child in root.iter('rxclassMinConceptItem')]))
        return atc_codes
    else:
        print(f"Failed to retrieve ATC codes for RxCUI: {rxcui}. Status code: {response_atc.status_code}")
        return ''
    
# Function to extract ATC codes BY DRUG NAME if no rxcui for NDC code ----------API 2-----------------
def extract_atc_codes_by_name(drug_name):
    # Define base URL
    base_url = "https://rxnav.nlm.nih.gov"
    # Define API endpoint for getting ATC codes by drug name
    endpoint_atc = f"/REST/rxclass/class/byDrugName.json?drugName={drug_name}&relaSource=ATCPROD"
    # Construct full URL
    url_atc = f"{base_url}{endpoint_atc}"
    # Make GET request
    response_atc = requests.get(url_atc)
    # Check if request was successful (status code 200)
    if response_atc.status_code == 200:
        # Parse JSON response
        data_atc = response_atc.json()
        # Extract unique classId values if response structure is correct
        if 'rxclassDrugInfoList' in data_atc and 'rxclassDrugInfo' in data_atc['rxclassDrugInfoList']:
            # Extract class IDs and take only the first 4 characters
            class_ids = [item['rxclassMinConceptItem']['classId'][:4] for item in data_atc['rxclassDrugInfoList']['rxclassDrugInfo']]
            return class_ids
        else:
            print("Unexpected response structure")
            return []
    else:
        print(f"Failed to retrieve ATC codes for drug: {drug_name}. Status code: {response_atc.status_code}")
        return []
#--------------------------------------------------------------------------------------------------------------






#Logic to Extract ATC THIRD LEVEL CODE FROM API Functions-------------------------------------------------------------------------------------

top_1000_medication = top_medications_df.nlargest(1000, 'frequency')
# Create an empty column 'ATC_CODE_THIRD_LVL'
top_1000_medication['ATC_CODE_THIRD_LVL'] = ''
# Loop through all rows in the DataFrame
top_medication_drug_names_df = pd.merge(top_1000_medication[['NDC']], top_medication_drug_names_df[['NDC', 'DRUG']], on='NDC', how='inner')

for index, row in top_1000_medication.iterrows():
    # Get the NDC value from the current row and pad with zeros
    ndc_original = int(row.iloc[0])
    ndc = str(row.iloc[0]).zfill(11)
    # Store the length of the original NDC before zfill
    original_ndc_length = len(str(row.iloc[0]))

    # Define base URL
    base_url = "https://rxnav.nlm.nih.gov"
    # Define API endpoint
    endpoint = "/REST/relatedndc.json"
    # Construct full URL with ndc parameter replaced by the NDC value
    url = f"{base_url}{endpoint}?ndc={ndc}&relation=drug&ndcstatus=all"

    # Make GET request
    response = requests.get(url)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        # Extract unique RxCUIs
        rxcui_list = [item['rxcui'] for item in data.get('ndcInfoList', {}).get('ndcInfo', [])]
        rxcui_list = list(set(rxcui_list))  # unique rxcui
        # List to store ATC codes for the current drug
        atc_codes_list = []

        # Loop through the list of RxCUIs
        for rxcui in rxcui_list:
            # Call the function to extract ATC codes
            atc_codes = extract_atc_codes(rxcui)
            # If ATC codes are retrieved, append them to the list
            if atc_codes:
                atc_codes_list.append(atc_codes)
                
        # # If ATC codes list is EMPTY, try to get ATC codes from prescription DataFrame by Drug Name
        if not atc_codes_list:
            # Filter prescription DataFrame by NDC without leading zeros
            prescription_match = top_medication_drug_names_df[top_medication_drug_names_df['NDC'] == ndc_original]
            # Check if there's a match in prescription DataFrame
            if not prescription_match.empty:
                for _, row in prescription_match.iterrows():
                    # Extract drug name from 'Drug_Candidate' column
                    drug_name = row['DRUG']
                    # Split drug name into words
                    words = drug_name.split()
                    # Iterate over each word
                    for i in range(len(words)):
                        # Extract the first i+1 words as the drug name
                        drug_name_cleaned = ' '.join(words[:i+1])
                        # Call function to extract ATC codes by drug name
                        atc_codes_by_name = extract_atc_codes_by_name(drug_name_cleaned)
                        # If ATC codes are retrieved, append them to the list and break the loop
                        if atc_codes_by_name:
                            atc_codes_list.extend(atc_codes_by_name)
                            break
                    # If ATC codes list is not empty, break the loop
                    if atc_codes_list:
                        break
        if not all(isinstance(code, str) for code in atc_codes_list):
            atc_codes_list = [code for sublist in atc_codes_list for code in sublist]

        atc_codes_list = list(set(atc_codes_list)) #geet only uniqye ATC codes third level
        # Add the list of ATC codes to the new column 'ATC_CODE_THIRD_LVL'
        top_1000_medication.at[index, 'ATC_CODE_THIRD_LVL'] = atc_codes_list

    else:
        print(f"Failed to retrieve data for NDC: {ndc}. Status code: {response.status_code}")


# Empty NDC lists that the API could not find ATC codes for   
#   # Create a DataFrame with NDCs having empty 'ATC_CODE_THIRD_LVL'
ndc_w_no_ATC = top_1000_medication.loc[top_1000_medication['ATC_CODE_THIRD_LVL'].apply(len) == 0, 'NDC']

ndc_w_no_ATC.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\ndc_w_no_ATC.csv",index=False)
# #--------------------------------------------------------------------------------------------------------------------------------------------





# #REDUCE MULTIPLE ATC to make it so its as close to 180 unique ATC Codes in paper----------------------------------------------------------------

# Read CSV into DataFrame
#####NOW we have just the rest of  the 64 drugs remaining out of teh 205 with no ATC code from drug name issue
# Function to convert string formatted as list to actual list
# def parse_list_string(s):
#     try:
#         # Safely evaluate the string literal as a Python object
#         parsed_list = ast.literal_eval(s)
#         # If the parsed object is a list, return it
#         if isinstance(parsed_list, list):
#             return [code.strip() for code in parsed_list]  # Strip whitespace from each code
#     except (SyntaxError, ValueError):
#         pass  # If parsing fails, return None or handle the error as needed
#     return []
# top_1000_medication['ATC_CODE_THIRD_LVL'] = top_1000_medication['ATC_CODE_THIRD_LVL'].apply(parse_list_string)
# # Truncate each string in the lists to keep only the first 4 characters
# top_1000_medication['ATC_CODE_THIRD_LVL'] = top_1000_medication['ATC_CODE_THIRD_LVL'].apply(lambda lst: [code[:4] for code in lst])
# # Filter out rows where the 'ATC_CODE_THIRD_LVL' column is an empty list
# top_1000_medication = top_1000_medication[top_1000_medication['ATC_CODE_THIRD_LVL'].astype(bool)]

# # Reset the index after filtering
# top_1000_medication.reset_index(drop=True, inplace=True)


atc_counts = {}
for atc_list in top_1000_medication["ATC_CODE_THIRD_LVL"]:
    for atc in atc_list:
        if atc in atc_counts:
            atc_counts[atc] += 1
        else:
            atc_counts[atc] = 1

ndc_to_atc = {}
for index, row in top_1000_medication.iterrows():
    print(index)
    if row["ATC_CODE_THIRD_LVL"]:  # Check if the list is not empty
        chosen_atc = sorted(row["ATC_CODE_THIRD_LVL"], key=lambda x: atc_counts.get(x, 0), reverse=True)[0]
        ndc_to_atc[row["NDC"]] = chosen_atc
    
final_NDC_ATC_df = pd.DataFrame(list(ndc_to_atc.items()), columns=['NDC', 'Chosen_ATC'])

unique_chosen_atc = final_NDC_ATC_df['Chosen_ATC'].drop_duplicates()

# If you want to reset the index of the resulting DataFrame:
unique_chosen_atc.reset_index(drop=True, inplace=True)

# If you want to convert the unique values back to a list:
unique_chosen_atc_list = unique_chosen_atc.tolist()
unique_chosen_atc_df = pd.DataFrame(unique_chosen_atc_list, columns=['Chosen_ATC'])

#save to csv
final_NDC_ATC_df.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\final_NDC_ATC_df.csv",index=False)
unique_chosen_atc.to_csv(r"C:\Users\nolot\OneDrive\Desktop\Project_TeamF2\Dynamic-Treatment-Recommendation-Supervised-Reinforcement-Learning-with-Recurrent-Neural-Network\RAW_DATA_QUERIES_MIMIC_III_DB\unique_chosen_atc.csv",index = False)




















   
