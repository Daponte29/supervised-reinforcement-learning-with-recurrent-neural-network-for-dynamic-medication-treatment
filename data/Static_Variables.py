#Queries of Raw MIMIC III database--------------


#Static Variables:
'''
-- Title: Extract height and weight for HADM_IDs
-- Description: This query gets the first, minimum, and maximum weight and height for each HADM_ID.
-- MIMIC version: MIMIC-III v1.4
--////////////////////////////////////////////////////
-- Prepare height
-- prep height
WITH ht_stg AS
(
  SELECT 
    c.subject_id, c.icustay_id, c.charttime,
    CASE
      WHEN c.itemid IN (920, 1394, 4187, 3486, 226707)
      THEN
        CASE
        WHEN c.charttime <= DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
         AND (c.valuenum * 2.54) < 80
          THEN c.valuenum * 2.54
        WHEN c.charttime >  DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
         AND (c.valuenum * 2.54) > 120
         AND (c.valuenum * 2.54) < 230
          THEN c.valuenum * 2.54
        ELSE NULL END
      ELSE
        CASE
        WHEN c.charttime <= DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
         AND c.valuenum < 80
          THEN c.valuenum
        WHEN c.charttime >  DATETIME_ADD(pt.dob, INTERVAL 1 YEAR)
         AND c.valuenum > 120
         AND c.valuenum < 230
          THEN c.valuenum
        ELSE NULL END
    END AS height
  FROM `physionet-data.mimiciii_clinical.chartevents` c
  INNER JOIN `physionet-data.mimiciii_clinical.patients` pt
    ON c.subject_id = pt.subject_id
  WHERE c.valuenum IS NOT NULL
  AND c.valuenum != 0
  AND COALESCE(c.error, 0) = 0
  AND c.itemid IN
  (
    920, 1394, 4187, 3486,
    3485, 4188,
    226707
  )
)
SELECT 
  ie.icustay_id,
  ie.hadm_id AS HADM_ID, -- Renamed to HADM_ID
  ROUND(CAST(wt.weight_first AS NUMERIC), 2) AS weight_first,
  ROUND(CAST(wt.weight_min AS NUMERIC), 2) AS weight_min,
  ROUND(CAST(wt.weight_max AS NUMERIC), 2) AS weight_max,
  ROUND(CAST(ht.height_first AS NUMERIC), 2) AS height_first,
  ROUND(CAST(ht.height_min AS NUMERIC), 2) AS height_min,
  ROUND(CAST(ht.height_max AS NUMERIC), 2) AS height_max
FROM `physionet-data.mimiciii_clinical.icustays` ie
LEFT JOIN
(
  SELECT icustay_id,
    MIN(CASE WHEN rn = 1 THEN weight ELSE NULL END) as weight_first,
    MIN(weight) AS weight_min,
    MAX(weight) AS weight_max
  FROM
  (
    SELECT
      icustay_id,
      weight,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) as rn
    FROM `physionet-data.mimiciii_derived.weight_durations`
  ) wt_stg
  GROUP BY icustay_id
) wt
  ON ie.icustay_id = wt.icustay_id
LEFT JOIN
(
  SELECT icustay_id,
    MIN(CASE WHEN rn = 1 THEN height ELSE NULL END) as height_first,
    MIN(height) AS height_min,
    MAX(height) AS height_max
  FROM
  (
    SELECT
      icustay_id,
      height,
      ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY charttime) as rn
    FROM ht_stg
  ) ht_stg2
  GROUP BY icustay_id
) ht
  ON ie.icustay_id = ht.icustay_id
ORDER BY ie.icustay_id;

--//^^need to filter above query for SUBJECT_ID with age less than 18

--//OTHER STATIC Variables//--------------------------------------------------------------------------------------------
SELECT
    adm.SUBJECT_ID,
    adm.HADM_ID,
    CASE 
        WHEN DATE_DIFF(adm.ADMITTIME, p.DOB, year) > 120 THEN 89
        ELSE DATE_DIFF(adm.ADMITTIME, p.DOB, year)
    END AS age_in_years,
    p.GENDER,
    adm.RELIGION,
    adm.ETHNICITY,
    adm.MARITAL_STATUS,
    adm.LANGUAGE,
    adm.ADMITTIME,
    adm.DISCHTIME,
    DATE_DIFF(adm.DISCHTIME, adm.ADMITTIME, day) AS treatment_plan_days,
    adm.DEATHTIME
FROM 
    `physionet-data.mimiciii_clinical.admissions` adm
JOIN 
    `physionet-data.mimiciii_clinical.patients` p ON adm.SUBJECT_ID = p.SUBJECT_ID
WHERE 
    DATE_DIFF(adm.ADMITTIME, p.DOB, year) >= 18


--////END STATIC VARIABLES FOR PATIENTS///----------------------------------------------------------------------------------   
'''


 



#Filtered Tables:--------------------------------------------------------------------------------------------------------------
filtered_tables ='''
 --//FILTER DIAGNOSES TABLE 
SELECT SUBJECT_ID,HADM_ID,ICD9_CODE
 FROM `physionet-data.mimiciii_clinical.diagnoses_icd` 
WHERE HADM_ID IN (
  SELECT DISTINCT(HADM_ID) FROM `sharp-nation-416018.static_var.static_var`) 
  AND ICD9_CODE IN ( 
    SELECT DISTINCT(ICD9_CODE) FROM  `sharp-nation-416018.2000_diag.2000_diag`);

 --//FILTER PERSCRIPTION TABLE
SELECT 

HADM_ID,
STARTDATE,
ENDDATE,
Chosen_ATC 
FROM `physionet-data.mimiciii_clinical.prescriptions` p
RIGHT JOIN `sharp-nation-416018.med.ndc_top_med` med ON med.NDC = p.NDC
WHERE HADM_ID IN (
  SELECT DISTINCT(HADM_ID) FROM `sharp-nation-416018.static_var.static_var`)
GROUP BY HADM_ID, STARTDATE, ENDDATE, Chosen_ATC;
  
--FILTER PERSCRIPTION TABLE BASED ON FINAL FILTERED HADM_ID 3/30
SELECT 
HADM_ID,
STARTDATE,
ENDDATE,
Chosen_ATC 
FROM `physionet-data.mimiciii_clinical.prescriptions` p
RIGHT JOIN `sharp-nation-416018.med.ndc_top_med` med ON med.NDC = p.NDC
WHERE HADM_ID IN (
  SELECT DISTINCT(HADM_ID) FROM `sharp-nation-416018.FINAL_HADM_ID.FINAL_HADM_ID` )
  GROUP BY HADM_ID, STARTDATE, ENDDATE, Chosen_ATC;

   '''
   
  