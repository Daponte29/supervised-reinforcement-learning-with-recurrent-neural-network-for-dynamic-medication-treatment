-- ITEMIDs used:

-- CAREVUE
--    723 as gcsverbal
--    454 as gcsmotor
--    184 as gcseyes

-- METAVISION
--    223900 GCS - Verbal Response
--    223901 GCS - Motor Response
--    220739 GCS - Eye Opening

-- The code combines the ITEMIDs into the carevue itemids, then pivots those
-- So 223900 is changed to 723, then the ITEMID 723 is pivoted to form gcsverbal

-- Note:
--  The GCS for sedated patients is defaulted to 15 in this code.
--  This is in line with how the data is meant to be collected.
--  e.g., from the SAPS II publication:
--    For sedated patients, the Glasgow Coma Score before sedation was used.
--    This was ascertained either from interviewing the physician who ordered the sedation,
--    or by reviewing the patient's medical record.
CREATE TABLE mimic-iii-srl-rnn.SRL_RNN.gcs AS
with base as
(
  select ce.HADM_ID, ce.icustay_id, ce.charttime
  -- pivot each value into its own column
  , max(case when ce.ITEMID in (454,223901) then ce.valuenum else null end) as gcsmotor
  , max(case
      when ce.ITEMID in (723,223900) then ce.valuenum
      else null end) as gcsverbal
  , max(case when ce.ITEMID in (184,220739) then ce.valuenum else null end) as gcseyes
  -- convert the data into a number, reserving a value of 0 for ET/Trach
  , ROW_NUMBER ()
          OVER (PARTITION BY ce.icustay_id ORDER BY ce.charttime ASC) as rn
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  -- Isolate the desired GCS variables
  where ce.ITEMID in
  (
    -- 198 -- GCS
    -- GCS components, CareVue
    184, 454, 723
    -- GCS components, Metavision
    , 223900, 223901, 220739
  )
  -- exclude rows marked as error
  AND (ce.error IS NULL OR ce.error != 1)
  group by ce.HADM_ID, ce.icustay_id, ce.charttime),
base_2 as(
SELECT 
  hadm_id,
  icustay_id,
  charttime,
  COALESCE(
    gcsmotor,
    LAST_VALUE(gcsmotor IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(gcsmotor IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS gcsm,
  COALESCE(
    gcsverbal,
    LAST_VALUE(gcsverbal IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(gcsverbal IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS gcsv,
  COALESCE(
    gcseyes,
    LAST_VALUE(gcseyes IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    FIRST_VALUE(gcseyes IGNORE NULLS) OVER (
      PARTITION BY hadm_id, icustay_id ORDER BY charttime ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
  ) AS gcse
FROM 
  base
ORDER BY 
  hadm_id, icustay_id, charttime),
base_3 as(
select base_2.*, ad.admittime ,(gcsm+gcsv+gcse) as gcs,cast(TRUNC(DATETIME_DIFF(charttime,ADMITTIME,MINUTE)/1440)as INT)  as time_stamp --every 24 hours as an internal from admittime
from base_2
left join `physionet-data.mimiciii_clinical.admissions` ad on base_2.hadm_id=ad.hadm_id)
select hadm_id,time_stamp,avg(gcs) gcs
from base_3
where time_stamp>=0 and icustay_id is  not null
group by hadm_id,time_stamp
order by hadm_id,time_stamp
