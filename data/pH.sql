CREATE TABLE mimic-iii-srl-rnn.SRL_RNN.pH AS
with ph as(select hadm_id, charttime,
(case
    when itemid=50820
      then case
        when valuenum>0 and valuenum <14
          then valuenum
        else null end
    end) as pH_blood,
(case
    when itemid=50831
      then case
        when valuenum>0 and valuenum <14
          then valuenum
        else null end
    end) as pH_other,
(case
    when itemid in (51094,51491)
      then case
        when valuenum>0 and valuenum <14
          then valuenum
        else null end
    end) as pH_urine
FROM `physionet-data.mimiciii_clinical.labevents`
where ITEMID in (  50820 --pH	Blood	Blood Gas
  ,50831 --	pH	Other Body Fluid	Blood Gas
  ,51094 --	pH	Urine	Chemistry
  ,51491 --	pH	Urine	Hematology
  ) and
  HADM_ID is not null),
ph2 as(
select ph.hadm_id,charttime,avg(pH_blood) pH_blood ,avg(pH_other) pH_other,avg(pH_urine) pH_urine
from ph
group by hadm_id,charttime)
select ph2.hadm_id,avg(pH_blood) pH_blood ,avg(pH_other) pH_other,avg(pH_urine) pH_urine,cast(TRUNC(DATETIME_DIFF(charttime,ADMITTIME,MINUTE)/1440)as INT)  as time_stamp
from ph2
left join `physionet-data.mimiciii_clinical.admissions` ad on ph2.hadm_id=ad.hadm_id
group by HADM_ID,time_stamp
order by HADM_ID,time_stamp