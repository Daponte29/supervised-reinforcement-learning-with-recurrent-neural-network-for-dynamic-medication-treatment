CREATE TABLE mimic-iii-srl-rnn.SRL_RNN.time_series_variable AS
with base as(
  select vs.*,uo.urineoutput
  from mimic-iii-srl-rnn.SRL_RNN.vital_sign vs
  full outer join mimic-iii-srl-rnn.SRL_RNN.urineout uo on uo.hadm_id=vs.hadm_id and uo.time_stamp=vs.time_stamp
),
base_2 as(
  select base.*,ph.pH_blood,ph.pH_other,ph.pH_urine
  from base
  full outer join mimic-iii-srl-rnn.SRL_RNN.pH ph on ph.hadm_id=base.hadm_id and ph.time_stamp=base.time_stamp
),
base_3 as(
  select base_2.*,gcs.gcs
  from base_2
  full outer join mimic-iii-srl-rnn.SRL_RNN.gcs gcs on gcs.hadm_id=base_2.hadm_id and gcs.time_stamp=base_2.time_stamp
),
base_4 as(
  select base_3.*,fi.fio2
  from base_3
  full outer join mimic-iii-srl-rnn.SRL_RNN.fio2 fi on fi.hadm_id=base_3.hadm_id and fi.time_stamp=base_3.time_stamp
)
select *
from base_4
where hadm_id is not null
order by hadm_id,time_stamp
