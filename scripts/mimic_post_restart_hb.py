# -*- coding: utf-8 -*-
"""Post-restart haemoglobin decline in MIMIC-IV exposed cohort (Reviewer #1 point 3)."""
import duckdb
import numpy as np
import pandas as pd

POOL = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\data\pool_t3_v3.csv'
DB = 'F:/MIMIC3.1/mimiciv.duckdb'
OUT = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\mimic_post_restart_hb.json'

df = pd.read_csv(POOL, low_memory=False)
print('pool columns with id/time:')
print([c for c in df.columns if any(k in c.lower() for k in ['id', 'time', 'stay', 'hadm', 'subject', 'intime', 'outtime'])])

m = df[(df['db'] == 'mimic') & (df['baseline_at'].notna())].copy()
print('\nMIMIC exposed n =', len(m))
print(m[['subject_id', 'hadm_id', 'stay_id', 'icu_intime', 'icu_outtime', 'resume_h', 'hosp_death']].head(10).to_string())

con = duckdb.connect(DB, read_only=True)
# haemoglobin itemids in MIMIC-IV: 50813 (Haemoglobin, blood gas) and 51222 (Haemoglobin, lab)
hb = con.execute("""
    SELECT subject_id, hadm_id, charttime, itemid, valuenum
    FROM hosp.labevents
    WHERE itemid IN (50813, 51222) AND valuenum IS NOT NULL AND valuenum BETWEEN 1 AND 25
""").fetchdf()
print('\nHb rows:', len(hb), '| subjects:', hb['subject_id'].nunique())

if 'icu_intime' in m.columns and m['icu_intime'].notna().any():
    m['intime_dt'] = pd.to_datetime(m['icu_intime'], errors='coerce')
else:
    m['intime_dt'] = pd.NA

res = []
for _, r in m.iterrows():
    if pd.isna(r.get('resume_h')) or pd.isna(r['intime_dt']):
        continue
    t_restart = r['intime_dt'] + pd.Timedelta(hours=float(r['resume_h']))
    sub = hb[(hb['subject_id'] == int(r['subject_id'])) & (hb['hadm_id'] == int(float(r['hadm_id'])))] .copy()
    sub['charttime'] = pd.to_datetime(sub['charttime'])
    pre = sub[(sub['charttime'] >= t_restart - pd.Timedelta(hours=24)) & (sub['charttime'] <= t_restart)]
    post = sub[(sub['charttime'] > t_restart) & (sub['charttime'] <= t_restart + pd.Timedelta(hours=72))]
    if len(pre) == 0 or len(post) == 0:
        res.append({'subject_id': r['subject_id'], 'resume_h': r['resume_h'], 'pre_hb': np.nan,
                    'post_min_hb': np.nan, 'drop_2': np.nan})
        continue
    pre_hb = pre.sort_values('charttime')['valuenum'].iloc[-1]
    post_min = post['valuenum'].min()
    res.append({'subject_id': r['subject_id'], 'resume_h': r['resume_h'], 'pre_hb': float(pre_hb),
                'post_min_hb': float(post_min), 'drop_2': int((pre_hb - post_min) >= 2.0)})

rr = pd.DataFrame(res)
print('\nEvaluable restarted patients:', len(rr))
print('with paired Hb:', int(rr['drop_2'].notna().sum()))
if rr['drop_2'].notna().any():
    n = int(rr['drop_2'].notna().sum()); ev = int(rr['drop_2'].sum())
    print(f'post-restart Hb drop >=2 g/dL: {ev}/{n} ({ev/n*100:.1f}%)')
    print('median pre-Hb:', round(rr['pre_hb'].median(), 2), '| median post-min:', round(rr['post_min_hb'].median(), 2))

rr.to_csv(r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\mimic_post_restart_hb.csv', index=False)
import json
json.dump({'n_restarted_mimic': int(len(rr)), 'n_paired': int(rr['drop_2'].notna().sum()),
           'n_drop2': int(rr['drop_2'].sum()) if rr['drop_2'].notna().any() else 0},
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved', OUT)
