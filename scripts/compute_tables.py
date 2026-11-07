# -*- coding: utf-8 -*-
"""Compute Table 1-3 data and verify against paper numbers."""
import pandas as pd, numpy as np, os, json

D = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\data'
dfs = {db: pd.read_csv(os.path.join(D, f'{db}_t3_v2.csv')) for db in ['mimic','eicu','nwicu','sicdb','umcdb']}
pool = pd.concat(dfs.values(), ignore_index=True)
print('POOL n =', len(pool))
print('per db:', pool['db'].value_counts().to_dict())

at = pool[pool['baseline_at'].notna()].copy()
print('AT n =', len(at))
print('AT per db:', at['db'].value_counts().to_dict())

def cls(x):
    if pd.isna(x): return None
    parts = str(x).split('|')
    if 'warfarin' in parts: return 'VKA'
    if 'doac' in parts: return 'DOAC'
    return 'Antiplatelet'
at['cls_primary'] = at['baseline_at'].apply(cls)
print('class primary:', at['cls_primary'].value_counts().to_dict())
print('raw labels:', at['baseline_at'].value_counts().to_dict())

at['rev72'] = (at['reversal'].notna()) & (at['reversal_h_first'] <= 72)
print('reversal<=72h n =', int(at['rev72'].sum()))
print('reversal ever n =', int(at['reversal'].notna().sum()))
for c in ['VKA','DOAC','Antiplatelet']:
    sub = at[at['cls_primary'] == c]
    print('  %s: rev72 %d/%d (%.1f%%)' % (c, int(sub['rev72'].sum()), len(sub), sub['rev72'].mean()*100))
rv = at.loc[at['rev72'], 'reversal_h_first']
print('rev time median (IQR): %.1f [%.1f-%.1f]' % (rv.median(), rv.quantile(.25), rv.quantile(.75)))

at['rst48'] = (at['resume_h'] > 48)
print('restart>48h n =', int(at['rst48'].sum()))
rh = at.loc[at['rst48'], 'resume_h']
print('restart median (IQR): %.1f [%.1f-%.1f]' % (rh.median(), rh.quantile(.25), rh.quantile(.75)))
print('restarted among reversed:', int(at.loc[at['rev72'], 'rst48'].sum()), '/', int(at['rev72'].sum()))
print('tertile cutoffs:', np.percentile(rh, [33.3, 66.7]))
labels = pd.qcut(rh, 3, labels=['early','mid','late'])
for lab in ['early','mid','late']:
    sub = rh[labels == lab]
    print('  %s: n=%d median=%.1f' % (lab, len(sub), sub.median()))
for grp, sub in at.groupby(at['rst48']):
    print('rst48=%s n=%d death=%.1f%% throm=%.1f%%' % (grp, len(sub), sub['hosp_death'].mean()*100, sub['thrombosis_any'].mean()*100))

ev = pool[pool['ever_at'].notna()]
print('ever n =', len(ev))

# Table 1 characteristics by database and AT vs non-AT
print()
print('=== Table 1: pooled characteristics by database ===')
def q(x, fn):
    v = pd.to_numeric(x, errors='coerce')
    return fn(v.dropna())
for db in ['mimic','eicu','nwicu','sicdb','umcdb']:
    s = dfs[db]
    print('%s: n=%d age=%.0f(%.0f) male=%.1f%% gcs_med=%s charlson_med=%s icu_los=%.1f neuro=%.1f%% mv=%.1f%% death=%.1f%%' % (
        db, len(s),
        q(s['age'], np.mean), q(s['age'], np.std),
        (s['sex'].astype(str).str.upper() == 'M').mean()*100,
        q(s['gcs_total'], np.median), q(s['charlson_age'], np.median),
        q(s['icu_los_days'], np.median),
        s['neuro_surgery'].fillna(0).mean()*100, s['mv_flag'].fillna(0).mean()*100,
        s['hosp_death'].fillna(0).mean()*100))
print()
print('=== AT vs non-AT ===')
for grp, sub in [('AT', at), ('non-AT', pool[pool['baseline_at'].isna()])]:
    print('%s: n=%d age=%.1f(%.1f) male=%.1f%% gcs_med=%s charlson_med=%s icu_los_med=%.1f neuro=%.1f%% mv=%.1f%% death=%.1f%% throm=%.1f%%' % (
        grp, len(sub), q(sub['age'], np.mean), q(sub['age'], np.std),
        (sub['sex'].astype(str).str.upper() == 'M').mean()*100,
        q(sub['gcs_total'], np.median), q(sub['charlson_age'], np.median),
        q(sub['icu_los_days'], np.median),
        sub['neuro_surgery'].fillna(0).mean()*100, sub['mv_flag'].fillna(0).mean()*100,
        sub['hosp_death'].fillna(0).mean()*100, sub['thrombosis_any'].fillna(0).mean()*100))
