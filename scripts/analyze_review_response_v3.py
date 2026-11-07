# -*- coding: utf-8 -*-
"""v3: corrected early-exit definition + robust hb_drop2 models (reduced covariates)."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, fisher_exact

warnings.filterwarnings('ignore')

P = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\data\pool_t3_v3.csv'
OUT = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\review_response_analyses_v3.json'

df = pd.read_csv(P, low_memory=False)
for c in ['age', 'gcs_total', 'gcs_motor_only', 'charlson_age', 'hosp_death', 'hb_drop2',
          'thrombosis_any', 'hosp_los_days', 'admit_inr']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
df['male'] = (df['sex'].astype(str).str.lower().str.startswith('m')).astype(int)
df['gcs_any'] = df['gcs_total'].where(df['gcs_total'].notna(), df['gcs_motor_only'])
ex = df[df['baseline_at'].notna()].copy()
ex['restarted'] = ex['resume_cls'].notna().astype(int)

def firth_fit(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float)
    assert np.isfinite(X).all()
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(500):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        XWX = X.T @ (X * w[:, None])
        try:
            XWXI = np.linalg.inv(XWX)
        except np.linalg.LinAlgError:
            XWX = XWX + np.eye(p) * 1e-8; XWXI = np.linalg.inv(XWX)
        h = np.sum((X @ XWXI) * (X * w[:, None]), axis=1)
        score = X.T @ (y - mu + h * (0.5 - mu))
        try:
            delta = np.linalg.solve(XWX, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(XWX + np.eye(p) * 1e-8, score)
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-9:
            break
    eta = X @ beta
    mu = 1 / (1 + np.exp(-np.clip(eta, -35, 35))); w = np.clip(mu * (1 - mu), 1e-12, None)
    XWX = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.inv(XWX + np.eye(p) * 1e-8)
    return beta, np.sqrt(np.diag(cov))

def run(out, expo, data, covars):
    d = data[data[out].notna()].copy()
    d[expo] = pd.to_numeric(d[expo], errors='coerce'); d = d[d[expo].notna()]
    if d[expo].nunique() < 2 or d[out].sum() < 3:
        return {'error': 'insufficient', 'n': len(d), 'events': int(d[out].sum())}
    Xd = pd.DataFrame(index=d.index)
    for c in covars:
        if c in d:
            v = pd.to_numeric(d[c], errors='coerce')
            if v.notna().sum() == 0 or v.nunique(dropna=True) <= 1:
                continue
            med = v.median()
            if pd.isna(med):
                continue
            Xd[c] = v.fillna(med)
    for db in ['eicu', 'nwicu', 'sicdb', 'umcdb']:
        col = (d['db'] == db).astype(int)
        if col.nunique() > 1:
            Xd[f'db_{db}'] = col
    Xd[expo] = d[expo].values
    try:
        beta, se = firth_fit(d[out].astype(float), np.column_stack([np.ones(len(d)), Xd.values.astype(float)]))
    except Exception as e:
        return {'error': str(e)[:80], 'n': len(d), 'events': int(d[out].sum())}
    idx = Xd.columns.get_loc(expo) + 1
    b, s = beta[idx], se[idx]
    if not np.isfinite(b) or not np.isfinite(s) or s <= 0:
        return {'error': 'non-finite', 'n': len(d), 'events': int(d[out].sum()), 'b': float(b), 'se': float(s)}
    return {'or': round(float(np.exp(b)), 2), 'lo': round(float(np.exp(b - 1.96 * s)), 2),
            'hi': round(float(np.exp(b + 1.96 * s)), 2),
            'p': round(float(2 * (1 - norm.cdf(abs(b / s)))), 4), 'n': len(d), 'events': int(d[out].sum())}

MIN = ['age', 'male', 'gcs_any', 'charlson_age']
TINY = ['age', 'male', 'gcs_any']

R = {}
print('=== corrected early-exit (LOS<48h only; reliable LOS dbs) ===')
rel = ex[ex['db'].isin(['mimic', 'eicu', 'nwicu', 'sicdb'])].copy()
rel['early_exit'] = (rel['hosp_los_days'] < 2).astype(int)
R['early_exit'] = {
    'n_total': int(len(rel)), 'n_early_exit': int(rel['early_exit'].sum()),
    'n_early_death': int(((rel['early_exit'] == 1) & (rel['hosp_death'].fillna(0) == 1)).sum()),
    'n_early_discharge': int(((rel['early_exit'] == 1) & (rel['hosp_death'].fillna(0) == 0)).sum()),
    'restarted_in_early_exit': int(rel.loc[rel['early_exit'] == 1, 'restarted'].sum()),
}
print(R['early_exit'])
keep = rel[rel['early_exit'] == 0]
R['early_exit_models'] = {}
for out in ['hosp_death', 'thrombosis_any', 'hb_drop2']:
    R['early_exit_models'][f'all->{out}'] = run(out, 'restarted', ex, MIN)
    R['early_exit_models'][f'excl->{out}'] = run(out, 'restarted', keep, MIN)
    print(f'  {out}: all {R["early_exit_models"][f"all->{out}"]} | excl {R["early_exit_models"][f"excl->{out}"]}')

print('\n=== hb_drop2 with reduced covariate sets ===')
R['hb_drop2'] = {}
for covname, covs in [('min4', MIN), ('tiny3', TINY)]:
    for tag, expo in [('restart', 'restarted'), ('strict_rev', None)]:
        if expo is None:
            continue
        R['hb_drop2'][f'{tag}_{covname}'] = run('hb_drop2', expo, ex, covs)
        print(f'  {tag} {covname}: {R["hb_drop2"][f"{tag}_{covname}"]}')
# crude fisher for hb_drop2 restart
d = ex[['restarted', 'hb_drop2']].dropna()
tab = pd.crosstab(d['restarted'], d['hb_drop2'])
a, b = int(tab.iloc[1, 1]), int(tab.iloc[1, 0]); c, e = int(tab.iloc[0, 1]), int(tab.iloc[0, 0])
R['hb_drop2']['crude_restart'] = {'a': a, 'b': b, 'c': c, 'd': e,
                                  'crude_or': round((a * e) / (b * c), 2), 'fisher_p': round(float(fisher_exact([[a, b], [c, e]])[1]), 4)}
print('  crude restart:', R['hb_drop2']['crude_restart'])

json.dump(R, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
print('\nsaved', OUT)
