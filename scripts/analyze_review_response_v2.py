# -*- coding: utf-8 -*-
"""Re-analysis v2 (fixed NaN handling + reduced covariates for strata)."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, fisher_exact

warnings.filterwarnings('ignore')

P = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\data\pool_t3_v3.csv'
OUT_JSON = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\review_response_analyses_v2.json'
OUT_TXT = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\review_response_analyses_v2.txt'

df = pd.read_csv(P, low_memory=False)
for c in ['age', 'gcs_total', 'gcs_motor_only', 'charlson_age', 'charlson', 'hr1', 'sbp1', 'spo2_1',
          'temp_c1', 'l_wbc', 'l_cr', 'l_glu', 'l_na', 'l_k', 'l_lac', 'l_alb', 'l_tbili', 'vaso_flag',
          'mv_flag', 'neuro_surgery', 'admit_to_icu_h', 'cc_cvd', 'cc_renal', 'cc_dm', 'cc_cancer',
          'hosp_death', 'hb_drop2', 'tx_any', 'thrombosis_any', 'resume_h', 'hosp_los_days',
          'icu_los_days', 'admit_inr', 'admit_hgb']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
for c in ['tx_rbc', 'tx_ffp', 'tx_plt']:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
df['male'] = (df['sex'].astype(str).str.lower().str.startswith('m')).astype(int)
df['gcs_any'] = df['gcs_total'].where(df['gcs_total'].notna(), df['gcs_motor_only'])

ex = df[df['baseline_at'].notna()].copy()

def comp(s):
    return set(str(s).split('|')) if pd.notna(s) else set()

DRUGS = ['vitk', 'pcc', 'idarucizumab', 'andexanet', 'protamine']
BLOOD = ['ffp', 'platelets']
ex['rev_strict'] = ex['reversal'].apply(lambda s: int(any(k in comp(s) for k in DRUGS)))
ex['rev_any'] = ex['reversal'].notna().astype(int)
ex['restarted'] = ex['resume_cls'].notna().astype(int)
ex['tx_rbc_bin'] = (ex['tx_rbc'] > 0).astype(int)
ex['tx_ffp_bin'] = (ex['tx_ffp'] > 0).astype(int)
ex['tx_plt_bin'] = (ex['tx_plt'] > 0).astype(int)
ex['sdh_any'] = ex['s06_subtypes'].astype(str).str.contains('sdh').astype(int)
ex['class_ap'] = ex['baseline_at'].str.contains('antiplatelet').astype(int)
ex['class_vka'] = ex['baseline_at'].str.contains('warfarin').astype(int)
ex['class_doac'] = ex['baseline_at'].str.contains('doac').astype(int)

FULL = ['hr1', 'sbp1', 'spo2_1', 'temp_c1', 'l_wbc', 'l_cr', 'l_glu', 'l_na', 'l_k', 'l_lac', 'l_alb',
        'l_tbili', 'vaso_flag', 'mv_flag', 'neuro_surgery', 'admit_to_icu_h', 'cc_cvd', 'cc_renal',
        'cc_dm', 'cc_cancer']
MIN = ['age', 'male', 'gcs_any', 'charlson_age']
DBS = ['mimic', 'eicu', 'nwicu', 'sicdb', 'umcdb']

def _prep(d, covars):
    Xd = pd.DataFrame(index=d.index)
    for c in covars:
        if c in d.columns:
            Xd[c] = pd.to_numeric(d[c], errors='coerce')
    keep = []
    for c in Xd.columns:
        v = Xd[c]
        if v.notna().sum() == 0:
            continue
        if v.nunique(dropna=True) <= 1:
            continue
        med = v.median()
        if pd.isna(med):
            continue
        Xd[c] = v.fillna(med)
        keep.append(c)
    Xd = Xd[keep]
    if 'db' in d.columns:
        for db in DBS[1:]:
            col = (d['db'] == db).astype(int)
            if col.nunique() > 1:
                Xd[f'db_{db}'] = col
    return Xd

def firth_fit(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float)
    assert np.isfinite(X).all(), 'non-finite covariate'
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(400):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
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
        beta = beta + np.clip(delta, -10, 10)
        if np.max(np.abs(delta)) < 1e-9:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta)); w = np.clip(mu * (1 - mu), 1e-10, None)
    XWX = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.inv(XWX + np.eye(p) * 1e-8)
    return beta, np.sqrt(np.diag(cov))

def firth_run(outcome, exposure, data, covars=None, label=''):
    covars = MIN + FULL if covars is None else covars
    d = data[data[outcome].notna()].copy()
    d[exposure] = pd.to_numeric(d[exposure], errors='coerce')
    d = d[d[exposure].notna()]
    if d[exposure].nunique() < 2 or d[outcome].sum() < 3:
        return {'label': label, 'error': 'insufficient', 'n': int(len(d)), 'events': int(d[outcome].sum())}
    Xd = _prep(d, covars)
    Xd[exposure] = d[exposure].values
    y = d[outcome].astype(float)
    try:
        beta, se = firth_fit(y, np.column_stack([np.ones(len(d)), Xd.values.astype(float)]))
    except Exception as e:
        return {'label': label, 'error': str(e)[:80], 'n': int(len(d)), 'events': int(d[outcome].sum())}
    idx = Xd.columns.get_loc(exposure) + 1
    b, s = beta[idx], se[idx]
    if not np.isfinite(b) or not np.isfinite(s) or s <= 0:
        return {'label': label, 'error': 'non-finite', 'n': int(len(d)), 'events': int(d[outcome].sum())}
    return {'label': label, 'or': round(float(np.exp(b)), 2), 'lo': round(float(np.exp(b - 1.96 * s)), 2),
            'hi': round(float(np.exp(b + 1.96 * s)), 2),
            'p': round(float(2 * (1 - norm.cdf(abs(b / s)))), 4), 'n': int(len(d)),
            'events': int(d[outcome].sum())}

def crude_2x2(data, expo, out):
    d = data[[expo, out]].dropna()
    tab = pd.crosstab(d[expo], d[out])
    if tab.shape != (2, 2):
        return None
    a = int(tab.iloc[1, 1]); b = int(tab.iloc[1, 0]); c = int(tab.iloc[0, 1]); e = int(tab.iloc[0, 0])
    orv = (a * e) / (b * c) if b * c else np.inf
    odds, p = fisher_exact([[a, b], [c, e]])
    return {'a': a, 'b': b, 'c': c, 'd': e, 'crude_or': round(float(orv), 2),
            'fisher_p': round(float(p), 4), 'risk_exposed': round(a / (a + b) * 100, 1) if a + b else None,
            'risk_unexposed': round(c / (c + e) * 100, 1) if c + e else None}

R = {'n_exposed': int(len(ex))}
R['reversal_composition'] = {
    'strict_drug': int(ex['rev_strict'].sum()), 'any_reversal': int(ex['rev_any'].sum()),
    'vitk': int(ex['reversal'].apply(lambda s: 'vitk' in comp(s)).sum()),
    'pcc': int(ex['reversal'].apply(lambda s: 'pcc' in comp(s)).sum()),
    'ffp': int(ex['reversal'].apply(lambda s: 'ffp' in comp(s)).sum()),
    'platelets': int(ex['reversal'].apply(lambda s: 'platelets' in comp(s)).sum()),
    'txa': int(ex['reversal'].apply(lambda s: 'txa' in comp(s)).sum()),
    'protamine': int(ex['reversal'].apply(lambda s: 'protamine' in comp(s)).sum()),
}

print('=== 1.1 strict vs any reversal ===')
R['r11'] = {}
for tag, expo in [('strict', 'rev_strict'), ('any', 'rev_any')]:
    for out in ['tx_any', 'tx_rbc_bin', 'tx_ffp_bin', 'tx_plt_bin', 'hosp_death', 'hb_drop2', 'thrombosis_any']:
        res = firth_run(out, expo, ex, label=f'{tag}->{out}')
        R['r11'][f'{tag}->{out}'] = res
        R['r11'][f'{tag}->{out}_crude'] = crude_2x2(ex, expo, out)
        print(f'  {tag} -> {out}: {res} | crude {R["r11"][f"{tag}->{out}_crude"]}')

print('\n=== 1.2 early exit sensitivity (reliable LOS only: mimic/nwicu/sicdb) ===')
# UMCdb LOS values are implausible (<1e-4 days) -> exclude that database from early-exit definition
reliable = ex[ex['db'].isin(['mimic', 'eicu', 'nwicu', 'sicdb'])].copy()
reliable['early_exit'] = ((reliable['hosp_death'].fillna(0).astype(int) == 1) |
                          (reliable['hosp_los_days'] < 2)).astype(int)
R['r12'] = {
    'early_exit_n': int(reliable['early_exit'].sum()),
    'early_death_n': int(((reliable['hosp_death'].fillna(0).astype(int) == 1) & (reliable['hosp_los_days'] < 2)).sum()),
    'early_discharge_n': int(((reliable['hosp_death'].fillna(0).astype(int) == 0) & (reliable['hosp_los_days'] < 2)).sum()),
    'umcdb_los_note': 'UMCdb hosp_los_days values <1e-4 day (unit artefact); excluded from this sensitivity definition',
}
for out in ['hosp_death', 'thrombosis_any', 'hb_drop2']:
    R['r12'][f'all->{out}'] = firth_run(out, 'restarted', ex, label=f'all->{out}')
    R['r12'][f'excl_early_exit->{out}'] = firth_run(out, 'restarted', reliable[reliable['early_exit'] == 0],
                                                   label=f'excl->{out}')
    print(f'  {out}: all {R["r12"][f"all->{out}"]} | excl {R["r12"][f"excl_early_exit->{out}"]}')

print('\n=== 1.4 SDH strata ===')
R['r14'] = {'sdh_n': int(ex['sdh_any'].sum()), 'sdh_pct': round(ex['sdh_any'].mean() * 100, 1)}
sub = ['age', 'male', 'gcs_any', 'charlson_age', 'neuro_surgery', 'mv_flag', 'vaso_flag']
for out in ['thrombosis_any', 'hosp_death']:
    for tag, d in [('sdh', ex[ex['sdh_any'] == 1]), ('non_sdh', ex[ex['sdh_any'] == 0])]:
        R['r14'][f'{tag}->{out}'] = firth_run(out, 'restarted', d, covars=sub, label=f'{tag}->{out}')
        R['r14'][f'{tag}->{out}_crude'] = crude_2x2(d, 'restarted', out)
        print(f'  {tag} -> {out}: {R["r14"][f"{tag}->{out}"]} | crude {R["r14"][f"{tag}->{out}_crude"]}')

print('\n=== 1.5 AT class strata ===')
R['r15'] = {'n_ap': int(ex['class_ap'].sum()), 'n_vka': int(ex['class_vka'].sum()), 'n_doac': int(ex['class_doac'].sum())}
for out in ['thrombosis_any', 'hosp_death']:
    for tag, d in [('ap_only', ex[ex['baseline_at'] == 'antiplatelet']),
                   ('vka_any', ex[ex['class_vka'] == 1]),
                   ('doac_any', ex[ex['class_doac'] == 1])]:
        R['r15'][f'{tag}->{out}'] = firth_run(out, 'restarted', d, covars=sub, label=f'{tag}->{out}')
        R['r15'][f'{tag}->{out}_crude'] = crude_2x2(d, 'restarted', out)
        print(f'  {tag} -> {out}: {R["r15"][f"{tag}->{out}"]} | crude {R["r15"][f"{tag}->{out}_crude"]}')

print('\n=== 1.7 per-db exposure / availability ===')
R['r17'] = {}
for db, g in df.groupby('db'):
    ge = g[g['baseline_at'].notna()]
    R['r17'][db] = {
        'n': int(len(g)), 'exposed': int(len(ge)), 'exposed_pct': round(len(ge) / len(g) * 100, 1),
        'reversal_pct': round(ge['reversal'].notna().mean() * 100, 1),
        'restart_pct': round(ge['resume_cls'].notna().mean() * 100, 1),
        'thrombosis_available_pct': round(g['thrombosis_any'].notna().mean() * 100, 1),
        'hb_drop2_available_pct': round(g['hb_drop2'].notna().mean() * 100, 1),
        'age_missing_pct': round(g['age'].isna().mean() * 100, 1),
        'gcs_missing_pct': round(g['gcs_any'].isna().mean() * 100, 1),
        'inr_missing_pct': round(g['admit_inr'].isna().mean() * 100, 1),
    }
    print(f'  {db}: {R["r17"][db]}')
R['r17']['exposed_key_missing'] = {c: round(ex[c].isna().mean() * 100, 1)
                                   for c in ['age', 'gcs_any', 'admit_inr', 'admit_hgb', 'l_lac',
                                             'charlson', 'thrombosis_any', 'hb_drop2', 'hosp_death']}
print('  exposed key missingness:', R['r17']['exposed_key_missing'])

json.dump(R, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
lines = ['RE-ANALYSIS v2 (fixed NaN handling)', '=' * 70, f'exposed n={len(ex)}', '']
for k, v in R.items():
    lines.append(f'## {k}')
    lines.append(json.dumps(v, ensure_ascii=False, indent=1, default=str))
open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(lines))
print('\nDONE ->', OUT_JSON)
