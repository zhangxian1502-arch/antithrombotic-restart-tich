# -*- coding: utf-8 -*-
"""T3 extraction: eICU-CRD 2.0. Cohort = CNS intracranial injury (trauma). Baseline via admissionDrug; meds via medication."""
import duckdb, pandas as pd, numpy as np

con = duckdb.connect('F:/EICU 2.0/eicu.duckdb', read_only=True)
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/eicu_t3.csv'

# ---------- 1. Cohort ----------
q = """
SELECT p.patientunitstayid, p.uniquepid, p.hospitalid, p.gender, p.age, p.ethnicity,
       p.hospitaladmitoffset, p.unitdischargeoffset, p.unitdischargestatus,
       p.hospitaldischargestatus
FROM patient p
WHERE p.patientunitstayid IN (
  SELECT DISTINCT d.patientunitstayid FROM diagnosis d
  WHERE d.diagnosisstring LIKE '%trauma - CNS%intracranial injury%')
"""
cohort = con.execute(q).fetchdf()
ap = con.execute("SELECT patientunitstayid, apachescore FROM apachepatientresult WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)").fetchdf()
ap['apachescore'] = pd.to_numeric(ap['apachescore'], errors='coerce')
ap = ap[ap['apachescore']>0].sort_values('apachescore', ascending=False).drop_duplicates('patientunitstayid')
cohort = cohort.merge(ap[['patientunitstayid','apachescore']], on='patientunitstayid', how='left')
print('eICU cohort:', len(cohort))
# subtype from diagnosis strings
diag = con.execute("""
SELECT patientunitstayid, diagnosisstring FROM diagnosis
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
def sub(s):
    s = s.lower()
    if 'subdural' in s: return 'sdh'
    if 'subarachnoid' in s: return 'tsah'
    if 'intracerebral hemorrhage' in s: return 'tich'
    if 'epidural' in s: return 'edh'
    return 'other'
diag['sub'] = diag['diagnosisstring'].apply(sub)
subsum = diag[diag['sub']!='other'].groupby('patientunitstayid')['sub'].agg(lambda s:'|'.join(sorted(set(s)))).rename('s06_subtypes').reset_index()
cohort = cohort.merge(subsum, on='patientunitstayid', how='left')
cohort['s06_subtypes'] = cohort['s06_subtypes'].fillna('unspecified')
# ICU anchor: unitadmitoffset is minutes from hospital admission to ICU admission

# ---------- 2. Baseline antithrombotic: admissionDrug (home meds) ----------
def med_class(drug):
    if drug is None: return None
    dl = drug.lower()
    if 'warfarin' in dl: return 'warfarin'
    if any(x in dl for x in ['apixaban','rivaroxaban','dabigatran','edoxaban','eliquis','xarelto','pradaxa']): return 'doac'
    if any(x in dl for x in ['aspirin','clopidogrel','ticagrelor','prasugrel','dipyridamole','cilostazol','plavix','brilinta']): return 'antiplatelet'
    if any(x in dl for x in ['enoxaparin','lovenox','heparin','dalteparin','fondaparinux']): return 'heparin'
    return None

adm_drug = con.execute("""
SELECT patientunitstayid, drugname FROM admissionDrug
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
adm_drug['cls'] = adm_drug['drugname'].apply(med_class)
base = adm_drug[adm_drug['cls'].isin(['warfarin','doac','antiplatelet'])]
base_g = base.groupby('patientunitstayid')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('baseline_at').reset_index()
cohort = cohort.merge(base_g, on='patientunitstayid', how='left')

# ---------- 3. In-hospital meds: reversal (0-72h of ICU), resumption (>48h), ever ----------
def reversal_class(drug):
    if drug is None: return None
    dl = drug.lower()
    if any(x in dl for x in ['phytonadione','vitamin k','mephyton']): return 'vitk'
    if any(x in dl for x in ['kcentra','prothrombin complex','pcc','prothrombinex','4-factor','four factor']): return 'pcc'
    if any(x in dl for x in ['fresh frozen','ffp','plasma']): return 'ffp'
    if any(x in dl for x in ['platelet','platelets']): return 'platelets'
    if any(x in dl for x in ['idarucizumab','praxbind']): return 'idarucizumab'
    if any(x in dl for x in ['andexanet','andexxa']): return 'andexanet'
    if any(x in dl for x in ['protamine']): return 'protamine'
    if any(x in dl for x in ['tranexamic','txa']): return 'txa'
    return None

med = con.execute("""
SELECT patientunitstayid, drugname, drugstartoffset, drugstopoffset
FROM medication WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
med['h'] = pd.to_numeric(med['drugstartoffset'], errors='coerce') / 60.0
med['cls'] = med['drugname'].apply(med_class)
med['rev'] = med['drugname'].apply(reversal_class)
rev = med[(med['h']>=0)&(med['h']<=72)&(med['rev'].notna())]
rev_g = rev.groupby('patientunitstayid')['rev'].agg(lambda s:'|'.join(sorted(set(s)))).rename('reversal').reset_index()
rev_first = rev.sort_values('h').groupby('patientunitstayid').first()['h'].rename('reversal_h_first').reset_index()
ever = med[(med['h']>=-24)&(med['h']<=30*24)&(med['cls'].isin(['warfarin','doac','antiplatelet']))]
ever_g = ever.groupby('patientunitstayid')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('ever_at').reset_index()
res = med[(med['h']>48)&(med['cls'].isin(['warfarin','doac','antiplatelet']))].sort_values('h').groupby('patientunitstayid').first().reset_index()
res_g = res[['patientunitstayid','cls','h']].rename(columns={'cls':'resume_cls','h':'resume_h'})
cohort = cohort.merge(rev_g, on='patientunitstayid', how='left')
cohort = cohort.merge(rev_first, on='patientunitstayid', how='left')
cohort = cohort.merge(ever_g, on='patientunitstayid', how='left')
cohort = cohort.merge(res_g, on='patientunitstayid', how='left')

# reversal agents may also be in infusionDrug
inf = con.execute("""
SELECT patientunitstayid, drugname, infusionoffset FROM infusionDrug
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
inf['rev'] = inf['drugname'].apply(reversal_class)
inf['h'] = pd.to_numeric(inf['infusionoffset'], errors='coerce') / 60.0
inf_rev = inf[(inf['h']>=0)&(inf['h']<=72)&(inf['rev'].notna())]
if len(inf_rev):
    ir_g = inf_rev.groupby('patientunitstayid')['rev'].agg(lambda s:'|'.join(sorted(set(s)))).rename('reversal_inf').reset_index()
    ir_first = inf_rev.sort_values('h').groupby('patientunitstayid').first()['h'].rename('reversal_inf_h_first').reset_index()
    cohort = cohort.merge(ir_g, on='patientunitstayid', how='left')
    cohort = cohort.merge(ir_first, on='patientunitstayid', how='left')
    comb = pd.concat([rev_g, ir_g.rename(columns={'reversal_inf':'reversal'})])
    comb = comb.groupby('patientunitstayid')['reversal'].agg(lambda s:'|'.join(sorted(set(s)))).reset_index()
    cohort = cohort.drop(columns=['reversal'])
    cohort = cohort.merge(comb, on='patientunitstayid', how='left')
    rf = pd.concat([rev_first, ir_first.rename(columns={'reversal_inf_h_first':'reversal_h_first'})])
    rf = rf.groupby('patientunitstayid')['reversal_h_first'].min().reset_index()
    cohort = cohort.drop(columns=['reversal_h_first'])
    cohort = cohort.merge(rf, on='patientunitstayid', how='left')

# ---------- 4. Labs: first in 24h; Hgb min 72h ----------
lab = con.execute("""
SELECT patientunitstayid, labname, labresult, labresultoffset FROM lab
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)
    AND (lower(labname) LIKE '%inr%' OR lower(labname) LIKE '%ptt%' OR lower(labname) LIKE '%platelet%'
       OR lower(labname) LIKE '%hgb%' OR lower(labname) LIKE '%fibrinogen%')""").fetchdf()
lab['result'] = pd.to_numeric(lab['labresult'], errors='coerce')
lab['h'] = pd.to_numeric(lab['labresultoffset'], errors='coerce') / 60.0
def lab_cls(n):
    n = (n or '').lower()
    if 'inr' in n: return 'inr'
    if 'ptt' in n and 'ratio' not in n: return 'ptt'
    if 'platelet' in n: return 'plt'
    if 'hgb' in n: return 'hgb'
    if 'fibrinogen' in n: return 'fib'
    return None
lab['lab'] = lab['labname'].apply(lab_cls)
lab = lab[lab['lab'].notna() & lab['result'].notna()]
first24 = lab[(lab['h']>=-24)&(lab['h']<=24)].sort_values('h').groupby(['patientunitstayid','lab']).first().reset_index()
coag = first24.pivot_table(index='patientunitstayid', columns='lab', values='result', aggfunc='first').reset_index()
coag.columns = ['patientunitstayid'] + ['admit_'+c for c in coag.columns[1:]]
cohort = cohort.merge(coag, on='patientunitstayid', how='left')
hgb72 = lab[(lab['lab']=='hgb')&(lab['h']>=-24)&(lab['h']<=72)].groupby('patientunitstayid')['result'].min().rename('hgb_min72').reset_index()
cohort = cohort.merge(hgb72, on='patientunitstayid', how='left')
cohort['hb_drop2'] = ((cohort['admit_hgb']-cohort['hgb_min72'])>=2).astype(float)
cohort['hb_drop2'] = cohort['hb_drop2'].where(cohort['admit_hgb'].notna() & cohort['hgb_min72'].notna())

# ---------- 5. GCS (nurseCharting), transfusions & surgery (treatment) ----------
nc = con.execute("""
SELECT patientunitstayid, nursingchartoffset, nursingchartvalue
FROM nurseCharting WHERE nursingchartcelltypevalname='GCS Total'
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
nc['val'] = pd.to_numeric(nc['nursingchartvalue'], errors='coerce')
nc['h'] = pd.to_numeric(nc['nursingchartoffset'], errors='coerce') / 60.0
nc = nc[(nc['h']>=-24)&(nc['h']<=24)&nc['val'].notna()]
gcs_first = nc.sort_values('h').groupby('patientunitstayid').first()['val'].rename('gcs_total').reset_index()
cohort = cohort.merge(gcs_first, on='patientunitstayid', how='left')
cohort['gcs_motor_only'] = np.nan

trt = con.execute("""
SELECT patientunitstayid, treatmentstring, treatmentoffset FROM treatment
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort)""").fetchdf()
def tx_cls(s):
    if s is None: return None
    s = s.lower()
    if 'packed red blood' in s or 'prbc' in s or 'red blood cell' in s: return 'rbc'
    if 'fresh frozen' in s or 'ffp' in s or 'plasma' in s: return 'ffp'
    if 'platelet' in s: return 'plt'
    if 'cryoprecipitate' in s: return 'cryo'
    return None
trt['tx'] = trt['treatmentstring'].apply(tx_cls)
txd = trt[trt['tx'].notna()].copy()
txd['h'] = pd.to_numeric(txd['treatmentoffset'], errors='coerce') / 60.0
txg = txd.groupby('patientunitstayid').agg(
    tx_rbc=('tx', lambda s:int((s=='rbc').sum())),
    tx_ffp=('tx', lambda s:int((s=='ffp').sum())),
    tx_plt=('tx', lambda s:int((s=='plt').sum())),
    tx_any=('tx', lambda s:int(len(s)>0)),
    tx_first_h=('h','min')).reset_index()
cohort = cohort.merge(txg, on='patientunitstayid', how='left')
cohort['tx_any'] = cohort['tx_any'].fillna(0).astype(int)

ns = trt[trt['treatmentstring'].str.contains('crani|burr hole|ventriculostomy', case=False, na=False)]
cohort['neuro_surgery'] = cohort['patientunitstayid'].isin(ns['patientunitstayid']).astype(int)
mv = trt[trt['treatmentstring'].str.contains('ventilat', case=False, na=False)]
cohort['mv_flag'] = cohort['patientunitstayid'].isin(mv['patientunitstayid']).astype(int)

# ---------- 6. Thrombotic events via diagnosis strings ----------
def thrombo(s):
    if s is None: return dict(pe=0, mi=0, is_=0, dvt=0, arte=0)
    s = s.lower()
    return dict(pe=int('pulmonary embol' in s),
                mi=int('myocardial infarction' in s),
                is_=int('ischemic stroke' in s or 'ischemic cerebral' in s),
                dvt=int('deep vein thromb' in s or 'dvt' in s),
                arte=int('arterial embolism' in s))
trows = diag.apply(lambda r: pd.Series({'patientunitstayid':r['patientunitstayid'], **thrombo(r['diagnosisstring'])}), axis=1)
tg = trows.groupby('patientunitstayid')[['pe','mi','is_','dvt','arte']].max().reset_index()
tg['thrombosis_any'] = tg[['pe','mi','is_','dvt','arte']].max(axis=1)
cohort = cohort.merge(tg, on='patientunitstayid', how='left')
for c in ['pe','mi','is_','dvt','arte','thrombosis_any']:
    cohort[c] = cohort[c].fillna(0).astype(int)

# ---------- 7. Outcomes & tidy ----------
cohort['hosp_death'] = (cohort['hospitaldischargestatus']=='Expired').astype(int)
cohort['age_num'] = pd.to_numeric(cohort['age'].replace('> 89','90'), errors='coerce')
cohort['icu_los_days'] = pd.to_numeric(cohort['unitdischargeoffset'], errors='coerce')/1440.0
cohort['db'] = 'eicu'
cohort = cohort.rename(columns={'patientunitstayid':'stay_id','uniquepid':'subject_id','age_num':'age'})
cohort['sex'] = cohort['gender']
cohort['charlson'] = np.nan
cohort['charlson_age'] = np.nan
cohort['admit_map'] = np.nan
cohort['gcs_motor_only'] = np.nan
cohort['hosp_los_days'] = np.nan
cohort['admit_fib'] = cohort.get('admit_fib')
cohort['last_anticoag_stop_h'] = np.nan
cohort['is'] = cohort['is_']

keep = ['db','subject_id','stay_id','age','sex','icu_los_days','hosp_los_days','s06_subtypes',
        'gcs_total','gcs_motor_only','admit_map','admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb','hgb_min72','hb_drop2',
        'charlson','charlson_age','neuro_surgery','mv_flag',
        'baseline_at','ever_at','reversal','reversal_h_first','resume_cls','resume_h','last_anticoag_stop_h',
        'tx_rbc','tx_ffp','tx_plt','tx_any','tx_first_h',
        'pe','mi','is','dvt','arte','thrombosis_any',
        'hosp_death']
cohort[keep].to_csv(OUT, index=False)
print('saved', OUT, len(cohort))
print('baseline_at:', cohort['baseline_at'].value_counts(dropna=False).to_dict())
print('reversal:', cohort['reversal'].value_counts(dropna=False).head(8).to_dict())
print('hosp_death:', cohort['hosp_death'].sum())
