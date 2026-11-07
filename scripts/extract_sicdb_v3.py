# -*- coding: utf-8 -*-
"""T3 extraction v3: SICdb 1.0.8 (Salzburg). Cohort = ICD10Main S06 cases.
FIX: all Offsets are in SECONDS per official documentation
(https://www.sicdb.com/Documentation/Medication_Data: "Time in seconds after admission").
v2 bug: divided by 60 (minutes) -> all antithrombotic exposure windows empty.
Also fixes TimeOfStay (seconds) -> days conversion.
Drug names verified via d_references (medication.DrugID -> d_references.ReferenceGlobalID)."""
import duckdb, pandas as pd, numpy as np

con = duckdb.connect('F:/sicDB/sicdb.duckdb', read_only=True)
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/sicdb_t3_v3.csv'

cohort = con.execute("""
SELECT CaseID, PatientID, AgeOnAdmission, Sex, ICD10Main, ICD10MainText,
       ICUOffset, TimeOfStay, saps3, OffsetOfDeath, SurgicalSite, HospitalDischargeType, DischargeState
FROM cases WHERE ICD10Main LIKE 'S06%'""").fetchdf()
cohort = cohort.rename(columns={'CaseID':'stay_id','PatientID':'subject_id','AgeOnAdmission':'age','Sex':'sex'})
print('SICdb cohort:', len(cohort))

def sub(icd):
    if icd is None: return 'unspecified'
    c = icd.replace('.','').upper()
    if c.startswith('S064'): return 'edh'
    if c.startswith('S065'): return 'sdh'
    if c.startswith('S066'): return 'tsah'
    if c.startswith('S063'): return 'contusion'
    if c.startswith('S061'): return 'edema'
    if c.startswith('S062'): return 'diffuse'
    if c.startswith('S068'): return 'other'
    return 'unspecified'
cohort['s06_subtypes'] = cohort['ICD10Main'].apply(sub)

# ---- drug reference map (verified against d_references) ----
AT = {1657:'warfarin', 1750:'doac', 1954:'doac', 1955:'doac',
      1448:'antiplatelet', 1473:'antiplatelet', 1429:'antiplatelet', 1800:'antiplatelet', 1820:'antiplatelet',
      1536:'heparin', 1923:'heparin', 1558:'heparin'}
REV = {1477:'txa', 1530:'protamine', 2004:'idarucizumab', 2005:'andexanet', 2048:'platelets', 2088:'platelets'}
TX = {2046:'rbc', 2048:'plt', 2088:'plt'}

med = con.execute("""
SELECT CaseID, DrugID, "Offset", OffsetDrugEnd, IsSingleDose, GivenState FROM medication
WHERE CaseID IN (SELECT stay_id FROM cohort)""").fetchdf()
# FIX: seconds -> hours
med['h'] = pd.to_numeric(med['Offset'], errors='coerce') / 3600.0
med['cls'] = med['DrugID'].map(AT)
med['rev'] = med['DrugID'].map(REV)
med['tx'] = med['DrugID'].map(TX)

base = med[(med['h']>=-24)&(med['h']<=48)&(med['cls'].isin(['warfarin','doac','antiplatelet']))]
base_g = base.groupby('CaseID')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('baseline_at').reset_index()
rev = med[(med['h']>=0)&(med['h']<=72)&(med['rev'].notna())]
rev_g = rev.groupby('CaseID')['rev'].agg(lambda s:'|'.join(sorted(set(s)))).rename('reversal').reset_index()
rev_first = rev.sort_values('h').groupby('CaseID').first()['h'].rename('reversal_h_first').reset_index()
ever = med[(med['h']>=-24)&(med['h']<=30*24)&(med['cls'].isin(['warfarin','doac','antiplatelet']))]
ever_g = ever.groupby('CaseID')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('ever_at').reset_index()
res = med[(med['h']>48)&(med['cls'].isin(['warfarin','doac','antiplatelet']))].sort_values('h').groupby('CaseID').first().reset_index()
res_g = res[['CaseID','cls','h']].rename(columns={'cls':'resume_cls','h':'resume_h'})
cohort = cohort.merge(base_g, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort = cohort.merge(rev_g, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort = cohort.merge(rev_first, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort = cohort.merge(ever_g, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort = cohort.merge(res_g, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort['last_anticoag_stop_h'] = np.nan

# ---- labs (offsets also seconds -> hours) ----
LAB = {3128:'inr', 597:'aptt', 314:'plt', 315:'plt', 289:'hgb', 288:'hgb', 658:'hgb', 344:'fib'}
lab = con.execute("""
SELECT CaseID, LaboratoryID, "Offset", LaboratoryValue FROM laboratory
WHERE CaseID IN (SELECT stay_id FROM cohort)""").fetchdf()
lab['val'] = pd.to_numeric(lab['LaboratoryValue'], errors='coerce')
lab['h'] = pd.to_numeric(lab['Offset'], errors='coerce') / 3600.0
lab['lab'] = lab['LaboratoryID'].map(LAB)
lab = lab[lab['lab'].notna() & lab['val'].notna()]
first24 = lab[(lab['h']>=-24)&(lab['h']<=24)].sort_values('h').groupby(['CaseID','lab']).first().reset_index()
coag = first24.pivot_table(index='CaseID', columns='lab', values='val', aggfunc='first').reset_index()
coag.columns = ['CaseID'] + ['admit_'+c for c in coag.columns[1:]]
cohort = cohort.merge(coag, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
for _c in ['admit_inr','admit_aptt','admit_plt','admit_fib','admit_hgb']:
    if _c not in cohort.columns:
        cohort[_c] = np.nan
cohort = cohort.rename(columns={'admit_aptt':'admit_ptt'})
hgb72 = lab[(lab['lab']=='hgb')&(lab['h']>=-24)&(lab['h']<=72)].groupby('CaseID')['val'].min().rename('hgb_min72').reset_index()
cohort = cohort.merge(hgb72, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort['hb_drop2'] = ((cohort['admit_hgb']-cohort['hgb_min72'])>=2).astype(float)
cohort['hb_drop2'] = cohort['hb_drop2'].where(cohort['admit_hgb'].notna() & cohort['hgb_min72'].notna())

# ---- transfusions ----
tx = med[med['tx'].notna()]
txg = tx.groupby('CaseID').agg(
    tx_rbc=('tx', lambda s:int((s=='rbc').sum())),
    tx_plt=('tx', lambda s:int((s=='plt').sum())),
    tx_any=('tx', lambda s:int(len(s)>0)),
    tx_first_h=('h','min')).reset_index()
txg['tx_ffp'] = 0
cohort = cohort.merge(txg, left_on='stay_id', right_on='CaseID', how='left').drop(columns=['CaseID'])
cohort['tx_any'] = cohort['tx_any'].fillna(0).astype(int)
cohort['tx_rbc'] = cohort['tx_rbc'].fillna(0).astype(int)
cohort['tx_plt'] = cohort['tx_plt'].fillna(0).astype(int)
cohort['tx_ffp'] = cohort['tx_ffp'].fillna(0).astype(int)

# ---- surgery / MV / GCS / charlson ----
cohort['neuro_surgery'] = np.nan  # SurgicalSite mostly undocumented in S06; treated as unknown
cohort['mv_flag'] = np.nan
cohort['gcs_total'] = np.nan
cohort['gcs_motor_only'] = np.nan
cohort['admit_map'] = np.nan
cohort['charlson'] = np.nan
cohort['charlson_age'] = np.nan

# ---- outcomes ----
cohort['hosp_death'] = cohort['OffsetOfDeath'].notna().astype(int)
# FIX: TimeOfStay is in seconds -> days
cohort['hosp_los_days'] = pd.to_numeric(cohort['TimeOfStay'], errors='coerce') / 86400.0
cohort['icu_los_days'] = pd.to_numeric(cohort['TimeOfStay'], errors='coerce') / 86400.0
# thrombosis not coded in SICdb secondary diagnoses
for c in ['pe','mi','is','dvt','arte','thrombosis_any']:
    cohort[c] = np.nan

cohort['db'] = 'sicdb'
keep = ['db','subject_id','stay_id','age','sex','icu_los_days','hosp_los_days','s06_subtypes',
        'gcs_total','gcs_motor_only','admit_map','admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb','hgb_min72','hb_drop2',
        'charlson','charlson_age','neuro_surgery','mv_flag','saps3',
        'baseline_at','ever_at','reversal','reversal_h_first','resume_cls','resume_h','last_anticoag_stop_h',
        'tx_rbc','tx_ffp','tx_plt','tx_any','tx_first_h',
        'pe','mi','is','dvt','arte','thrombosis_any',
        'hosp_death']
cohort[keep].to_csv(OUT, index=False)
print('saved', OUT, len(cohort))
print('baseline_at:', cohort['baseline_at'].value_counts(dropna=False).to_dict())
print('ever_at:', cohort['ever_at'].value_counts(dropna=False).to_dict())
print('reversal:', cohort['reversal'].value_counts(dropna=False).head(8).to_dict())
print('resume:', cohort['resume_cls'].value_counts(dropna=False).head(8).to_dict())
print('hosp_death:', cohort['hosp_death'].sum())
print('icu_los_days median:', cohort['icu_los_days'].median())
