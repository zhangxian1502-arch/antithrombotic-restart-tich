# -*- coding: utf-8 -*-
"""T3 extraction: NWICU 0.1.0. Cohort = S06 ICU stays. MIMIC-like schema."""
import duckdb, pandas as pd, numpy as np

con = duckdb.connect('F:/NWICU/nwicu.duckdb', read_only=True)
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/nwicu_t3.csv'

# ---------- 1. Cohort ----------
cohort = con.execute("""
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los AS icu_los_days,
       a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.discharge_location,
       p.gender, p.anchor_age
FROM nw_icu.icustays i
JOIN nw_hosp.admissions a USING (hadm_id)
JOIN nw_hosp.patients p ON i.subject_id = p.subject_id
WHERE i.stay_id IN (
  SELECT DISTINCT i2.stay_id FROM nw_icu.icustays i2
  JOIN nw_hosp.diagnoses_icd d USING (hadm_id)
  WHERE d.icd_code LIKE 'S06%')""").fetchdf()
print('NWICU cohort:', len(cohort))
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
cohort['intime'] = pd.to_datetime(cohort['intime'])
cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
cohort['hosp_los_days'] = (cohort['dischtime']-cohort['admittime']).dt.total_seconds()/86400.0

d = con.execute("""
SELECT d.hadm_id, d.icd_code FROM nw_hosp.diagnoses_icd d
WHERE d.icd_code LIKE 'S06%' AND d.hadm_id IN (SELECT hadm_id FROM cohort)""").fetchdf()
d['sub'] = np.where(d.icd_code.str.startswith('S061'),'edema',
          np.where(d.icd_code.str.startswith('S062'),'diffuse',
          np.where(d.icd_code.str.startswith('S063'),'contusion',
          np.where(d.icd_code.str.startswith('S064'),'edh',
          np.where(d.icd_code.str.startswith('S065'),'sdh',
          np.where(d.icd_code.str.startswith('S066'),'tsah',
          np.where(d.icd_code.str.startswith('S068'),'other','unspecified')))))))
subsum = d.groupby('hadm_id')['sub'].agg(lambda s:'|'.join(sorted(set(s)))).rename('s06_subtypes').reset_index()
cohort = cohort.merge(subsum, on='hadm_id', how='left')

# ---------- 2. Prescriptions: baseline [-24,48h], reversal [0,72h], resumption >48h, ever ----------
def med_class(drug):
    if drug is None or (isinstance(drug,float) and pd.isna(drug)): return None
    dl = drug.lower()
    if 'warfarin' in dl: return 'warfarin'
    if any(x in dl for x in ['apixaban','rivaroxaban','dabigatran','edoxaban','eliquis','xarelto','pradaxa']): return 'doac'
    if any(x in dl for x in ['aspirin','clopidogrel','ticagrelor','prasugrel','dipyridamole','cilostazol','plavix','brilinta']): return 'antiplatelet'
    if any(x in dl for x in ['enoxaparin','lovenox','heparin','dalteparin','fondaparinux']): return 'heparin'
    return None

def reversal_class(drug):
    if drug is None or (isinstance(drug,float) and pd.isna(drug)): return None
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

rx = con.execute("""
SELECT hadm_id, starttime, stoptime, drug FROM nw_hosp.prescriptions
WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").fetchdf()
rx['start'] = pd.to_datetime(rx['starttime'])
rx['hadm_id'] = pd.to_numeric(rx['hadm_id'], errors='coerce').astype('Int64')
admit = cohort[['hadm_id','intime']].rename(columns={'intime':'anchor'}).drop_duplicates()
rx = rx.merge(admit, on='hadm_id', how='left')
rx['h'] = rx.apply(lambda r: (r['start']-r['anchor']).total_seconds()/3600.0 if pd.notna(r['start']) and pd.notna(r['anchor']) else np.nan, axis=1)
rx['cls'] = rx['drug'].apply(med_class)
rx['rev'] = rx['drug'].apply(reversal_class)

base = rx[(rx['h']>=-24)&(rx['h']<=48)&(rx['cls'].isin(['warfarin','doac','antiplatelet']))]
base_g = base.groupby('hadm_id')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('baseline_at').reset_index()
rev = rx[(rx['h']>=0)&(rx['h']<=72)&(rx['rev'].notna())]
rev_g = rev.groupby('hadm_id')['rev'].agg(lambda s:'|'.join(sorted(set(s)))).rename('reversal').reset_index()
rev_first = rev.sort_values('h').groupby('hadm_id').first()['h'].rename('reversal_h_first').reset_index()
ever = rx[(rx['h']>=-24)&(rx['h']<=30*24)&(rx['cls'].isin(['warfarin','doac','antiplatelet']))]
ever_g = ever.groupby('hadm_id')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('ever_at').reset_index()
res = rx[(rx['h']>48)&(rx['cls'].isin(['warfarin','doac','antiplatelet']))].sort_values('h').groupby('hadm_id').first().reset_index()
res_g = res[['hadm_id','cls','h']].rename(columns={'cls':'resume_cls','h':'resume_h'})
cohort = cohort.merge(base_g, on='hadm_id', how='left')
cohort = cohort.merge(rev_g, on='hadm_id', how='left')
cohort = cohort.merge(rev_first, on='hadm_id', how='left')
cohort = cohort.merge(ever_g, on='hadm_id', how='left')
cohort = cohort.merge(res_g, on='hadm_id', how='left')
cohort['last_anticoag_stop_h'] = np.nan

# ---------- 3. Labs ----------
lab = con.execute("""
SELECT l.hadm_id, l.itemid, l.charttime, l.valuenum FROM nw_hosp.labevents l
WHERE l.hadm_id IN (SELECT hadm_id FROM cohort)
  AND l.itemid IN (SELECT itemid FROM nw_hosp.d_labitems WHERE lower(label) LIKE '%inr%' OR lower(label) LIKE '%ptt%'
       OR lower(label) LIKE '%platelet%' OR lower(label) LIKE '%hemoglobin%' OR lower(label) LIKE '%fibrinogen%')""").fetchdf()
lab['charttime'] = pd.to_datetime(lab['charttime'])
li = con.execute("SELECT itemid, label FROM nw_hosp.d_labitems WHERE lower(label) LIKE '%inr%' OR lower(label) LIKE '%ptt%' OR lower(label) LIKE '%platelet%' OR lower(label) LIKE '%hemoglobin%' OR lower(label) LIKE '%fibrinogen%'").fetchdf()
def lab_cls(l):
    l = (l or '').lower()
    if 'inr' in l: return 'inr'
    if 'ptt' in l: return 'ptt'
    if 'platelet' in l: return 'plt'
    if l == 'hemoglobin': return 'hgb'
    if 'fibrinogen' in l: return 'fib'
    return None
labmap = {r['itemid']: lab_cls(r['label']) for _, r in li.iterrows()}
lab['lab'] = lab['itemid'].map(labmap)
lab = lab[lab['lab'].notna() & lab['valuenum'].notna()]
admit_lab = cohort[['hadm_id','intime']].rename(columns={'intime':'anchor'}).drop_duplicates()
lab = lab.merge(admit_lab, on='hadm_id', how='left')
lab['h'] = (lab['charttime']-lab['anchor']).dt.total_seconds()/3600.0
first24 = lab[(lab['h']>=-24)&(lab['h']<=24)].sort_values('h').groupby(['hadm_id','lab']).first().reset_index()
coag = first24.pivot_table(index='hadm_id', columns='lab', values='valuenum', aggfunc='first').reset_index()
coag.columns = ['hadm_id'] + ['admit_'+c for c in coag.columns[1:]]
cohort = cohort.merge(coag, on='hadm_id', how='left')
for _c in ['admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb']:
    if _c not in cohort.columns:
        cohort[_c] = np.nan
hgb72 = lab[(lab['lab']=='hgb')&(lab['h']>=-24)&(lab['h']<=72)].groupby('hadm_id')['valuenum'].min().rename('hgb_min72').reset_index()
cohort = cohort.merge(hgb72, on='hadm_id', how='left')
cohort['hb_drop2'] = ((cohort['admit_hgb']-cohort['hgb_min72'])>=2).astype(float)
cohort['hb_drop2'] = cohort['hb_drop2'].where(cohort['admit_hgb'].notna() & cohort['hgb_min72'].notna())

# ---------- 4. Transfusions via emar + procedureevents ----------
em = con.execute("""
SELECT hadm_id, medication, charttime FROM nw_hosp.emar
WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").fetchdf()
def tx_cls(m):
    if m is None: return None
    ml = m.lower()
    if 'packed red blood' in ml or 'prbc' in ml or ('red blood cell' in ml and 'washed' not in ml): return 'rbc'
    if 'fresh frozen' in ml or 'ffp' in ml or ('plasma' in ml and 'plasma-lyte' not in ml): return 'ffp'
    if 'platelet' in ml and 'platelet count' not in ml: return 'plt'
    if 'cryoprecipitate' in ml: return 'cryo'
    return None
em['tx'] = em['medication'].apply(tx_cls)
em = em[em['tx'].notna()].copy()
em['charttime'] = pd.to_datetime(em['charttime'])
em = em.merge(admit_lab, on='hadm_id', how='left')
em['h'] = (em['charttime']-em['anchor']).dt.total_seconds()/3600.0
pe = con.execute("""
SELECT p.stay_id, p.itemid, p.starttime, d.label FROM nw_icu.procedureevents p
JOIN nw_icu.d_items d USING (itemid)
WHERE p.stay_id IN (SELECT stay_id FROM cohort) AND (lower(d.label) LIKE '%transfus%' OR lower(d.label) LIKE '%red blood%' OR lower(d.label) LIKE '%platelet%' OR lower(d.label) LIKE '%plasma%')""").fetchdf()
pe['starttime'] = pd.to_datetime(pe['starttime'])
icu_t = cohort[['stay_id','intime']].rename(columns={'intime':'anchor'})
pe = pe.merge(icu_t, on='stay_id', how='left')
pe['h'] = (pe['starttime']-pe['anchor']).dt.total_seconds()/3600.0
pe['tx'] = pe['label'].apply(lambda l: 'rbc' if ('red blood' in (l or '').lower() or 'transfus' in (l or '').lower()) else 'ffp')
tx_all = pd.concat([em[['hadm_id','tx','h']].assign(sid=None, has='emar'),
                    pe[['stay_id','tx','h']].assign(hadm_id=None, has='proc')])
# per ICU stay: emar maps via hadm; procedure via stay
em_stay = em.merge(cohort[['hadm_id','stay_id']].drop_duplicates(), on='hadm_id', how='left')
txd = pd.concat([em_stay[['stay_id','tx','h']], pe[['stay_id','tx','h']]])
txg = txd.groupby('stay_id').agg(
    tx_rbc=('tx', lambda s:int((s=='rbc').sum())),
    tx_ffp=('tx', lambda s:int((s=='ffp').sum())),
    tx_plt=('tx', lambda s:int((s=='plt').sum())),
    tx_any=('tx', lambda s:int(len(s)>0)),
    tx_first_h=('h','min')).reset_index()
cohort = cohort.merge(txg, on='stay_id', how='left')
cohort['tx_any'] = cohort['tx_any'].fillna(0).astype(int)

# ---------- 5. MV / surgery / GCS (absent) ----------
mv = con.execute("""
SELECT DISTINCT stay_id FROM nw_icu.procedureevents WHERE itemid=787541 AND stay_id IN (SELECT stay_id FROM cohort)""").fetchdf()
cohort['mv_flag'] = cohort['stay_id'].isin(mv['stay_id']).astype(int)
cr = con.execute("""
SELECT DISTINCT stay_id FROM nw_icu.procedureevents p JOIN nw_icu.d_items d USING (itemid)
WHERE p.stay_id IN (SELECT stay_id FROM cohort) AND (lower(d.label) LIKE '%crani%' OR lower(d.label) LIKE '%burr hole%' OR lower(d.label) LIKE '%ventriculost%')""").fetchdf()
cohort['neuro_surgery'] = cohort['stay_id'].isin(cr['stay_id']).astype(int)
cohort['gcs_total'] = np.nan
cohort['gcs_motor_only'] = np.nan
cohort['admit_map'] = np.nan

# ---------- 6. Thrombosis + Charlson (same mappers as MIMIC) ----------
all_dx = con.execute("""
SELECT hadm_id, icd_code, icd_version FROM nw_hosp.diagnoses_icd
WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").fetchdf()
def thrombo(icd, ver):
    icd = icd.replace('.','')
    if ver==9:
        return {'pe': icd.startswith('4151'),'mi': icd.startswith('410'),
                'is': icd.startswith('433') or icd.startswith('434'),
                'dvt': icd.startswith('4534') or icd.startswith('4538'),'arte': icd.startswith('444')}
    return {'pe': icd.startswith('I26'),'mi': icd.startswith('I21') or icd.startswith('I22'),
            'is': icd.startswith('I63'),'dvt': icd.startswith('I82'),'arte': icd.startswith('I74')}
tlist = [(r['hadm_id'],)+tuple(thrombo(r['icd_code'].upper(), int(r['icd_version'])).values()) for _, r in all_dx.iterrows()]
tdf = pd.DataFrame(tlist, columns=['hadm_id','pe','mi','is','dvt','arte'])
tcols = ['pe','mi','is','dvt','arte']
tg = tdf.groupby('hadm_id')[tcols].max().reset_index()
tg['thrombosis_any'] = tg[tcols].max(axis=1)
cohort = cohort.merge(tg, on='hadm_id', how='left')
for c in tcols+['thrombosis_any']:
    cohort[c] = cohort[c].fillna(0).astype(int)

# Charlson (reuse compact Quan mapper)
def charlson_cats(icd, ver):
    code = icd.replace('.','').upper()
    cats = set()
    if ver==9:
        if code.startswith(('410','412')): cats.add('mi')
        if code.startswith(('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428')): cats.add('chf')
        if code.startswith(('440','441','4431','4432','4433','4434','4435','4436','4437','4438','4439','4471','5571','5579','V434')): cats.add('pvd')
        if code.startswith(('36234','430','431','432','433','434','435','436','437','438')): cats.add('cvd')
        if code.startswith(('290','2941','3312')): cats.add('dementia')
        if code.startswith(('4168','4169','490','491','492','493','494','495','496','497','498','499','500','501','502','503','504','505','5064','5081','5088')): cats.add('copd')
        if code.startswith(('4465','4466','7100','7101','7102','7103','7104','7140','7141','7142','7148','725')): cats.add('rheum')
        if code.startswith(('531','532','533','534')): cats.add('pud')
        if code.startswith(('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427')): cats.add('liver_mild')
        if code.startswith(('2500','2501','2502','2503','2508','2509')): cats.add('dm_mild')
        if code.startswith(('2504','2505','2506','2507')): cats.add('dm_sev')
        if code.startswith(('3341','342','343','3440','3441','3442','3443','3444','3445','3446','3449')): cats.add('hemi')
        if code.startswith(('40301','40311','40391','40402','40403','40412','40413','40492','40493','582','5830','5831','5832','5833','5834','5835','5836','5837','585','586','5880','V420','V451','V56')): cats.add('renal')
        if code.startswith(('140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','200','201','202','203','204','205','206','207','208','2386')): cats.add('cancer')
        if code.startswith(('196','197','198','199')): cats.add('mets')
        if code.startswith(('042','043','044')): cats.add('aids')
        if code.startswith(('4560','4561','4562','5722','5723','5724','5725','5726','5727','5728')): cats.add('liver_sev')
    else:
        if code.startswith(('I21','I22','I252')): cats.add('mi')
        if code.startswith(('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290')): cats.add('chf')
        if code.startswith(('I70','I71','I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959')): cats.add('pvd')
        if code.startswith(('G45','G46','H340','I60','I61','I62','I63','I64','I65','I66','I67','I68','I69')): cats.add('cvd')
        if code.startswith(('F00','F01','F02','F03','F051','G30','G311')): cats.add('dementia')
        if code.startswith(('I278','I279','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703')): cats.add('copd')
        if code.startswith(('M05','M06','M315','M32','M33','M34','M351','M353','M360')): cats.add('rheum')
        if code.startswith(('K25','K26','K27','K28')): cats.add('pud')
        if code.startswith(('B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944')): cats.add('liver_mild')
        if code.startswith(('E100','E101','E109','E110','E111','E119','E120','E121','E129','E130','E131','E139','E140','E141','E149')): cats.add('dm_mild')
        if code.startswith(('E102','E103','E104','E105','E106','E107','E108','E112','E113','E114','E115','E116','E117','E118','E122','E123','E124','E125','E126','E127','E128','E132','E133','E134','E135','E136','E137','E138','E142','E143','E144','E145','E146','E147','E148')): cats.add('dm_sev')
        if code.startswith(('G041','G114','G801','G802','G81','G82','G830','G831','G832','G833','G834','G839')): cats.add('hemi')
        if code.startswith(('I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N18','N19','N250','Z490','Z491','Z492','Z940','Z992')): cats.add('renal')
        if code.startswith(('C00','C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72','C73','C74','C75','C76','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','C97')): cats.add('cancer')
        if code.startswith(('C77','C78','C79','C80')): cats.add('mets')
        if code.startswith(('B20','B21','B22','B24')): cats.add('aids')
        if code.startswith(('I850','I859','I864','I982','K704','K711','K721','K729','K765','K766','K767')): cats.add('liver_sev')
    return cats
weights = {'mi':1,'chf':1,'pvd':1,'cvd':1,'dementia':1,'copd':1,'rheum':1,'pud':1,'liver_mild':1,'dm_mild':1,
           'dm_sev':2,'hemi':2,'renal':2,'cancer':2,'liver_sev':3,'mets':6,'aids':6}
cat_rows = []
for _, r in all_dx.iterrows():
    cats = charlson_cats(r['icd_code'], int(r['icd_version']))
    cat_rows.append([r['hadm_id']] + [1.0 if c in cats else 0.0 for c in weights.keys()])
cat_df = pd.DataFrame(cat_rows, columns=['hadm_id']+list(weights.keys())).astype({'hadm_id':'Int64'})
ch = cat_df.groupby('hadm_id')[list(weights.keys())].max().reset_index()
ch['charlson'] = sum(ch[c]*w for c,w in weights.items())
cohort = cohort.merge(ch[['hadm_id','charlson']], on='hadm_id', how='left')
cohort['charlson_age'] = cohort['charlson'] + np.where(cohort['anchor_age']>=40, (cohort['anchor_age']-40)//10, 0)

# ---------- 7. Outcomes & tidy ----------
cohort['hosp_death'] = ((cohort['deathtime'].notna()) | (cohort['discharge_location']=='EXPIRED') | (cohort['hospital_expire_flag']==1)).astype(int)
cohort['db'] = 'nwicu'
cohort = cohort.rename(columns={'anchor_age':'age','gender':'sex'})
keep = ['db','subject_id','hadm_id','stay_id','age','sex','icu_los_days','hosp_los_days','s06_subtypes',
        'gcs_total','gcs_motor_only','admit_map','admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb','hgb_min72','hb_drop2',
        'charlson','charlson_age','neuro_surgery','mv_flag',
        'baseline_at','ever_at','reversal','reversal_h_first','resume_cls','resume_h','last_anticoag_stop_h',
        'tx_rbc','tx_ffp','tx_plt','tx_any','tx_first_h',
        'pe','mi','is','dvt','arte','thrombosis_any',
        'hosp_death','intime','outtime','admittime']
cohort[keep].to_csv(OUT, index=False)
print('saved', OUT, len(cohort))
print('baseline_at:', cohort['baseline_at'].value_counts(dropna=False).to_dict())
print('reversal:', cohort['reversal'].value_counts(dropna=False).head(8).to_dict())
print('hosp_death:', cohort['hosp_death'].sum())
