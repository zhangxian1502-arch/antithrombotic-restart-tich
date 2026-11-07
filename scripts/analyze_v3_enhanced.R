# Enhanced analyses for antithrombotic-associated tICH paper (pool v3)
# RCS (restart time), E-value, PSM/IPW, Landmark
suppressMessages({library(rms); library(survival); library(MatchIt)})

pool <- read.csv('F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/pool_t3_v3.csv',
                 stringsAsFactors = FALSE, na.strings = c('NA','','NaN'))
cat('POOL n =', nrow(pool), '\n')

# AT cohort: baseline antithrombotic exposure
pool$at <- !is.na(pool$baseline_at)
at <- pool[pool$at, ]
cat('AT n =', nrow(at), '\n')

# derived flags
at$reversal_any <- !is.na(at$reversal)
at$restarted <- !is.na(at$resume_cls)
at$male <- as.integer(grepl('^[Mm]', at$sex))
at$age <- as.numeric(at$age)

# --- covariate prep (median imputation + missingness indicator, ext-tier style) ---
ext_vars <- c('age','male','gcs_any','charlson_age','hr1','sbp1','spo2_1','temp_c1',
              'l_wbc','l_cr','l_glu','l_na','l_k','l_lac','l_alb','l_tbili',
              'vaso_flag','mv_flag','neuro_surgery','admit_to_icu_h','cc_cvd','cc_renal','cc_dm','cc_cancer')
for (v in ext_vars) if (v %in% names(at)) at[[v]] <- as.numeric(at[[v]])
for (v in c('gcs_any','charlson_age')) if (v %in% names(at)) {
  at[[paste0(v,'_miss')]] <- as.integer(is.na(at[[v]]))
  at[[v]][is.na(at[[v]])] <- median(at[[v]], na.rm = TRUE)
}
# DB indicators
for (db in c('eicu','nwicu','sicdb','umcdb')) at[[paste0('db_',db)]] <- as.integer(at$db == db)

# ------------------------------------------------------------------ #
# 1) RCS: restart time (continuous, hours) vs thrombosis / death     #
# ------------------------------------------------------------------ #
rst <- at[!is.na(at$resume_h), ]
rst$thrombosis <- as.integer(rst$thrombosis_any == 1)
rst$death <- as.integer(rst$hosp_death == 1)
cat('restart n =', nrow(rst), ' thrombosis events =', sum(rst$thrombosis, na.rm=TRUE),
    ' death events =', sum(rst$death, na.rm=TRUE), '\n')

# only sites with thrombosis coding
rst_th <- rst[!is.na(rst$thrombosis_any), ]
rst_th$resume_h <- as.numeric(rst_th$resume_h)

dd <- datadist(rst_th[, c('resume_h','thrombosis','age','male','gcs_any','charlson_age')]); options(datadist = 'dd')
rcs4 <- NULL
tryCatch({
  m_rcs <- lrm(thrombosis ~ rcs(resume_h, 4) + age + male + gcs_any + charlson_age, data = rst_th, x = TRUE, y = TRUE)
  # prediction curve (linear predictor scale, then exp)
  newd <- expand.grid(resume_h = seq(48, quantile(rst_th$resume_h, .95), length.out = 80),
                      age = median(rst_th$age, na.rm=TRUE), male = 0,
                      gcs_any = median(rst_th$gcs_any, na.rm=TRUE), charlson_age = median(rst_th$charlson_age, na.rm=TRUE))
  pred <- predict(m_rcs, newd, se.fit = TRUE, type = 'lp')
  rcs4 <- data.frame(resume_h = newd$resume_h, or = exp(pred$linear.predictors),
                     lo = exp(pred$linear.predictors - 1.96*pred$se.fit),
                     hi = exp(pred$linear.predictors + 1.96*pred$se.fit))
  an <- anova(m_rcs)
  rcs4$p_nonlin <- an['resume_h','P']  # overall
  cat('RCS thrombosis: overall p =', an['resume_h','P'], '\n')
  write.csv(rcs4, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/rcs_thrombosis_curve.csv', row.names = FALSE)
}, error = function(e) cat('RCS thrombosis failed:', conditionMessage(e), '\n'))

# death RCS (few events; exploratory)
rst_de <- rst_th
tryCatch({
  m_rcs_d <- lrm(death ~ rcs(resume_h, 4) + age + male + gcs_any, data = rst_de, x = TRUE, y = TRUE)
  newd2 <- expand.grid(resume_h = seq(48, quantile(rst_de$resume_h, .95), length.out = 80),
                       age = median(rst_de$age, na.rm=TRUE), male = 0, gcs_any = median(rst_de$gcs_any, na.rm=TRUE))
  pred2 <- predict(m_rcs_d, newd2, se.fit = TRUE, type = 'lp')
  rcs_d <- data.frame(resume_h = newd2$resume_h, or = exp(pred2$linear.predictors),
                      lo = exp(pred2$linear.predictors - 1.96*pred2$se.fit),
                      hi = exp(pred2$linear.predictors + 1.96*pred2$se.fit))
  write.csv(rcs_d, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/rcs_death_curve.csv', row.names = FALSE)
  cat('RCS death done\n')
}, error = function(e) cat('RCS death failed:', conditionMessage(e), '\n'))

# ------------------------------------------------------------------ #
# 2) E-value (VanderWeele & Ding 2017)                               #
# ------------------------------------------------------------------ #
evalue <- function(or, lo, hi) {
  calc <- function(r) { if (r < 1) r <- 1/r; r + sqrt(r*(r-1)) }
  pt <- calc(or)
  # CI on risk scale: use bound closest to null
  bound <- ifelse(or >= 1, lo, hi)
  ci <- calc(bound)
  c(evalue_point = pt, evalue_ci = ci)
}
evalues <- data.frame(
  association = c('Restart -> death','Restart -> thrombosis','Reversal -> transfusion',
                  'Reversal -> thrombosis','Reversal -> death','Reversal -> Hb drop >=2'),
  OR = c(0.397, 3.081, 6.938, 2.125, 0.734, 1.337),
  lo = c(0.161, 1.591, 3.514, 0.965, 0.300, 0.654),
  hi = c(0.980, 5.965, 13.698, 4.679, 1.794, 2.735))
ev <- t(apply(evalues[, c('OR','lo','hi')], 1, function(r) evalue(r[1], r[2], r[3])))
evalues <- cbind(evalues, ev)
write.csv(evalues, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/evalue_results.csv', row.names = FALSE)
print(evalues)

# ------------------------------------------------------------------ #
# 3) PSM / IPW: restart vs no restart (within AT cohort)             #
# ------------------------------------------------------------------ #
ps_vars <- c('age','male','gcs_any','gcs_any_miss','charlson_age','charlson_age_miss',
             'hr1','sbp1','spo2_1','vaso_flag','mv_flag','neuro_surgery','admit_to_icu_h',
             'db_eicu','db_nwicu','db_sicdb','db_umcdb')
ps_vars <- ps_vars[ps_vars %in% names(at)]
for (v in c('hr1','sbp1','spo2_1','admit_to_icu_h','age','vaso_flag','mv_flag','neuro_surgery')) {
  at[[paste0(v,'_miss')]] <- as.integer(is.na(at[[v]]))
  at[[v]][is.na(at[[v]])] <- median(at[[v]], na.rm = TRUE)
}
at$restarted <- as.integer(at$restarted)
ps_formula <- as.formula(paste('restarted ~', paste(ps_vars, collapse = ' + ')))

# PSM: 1:1 nearest neighbour, caliper 0.2 SD
set.seed(20260919)
m_out <- tryCatch(matchit(ps_formula, data = at, method = 'nearest', ratio = 1, caliper = 0.2),
                  error = function(e) {cat('matchit error:', conditionMessage(e), '\n'); NULL})
if (!is.null(m_out)) {
  md <- match.data(m_out)
  cat('PSM matched n =', nrow(md), ' treated =', sum(md$restarted), ' control =', sum(1-md$restarted), '\n')
  psm_res <- data.frame(
    outcome = c('death','thrombosis','transfusion'),
    n = NA, OR = NA, lo = NA, hi = NA, p = NA, stringsAsFactors = FALSE)
  # death
  if (sum(md$hosp_death==1, na.rm=TRUE) >= 5) {
    f1 <- clogit(hosp_death ~ restarted + strata(subclass), data = md)
    ci1 <- exp(confint(f1)); psm_res[1,] <- c('death', sum(!is.na(md$hosp_death)), exp(coef(f1)), ci1[1], ci1[2], summary(f1)$coefficients[5])
  }
  # thrombosis (sites with data)
  mdt <- md[!is.na(md$thrombosis_any), ]
  if (nrow(mdt) >= 10 && sum(mdt$thrombosis_any==1) >= 5) {
    f2 <- clogit(thrombosis_any ~ restarted + strata(subclass), data = mdt)
    ci2 <- exp(confint(f2)); psm_res[2,] <- c('thrombosis', nrow(mdt), exp(coef(f2)), ci2[1], ci2[2], summary(f2)$coefficients[5])
  }
  # transfusion
  if (sum(md$tx_any==1) >= 5) {
    f3 <- clogit(tx_any ~ restarted + strata(subclass), data = md)
    ci3 <- exp(confint(f3)); psm_res[3,] <- c('transfusion', nrow(md), exp(coef(f3)), ci3[1], ci3[2], summary(f3)$coefficients[5])
  }
  # balance: standardized mean differences
  smd <- function(v) {
    d <- md[[v]]; tr <- d[md$restarted==1]; co <- d[md$restarted==0]
    (mean(tr,na.rm=TRUE)-mean(co,na.rm=TRUE))/sqrt((var(tr,na.rm=TRUE)+var(co,na.rm=TRUE))/2)
  }
  balance <- sapply(c('age','male','gcs_any','charlson_age','vaso_flag','mv_flag','neuro_surgery','sbp1','hr1'),
                    function(v) if (v %in% names(md)) smd(v) else NA)
  cat('PSM balance SMD:\n'); print(round(balance,3))
  write.csv(psm_res, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/psm_results.csv', row.names = FALSE)
  write.csv(data.frame(var=names(balance), smd=as.numeric(balance)),
            'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/psm_balance.csv', row.names = FALSE)
}

# IPW: stabilized weights (ATT)
w_out <- tryCatch(matchit(ps_formula, data = at, method = 'nearest', ratio = 1, caliper = 0.2, distance = 'glm'),
                  error = function(e) NULL)
# manual stabilized IPW via glm
ps_model <- glm(ps_formula, data = at, family = binomial)
at$ps <- predict(ps_model, type = 'response')
at$ipw <- ifelse(at$restarted == 1, 1, at$ps/(1-at$ps))   # ATT weights: treated=1, control=ps/(1-ps)
# stabilized
p_tr <- mean(at$restarted)
at$ipw_stab <- ifelse(at$restarted == 1, p_tr, p_tr*at$ps/(1-at$ps))

ipw_res <- data.frame(outcome = c('death','thrombosis','transfusion'), n=NA, OR=NA, lo=NA, hi=NA, p=NA, stringsAsFactors = FALSE)
if (sum(at$hosp_death==1) >= 5) {
  g1 <- glm(hosp_death ~ restarted + age + male + gcs_any + charlson_age, data = at, family = binomial, weights = ipw_stab)
  ci1 <- exp(confint(g1)); ipw_res[1,] <- c('death', sum(!is.na(at$hosp_death)), exp(coef(g1)[2]), ci1[2,1], ci1[2,2], summary(g1)$coefficients[2,4])
}
att <- at[!is.na(at$thrombosis_any), ]
if (nrow(att)>=10 && sum(att$thrombosis_any==1)>=5) {
  g2 <- glm(thrombosis_any ~ restarted + age + male + gcs_any + charlson_age, data = att, family = binomial, weights = ipw_stab)
  ci2 <- exp(confint(g2)); ipw_res[2,] <- c('thrombosis', nrow(att), exp(coef(g2)[2]), ci2[2,1], ci2[2,2], summary(g2)$coefficients[2,4])
}
if (sum(at$tx_any==1) >= 5) {
  g3 <- glm(tx_any ~ restarted + age + male + gcs_any + charlson_age, data = at, family = binomial, weights = ipw_stab)
  ci3 <- exp(confint(g3)); ipw_res[3,] <- c('transfusion', nrow(at), exp(coef(g3)[2]), ci3[2,1], ci3[2,2], summary(g3)$coefficients[2,4])
}
write.csv(ipw_res, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/ipw_results.csv', row.names = FALSE)
print('IPW results'); print(ipw_res)

# ------------------------------------------------------------------ #
# 4) Landmark analysis (72 h and 7 days)                             #
# ------------------------------------------------------------------ #
landmark <- function(land_h) {
  d <- at[!is.na(at$icu_los_days), ]
  d$icu_h <- d$icu_los_days * 24
  # survived to landmark: death either not recorded or after landmark (icu stay >= landmark)
  d$alive_lm <- (is.na(d$hosp_death) | d$hosp_death == 0) | (d$icu_h >= land_h)
  d <- d[d$alive_lm, ]
  # exposure: restart before landmark
  d$restart_lm <- as.integer(!is.na(d$resume_h) & d$resume_h <= land_h)
  out <- data.frame(landmark_h = land_h, n = nrow(d), restart_lm_n = sum(d$restart_lm))
  if (sum(d$hosp_death==1) >= 5 & sum(d$restart_lm) >= 5) {
    g <- glm(hosp_death ~ restart_lm + age + male + gcs_any + charlson_age, data = d, family = binomial)
    out$death_OR <- exp(coef(g)[2]); out$death_lo <- exp(confint(g))[2,1]; out$death_hi <- exp(confint(g))[2,2]; out$death_p <- summary(g)$coefficients[2,4]
  }
  dt <- d[!is.na(d$thrombosis_any), ]
  if (nrow(dt) >= 10 & sum(dt$thrombosis_any==1) >= 5 & sum(dt$restart_lm) >= 5) {
    g2 <- glm(thrombosis_any ~ restart_lm + age + male + gcs_any + charlson_age, data = dt, family = binomial)
    out$throm_OR <- exp(coef(g2)[2]); out$throm_lo <- exp(confint(g2))[2,1]; out$throm_hi <- exp(confint(g2))[2,2]; out$throm_p <- summary(g2)$coefficients[2,4]
  }
  out
}
lm72 <- landmark(72); lm168 <- landmark(168)
lm_res <- rbind(data.frame(lm72), data.frame(lm168))
write.csv(lm_res, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/landmark_results.csv', row.names = FALSE)
print('Landmark results'); print(lm_res)

cat('\nALL ENHANCED ANALYSES DONE\n')
