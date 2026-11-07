# Enhanced-analysis figures v2 (R + ggplot2 + Paul Tol palette + font-size ladder)
suppressMessages({library(ggplot2); library(ragg); library(patchwork)})

# ---- Paul Tol discrete palette ----
tol7 <- c('#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB')

base_theme <- theme_minimal(base_size = 8) +
  theme(text = element_text(family = 'Times New Roman', colour = 'black'),
        plot.title = element_text(size = 9, face = 'bold', hjust = 0.5),
        axis.title = element_text(size = 8.5),
        axis.text = element_text(size = 8, colour = 'black'),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        strip.text = element_text(size = 8.5, face = 'bold'),
        plot.caption = element_text(size = 7, hjust = 0),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = 'grey90', linewidth = 0.25),
        plot.margin = margin(4, 4, 4, 4, 'mm'))
theme_set(base_theme)

outdir <- 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/figures_enhanced'
dir.create(outdir, showWarnings = FALSE)

# ==================================================================
# Figure 5: RCS dose-response (already working)
# ==================================================================
rcs <- read.csv('F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/rcs_thrombosis_curve.csv')
ref <- rcs$or[which.min(abs(rcs$resume_h - 48))]
rcs$or_r <- rcs$or / ref; rcs$lo_r <- rcs$lo / ref; rcs$hi_r <- rcs$hi / ref
p_nonlin <- unique(round(rcs$p_nonlin, 4))

g1 <- ggplot(rcs, aes(x = resume_h, y = or_r)) +
  geom_hline(yintercept = 1, linetype = 2, colour = '#BBBBBB', linewidth = 0.4) +
  geom_ribbon(aes(ymin = lo_r, ymax = hi_r), fill = '#4477AA', alpha = 0.25) +
  geom_line(colour = '#4477AA', linewidth = 0.9) +
  scale_x_continuous(name = 'Time to antithrombotic restart after ICU admission (h)',
                     breaks = seq(48, 320, 48), limits = c(48, 320)) +
  scale_y_continuous(name = 'Adjusted OR for in-hospital thrombosis',
                     trans = 'log2', breaks = c(0.5, 1, 2, 4, 8)) +
  coord_cartesian(ylim = c(0.4, 9)) +
  annotate('text', x = 150, y = 8.2, label = paste0('Non-linearity P = ', p_nonlin),
           size = 2.6, family = 'Times New Roman', colour = 'black') +
  ggtitle('Dose\u2013response relationship between restart timing and thrombosis')
ggsave(file.path(outdir, 'Figure4_rcs_restart_thrombosis.pdf'), g1, width = 180, height = 110, units = 'mm', device = cairo_pdf)
ggsave(file.path(outdir, 'Figure4_rcs_restart_thrombosis.png'), g1, width = 180, height = 110, units = 'mm', dpi = 300, device = agg_png)
cat('RCS figure saved\n')

# ==================================================================
# Figure 7: database-stratified forest plots (two panels, patchwork)
# ==================================================================
pool <- read.csv('F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/pool_t3_v3.csv',
                 stringsAsFactors = FALSE, na.strings = c('NA','','NaN'))
pool$restarted <- !is.na(pool$resume_cls)
at <- pool[!is.na(pool$baseline_at), ]
at$thrombosis <- as.integer(at$thrombosis_any == 1)
at$restarted <- as.integer(at$restarted)

rest_perdb <- data.frame()
for (db in c('mimic','eicu','nwicu')) {
  d <- at[at$db == db & !is.na(at$thrombosis_any), ]
  if (sum(d$restarted) >= 5 && sum(d$thrombosis == 1) >= 5) {
    m <- glm(thrombosis ~ restarted, data = d, family = binomial)
    ci <- exp(confint(m))
    rest_perdb <- rbind(rest_perdb, data.frame(db = db, n = nrow(d),
                                               OR = exp(coef(m)[2]), lo = ci[2,1], hi = ci[2,2],
                                               p = summary(m)$coefficients[2,4]))
  }
}
cat('restart perdb:\n'); print(rest_perdb)

forest_plot <- function(dat, pooled_row, title) {
  # dat: rows for databases (db, n, OR, lo, hi); pooled_row: list(n, or, lo, hi)
  db_labels <- c(mimic = 'MIMIC-IV', eicu = 'eICU-CRD', nwicu = 'NWICU',
                 sicdb = 'SICdb', umcdb = 'AmsterdamUMCdb')
  # display order: top -> bottom = databases then pooled
  labs_top <- c(sprintf('%s (n = %d)', db_labels[dat$db], dat$n), 'Pooled (Firth)')
  ors  <- c(dat$OR, pooled_row$or)
  los  <- c(dat$lo, pooled_row$lo)
  his  <- c(dat$hi, pooled_row$hi)
  isp  <- c(rep(FALSE, nrow(dat)), TRUE)
  cols <- c(tol7[seq_len(nrow(dat))], '#EE6677')
  nrow_all <- length(ors)
  yvals <- rev(seq_len(nrow_all))            # top row has largest y
  d <- data.frame(lab = labs_top, or = ors, lo = los, hi = his,
                  is_pooled = isp, col = cols, y = yvals)
  d$ci_txt <- sprintf('%.2f (%.2f\u2013%.2f)', d$or, d$lo, d$hi)
  ggplot(d, aes(x = or, y = y, xmin = lo, xmax = hi)) +
    geom_vline(xintercept = 1, linetype = 2, colour = '#BBBBBB', linewidth = 0.4) +
    geom_errorbarh(width = 0.2, linewidth = 0.7, colour = d$col) +
    geom_point(aes(shape = is_pooled), size = 2.4, colour = d$col) +
    geom_text(aes(x = 80, label = ci_txt), hjust = 0, size = 3,
              family = 'Times New Roman', colour = 'black', show.legend = FALSE) +
    scale_x_log10(name = 'Adjusted odds ratio (log scale)',
                  breaks = c(0.1, 0.5, 1, 2, 4, 8, 16, 32, 64), limits = c(0.1, 300)) +
    scale_y_continuous(breaks = d$y, labels = d$lab) +
    scale_shape_manual(values = c('TRUE' = 18, 'FALSE' = 16), guide = 'none') +
    coord_cartesian(xlim = c(0.1, 300), ylim = c(0.5, max(d$y) + 0.5)) +
    ggtitle(title) +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_text(hjust = 1),
          plot.title = element_text(size = 9, face = 'bold', hjust = 0))
}

# restart -> thrombosis panel
p_rest <- forest_plot(rest_perdb, list(n = 502, or = 3.081, lo = 1.591, hi = 5.965),
                      '(a) Restart vs no restart \u2013 in-hospital thrombosis')
# reversal -> transfusion panel
rev_tx <- data.frame(db = c('mimic','eicu'), n = c(149, 319),
                     OR = c(7.779, 9.894), lo = c(1.329, 3.846), hi = c(45.52, 25.45), p = c(0.023, 2e-06))
p_rev <- forest_plot(rev_tx, list(n = 543, or = 6.938, lo = 3.514, hi = 13.698),
                     '(b) Reversal vs no reversal \u2013 transfusion requirement')

legend_row <- ggplot() + annotate('point', x = 1, y = 1, shape = 18, size = 3, colour = '#EE6677') +
  annotate('text', x = 1.02, y = 1, label = 'Pooled (Firth)', hjust = 0, size = 2.6, family = 'Times New Roman') +
  annotate('point', x = 1.25, y = 1, shape = 16, size = 3, colour = '#4477AA') +
  annotate('text', x = 1.27, y = 1, label = 'Per-database estimate', hjust = 0, size = 2.6, family = 'Times New Roman') +
  coord_cartesian(xlim = c(0.95, 1.75)) + theme_void()

g2 <- (p_rest / p_rev / legend_row) + plot_layout(heights = c(4, 3, 0.8))
ggsave(file.path(outdir, 'Figure5_forest_databases.pdf'), g2, width = 180, height = 130, units = 'mm', device = cairo_pdf)
ggsave(file.path(outdir, 'Figure5_forest_databases.png'), g2, width = 180, height = 130, units = 'mm', dpi = 300, device = agg_png)
cat('Forest figure saved\n')

write.csv(rest_perdb, 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/restart_thrombosis_perdb.csv', row.names = FALSE)
cat('ALL FIGURES DONE\n')
