# Figure 4: main forest plot re-drawn in R (ggplot2 + Paul Tol + font ladder)
# v3 numbers (from analysis_results_v3.json, verified) - data unchanged
suppressMessages({library(ggplot2); library(ragg)})

tol7 <- c('#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB')
REV <- tol7[5]   # #EE6677 red  = reversal
RST <- tol7[1]   # #4477AA blue = restart

base_theme <- theme_minimal(base_size = 8) +
  theme(text = element_text(family = 'Times New Roman', colour = 'black'),
        plot.title = element_text(size = 9, face = 'bold', hjust = 0.5),
        axis.title = element_text(size = 8.5),
        axis.text = element_text(size = 8, colour = 'black'),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        plot.caption = element_text(size = 7, hjust = 0),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = 'grey90', linewidth = 0.25),
        plot.margin = margin(4, 4, 4, 4, 'mm'))
theme_set(base_theme)

out <- 'F:/豆包文件夹/新文章16/投稿包_Clinical_Neurology_and_Neurosurgery'

# ---- data (rows top -> bottom) ----
# group label rows carry or = NA (no point / no line)
dat <- data.frame(
  y      = c(7, 6, 5, 4, 3, 2, 1, 0),
  lab    = c('Antithrombotic reversal (n = 65 vs 478)', 'In-hospital death',
             'Hb decline ≥ 2 g/dl (72 h)', 'Any transfusion', 'Thrombotic event',
             'Antithrombotic restart (n = 146 vs 397)', 'In-hospital death',
             'Thrombotic event'),
  or     = c(NA, 0.734, 1.337, 6.938, 2.125, NA, 0.397, 3.081),
  lo     = c(NA, 0.300, 0.654, 3.514, 0.965, NA, 0.161, 1.591),
  hi     = c(NA, 1.794, 2.735, 13.698, 4.679, NA, 0.980, 5.965),
  p      = c(NA, 'P=0.497', 'P=0.426', 'P<0.001', 'P=0.061',
             NA, 'P=0.045', 'P<0.001'),
  col    = c(REV, REV, REV, REV, REV, RST, RST, RST),
  is_grp = c(TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE),
  stringsAsFactors = FALSE
)
dat$ci_txt[!dat$is_grp] <- sprintf('%.2f (%.2f\u2013%.2f), %s',
                                   dat$or[!dat$is_grp], dat$lo[!dat$is_grp],
                                   dat$hi[!dat$is_grp], dat$p[!dat$is_grp])

p <- ggplot(dat, aes(x = or, y = y, xmin = lo, xmax = hi)) +
  geom_vline(xintercept = 1, linetype = 2, colour = '#BBBBBB', linewidth = 0.4) +
  # group label rows: bold italic text on the left, no error bar / point
  geom_text(data = dat[dat$is_grp, ], aes(x = 0.12, label = lab),
            hjust = 0, size = 2.8, fontface = 'bold', family = 'Times New Roman', colour = 'black') +
  geom_errorbarh(data = dat[!dat$is_grp, ], width = 0.25, linewidth = 0.7,
                 colour = dat$col[!dat$is_grp]) +
  geom_point(data = dat[!dat$is_grp, ], shape = 21, size = 2.6, fill = 'white',
             stroke = 1.1, colour = dat$col[!dat$is_grp]) +
  geom_text(data = dat[!dat$is_grp, ], aes(x = 60, label = ci_txt),
            hjust = 0, size = 3, family = 'Times New Roman', colour = 'black') +
  scale_x_log10(name = 'Adjusted odds ratio (95% CI), log scale',
                breaks = c(0.1, 0.5, 1, 2, 4, 8, 16, 32),
                limits = c(0.1, 5000)) +
  scale_y_continuous(breaks = dat$y[!dat$is_grp],
                     labels = dat$lab[!dat$is_grp],
                     limits = c(-0.5, 7.5)) +
  coord_cartesian(xlim = c(0.1, 5000), ylim = c(-0.5, 7.5)) +
  ggtitle('Reversal and restart versus clinical outcomes (Firth penalised logistic regression, n = 543)') +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_text(hjust = 1, face = 'plain'),
        panel.grid.major.y = element_blank())

ggsave(file.path(out, 'Figure3_forest.pdf'), p, width = 180, height = 110,
       units = 'mm', device = cairo_pdf)
ggsave(file.path(out, 'Figure3_forest.png'), p, width = 180, height = 110,
       units = 'mm', dpi = 300, device = agg_png)
ggsave(file.path(out, 'Figure3_forest.svg'), p, width = 180, height = 110,
       units = 'mm')
cat('Figure3 (R) saved\n')
