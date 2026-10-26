# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(latex2exp)
library(cowplot)


RS = read.csv("results/baseline_p.csv", header = FALSE) %>%
  `colnames<-`(c("p", "l", "method", "beta_diff", "subset", "recover", "len", "time", "true_s"))


method_levels <- c("MIP-CD", "EILLS-s", "ICP-s", "Causal Dantzig",
                   "BIP-s", "NegDRO", "CoCo", "IRM")

cols <- c(
  "MIP-CD"="#FF7F0E", "EILLS-s"="#D62728", "ICP-s"="#2CA02C",
  "Causal Dantzig"="#8C564B", "BIP-s"="#17BECF",
  "NegDRO"="#1F77B4", "CoCo"="#9467BD", "IRM"="#E377C2"
)

p_grid <- seq(10, 100, 10)
pointsize <- 2.2
linewidth <- .7
se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) > 1) sd(x)/sqrt(length(x)) else 0
}

# Data
df <- RS %>%
  mutate(
    p = as.numeric(p), 
    method = recode(method, "MIP-cd"="MIP-CD", "EILLS"="EILLS-s", "ICP"="ICP-s"),
    method = factor(method, levels=method_levels),
    beta_diff = as.numeric(beta_diff), time=as.numeric(time),
    recover = case_when(
      tolower(as.character(recover))=="true" ~ 1,
      tolower(as.character(recover))=="false" ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  group_by(p, method) %>%
  summarise(
    beta_diff_mean = mean(beta_diff, na.rm=TRUE), beta_diff_se=se(beta_diff),
    recover_mean = mean(recover, na.rm=TRUE), recover_se=se(recover),
    time_mean = mean(time, na.rm=TRUE), time_se=se(time),
    .groups = "drop"
  ) %>%
  mutate(
    across(
      c(beta_diff_mean, beta_diff_se,
        recover_mean, recover_se,
        time_mean, time_se),
      ~ ifelse(method != "ICP-s" & is.nan(.x), 0, .x)
    )) %>%
  mutate(
    beta_lo=beta_diff_mean-beta_diff_se, 
    beta_hi=beta_diff_mean+beta_diff_se,
    rec_lo=pmax(0, recover_mean-recover_se), 
    rec_hi=pmin(1, recover_mean+recover_se),
    time_lo=pmax(time_mean-time_se, 1e-12), 
    time_hi=time_mean+time_se
  )

col_scale <- scale_color_manual(NULL, values=cols, limits=method_levels, drop=FALSE)
fill_scale <- scale_fill_manual(NULL, values=cols, limits=method_levels, drop=FALSE)

base_theme <- theme_bw(base_size=10) +
  theme(
    axis.title=element_text(size=9), axis.text=element_text(size=9),
    panel.grid.major=element_line(color="grey88", linewidth=.35),
    panel.grid.minor=element_blank(),
    panel.border=element_rect(color="grey30", linewidth=.5),
    plot.margin=margin(5, 8, 5, 8),
    legend.position="none"
  )

# Draw: other methods -> EILLS -> MIP-cd
add_layers <- function(d, y, lo, hi) {
  ds <- list(
    d %>% filter(!method %in% c("IRM", "CoCo", "EILLS-s", "MIP-CD")),
    d %>% filter(method=="IRM"),
    d %>% filter(method=="CoCo"),
    d %>% filter(method=="EILLS-s"),
    d %>% filter(method=="MIP-CD")
  )
  
  c(
    lapply(ds, \(z)
           geom_ribbon(
             data=z, aes(x=p, ymin=.data[[lo]], ymax=.data[[hi]], fill=method),
             alpha=0.3, color=NA
           )),
    unlist(lapply(ds, \(z) list(
      geom_line(
        data=z, aes(x=p, y=.data[[y]], color=method, group=method),
        linewidth=linewidth
      ),
      geom_point(
        data=z, aes(x=p, y=.data[[y]], fill=method),
        shape=21, size=pointsize, stroke=.6, color="white"
      )
    )), recursive=FALSE)
  )
}

d1 <- df %>% filter(is.finite(beta_diff_mean))
d2 <- df %>% filter(is.finite(recover_mean))
d3 <- df %>% filter(is.finite(time_mean), time_mean > 0)

p1 <- ggplot() +
  add_layers(d1, "beta_diff_mean", "beta_lo", "beta_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=p_grid) +
  labs(
    x = expression("Dimension"~italic(p)),
    y = TeX("Coefficient error $\\|\\widehat{\\beta}-\\beta^*\\|_2$")
  ) +
  base_theme

p2 <- ggplot() +
  add_layers(d2, "recover_mean", "rec_lo", "rec_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=p_grid) +
  scale_y_continuous(limits=c(0, 1)) +
  labs(
    x = expression("Dimension"~italic(p)),
    y = "Proportion of perfect recovery"
  ) +
  base_theme

p3 <- ggplot() +
  add_layers(d3, "time_mean", "time_lo", "time_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=p_grid) +
  scale_y_log10(
    breaks=10^(-2:2),
    labels=scales::label_log(base=10)
  ) +
  labs(x=expression("Dimension"~italic(p)), y="Runtime (sec)") +
  base_theme

# Common legend with white point borders
legend_df <- expand_grid(
  method=factor(method_levels, levels=method_levels),
  x=c(1, 2)
) %>% mutate(y=1)

legend <- get_legend(
  ggplot(legend_df, aes(x, y, group=method)) +
    geom_line(aes(color=method), linewidth=linewidth) +
    geom_point(
      aes(fill=method),
      shape=21, size=pointsize, stroke=.6, color="white"
    ) +
    col_scale + fill_scale +
    theme_void() +
    theme(
      legend.position="bottom",
      legend.text=element_text(size=9),
      legend.key.width=grid::unit(1.1, "cm")
    ) +
    guides(
      color=guide_legend(nrow=1, byrow=TRUE),
      fill=guide_legend(nrow=1, byrow=TRUE)
    )
)

panels <- plot_grid(p1, p2, p3, nrow=1, align="h")
final_plot <- plot_grid(panels, legend, ncol=1, rel_heights=c(1, .1))

final_plot

print(final_plot)
dev.off()
