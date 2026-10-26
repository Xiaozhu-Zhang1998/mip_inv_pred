# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(patchwork)
library(tidyverse)
library(cowplot)

# Read results
df <- read_csv(
  "results/RS.csv",
  show_col_types = FALSE, 
  col_names = FALSE
) %>%
  `colnames<-`(c("E", "trial", "method", "tp", "fp", "fn", 'size', 'time')) %>%
  mutate(
    across(c(E, trial, tp, fp, fn, size), as.numeric),
    F1 = ifelse(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0),
    FD = fp
  ) %>%
  mutate(method = recode(method, "MIP" = "MIP-CD"))


method_levels <- c("MIP-CD", "EILLS", "ICP", "Causal Dantzig",
                   "BIP", "MIPtest", "FAIR")

cols <- c(
  "MIP-CD"="#FF7F0E", "EILLS"="#D62728", "ICP"="#2CA02C",
  "Causal Dantzig"="#8C564B", "BIP"="#17BECF",
  "MIPtest"="#9467BD", "FAIR"="#1F77B4"
)

pointsize <- 2.2
linewidth <- .7
se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) > 1) sd(x)/sqrt(length(x)) else 0
}

# Data
df_sum <- df %>%
  mutate(
    E=as.numeric(E),
    method=factor(method, levels=method_levels),
    F1=as.numeric(F1), FD=as.numeric(FD), time=as.numeric(time)
  ) %>%
  group_by(E, method) %>%
  summarise(
    F1_se=se(F1), F1_mean=mean(F1, na.rm=TRUE),
    FD_se=se(FD), FD_mean=mean(FD, na.rm=TRUE),
    time_se=se(time), time_mean=mean(time, na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(
    F1_lo=pmax(0, F1_mean-F1_se),
    F1_hi=pmin(1, F1_mean+F1_se),
    FD_lo=pmax(0, FD_mean-FD_se),
    FD_hi=FD_mean+FD_se,
    time_lo=pmax(time_mean-time_se, 1e-8),
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

# Draw: other methods -> MIPtest -> MIP
add_layers <- function(d, y, lo, hi) {
  ds <- list(
    d %>% filter(!method %in% c("MIP-CD", "MIPtest")),
    d %>% filter(method=="MIPtest"),
    d %>% filter(method=="MIP-CD")
  )
  
  c(
    lapply(ds, \(z)
           geom_ribbon(
             data=z, aes(x=E, ymin=.data[[lo]], ymax=.data[[hi]], fill=method),
             alpha=.3, color=NA
           )),
    unlist(lapply(ds, \(z) list(
      geom_line(
        data=z, aes(x=E, y=.data[[y]], color=method, group=method),
        linewidth=linewidth
      ),
      geom_point(
        data=z, aes(x=E, y=.data[[y]], fill=method),
        shape=21, size=pointsize, stroke=.6, color="white"
      )
    )), recursive=FALSE)
  )
}

d1 <- df_sum %>% filter(is.finite(F1_mean))
d2 <- df_sum %>% filter(is.finite(FD_mean))
d3 <- df_sum %>% filter(is.finite(time_mean), time_mean > 0)

p1 <- ggplot() +
  add_layers(d1, "F1_mean", "F1_lo", "F1_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=3:10) +
  scale_y_continuous(limits=c(0, 1)) +
  labs(x="Number of environments", y="F1 score") +
  base_theme

p2 <- ggplot() +
  add_layers(d2, "FD_mean", "FD_lo", "FD_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=3:10) +
  labs(x="Number of environments", y="False discoveries") +
  base_theme

p3 <- ggplot() +
  add_layers(d3, "time_mean", "time_lo", "time_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=3:10) +
  scale_y_log10(
    breaks=10^(-3:4),
    labels=scales::label_log(base=10)
  ) +
  labs(x="Number of environments", y="Runtime (sec)") +
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
final_plot <- plot_grid(panels, legend, ncol=1, rel_heights=c(1, .12))

final_plot


print(final_plot)
dev.off()
