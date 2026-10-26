# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot_fig08.pdf',width=10,height=5.5)
library(tidyverse)
library(patchwork)
library(latex2exp)
tab = read.csv("results/RS.csv", header = FALSE) %>%
  `colnames<-`(c("n", "l", "method", "len", "get0", "coverage", "invariant", "fd", "time"))



tab <- tab %>%
  filter(!(method %in% c("MIP-test-2", "Causal Dantzig"))) %>%
  mutate(across(c(get0, coverage, invariant), ~ as.integer(.x == "True"))) %>%
  mutate(method=recode(method, "Wasserstein"="WVM"))

methods <- c(
  "MIP-test-2", "MIP-test-4", "ICP", "WVM",
  "MIP", "EILLS", "Causal Dantzig", "BIP-s"
)

cols <- c(
  "MIP-test-2"="#FF7F0E", "MIP-test-4"="#D62728",
  "ICP"="#2CA02C", "WVM"="#1F77B4",
  "MIP"="#9467BD", "EILLS"="#8C564B",
  "Causal Dantzig"="#17BECF", "BIP-s"="#E377C2"
)

df <- tab %>%
  filter(method %in% methods) %>%
  group_by(n, method) %>%
  summarise(
    coverage=mean(coverage),
    invariant=mean(invariant),
    fd=mean(fd),
    .groups="drop"
  ) %>%
  arrange(method, n)

plot_xy <- function(x, y, xlab, ylab, xlim=NULL, ylim=NULL) {
  ggplot(df, aes(.data[[x]], .data[[y]])) +
    geom_point(
      aes(fill=method),
      shape=21, size=2, stroke=.4, col="white",
      position=position_jitter(width=.01, height=.01, seed = 1)
    ) +
    scale_fill_manual(values=cols, breaks=methods) +
    coord_cartesian(xlim=xlim, ylim=ylim) +
    labs(x=xlab, y=ylab, fill=NULL) +
    theme_bw(base_size=10) +
    theme(
      panel.grid.minor=element_blank(),
      panel.grid.major=element_line(color="grey90"),
      legend.position="bottom"
    )
}

p1 <- plot_xy(
  "coverage", "invariant",
  "Coverage",
  "Prop of invariance",
  c(0,1), c(0,1)
)

p2 <- plot_xy(
  "coverage", "fd",
  "Coverage",
  "False discoveries",
  c(0,1)
)

p3 <- plot_xy(
  "invariant", "fd",
  "Prop of invariance",
  "False discoveries",
  c(0,1)
)

(p1 | p2 | p3) +
  plot_layout(guides="collect") &
  theme(
    axis.title=element_text(size=9),
    legend.position="bottom",
    legend.text=element_text(size=9)
  ) &
  guides(fill=guide_legend(nrow=1, byrow=TRUE))


print((p1|p2|p3)+plot_layout(guides="collect"))
dev.off()
