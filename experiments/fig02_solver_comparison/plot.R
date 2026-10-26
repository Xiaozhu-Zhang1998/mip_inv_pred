# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(tidyverse)
library(patchwork)
library(latex2exp)
tab = read.csv("results/RS.csv", header = FALSE) %>%
  `colnames<-`(c("p", "l", "method", "diff_beta", "superset", "recovery", "len", "time"))


methods <- c("CD", "MIP (warm start)", "MIP")

df <- tab %>% 
  mutate(method=recode(method, "MIP(warmStart)"="MIP (warm start)")) %>%
  filter(method %in% methods) %>%
  filter(p < 60) 
  

plot_box <- function(y, ylab) {
  ggplot(df, aes(factor(p), .data[[y]], col=method)) +
    geom_boxplot(
      position=position_dodge(.75),
      width=.6, fill="white",
      outlier.size=.7, linewidth=.6
    ) +
    # scale_color_manual(values=cols, breaks=methods) +
    labs(x="Dimension p", y=ylab, col=NULL) +
    theme_bw(base_size=10) +
    theme(
      panel.grid.minor=element_blank(),
      panel.grid.major=element_line(color="grey90"),
      legend.position="bottom"
    )
}

p1 <- plot_box(
  "diff_beta",
  TeX("Coefficient error $\\|\\widehat{\\beta}-\\beta^*\\|_2$")
)

p2 <- plot_box("time", "Runtime (secs)") +
  scale_y_log10(
    breaks=10^(-3:3),
    labels=scales::label_log(base=10)
  )

(p1 | p2) +
  plot_layout(guides="collect") &
  theme(
    legend.position="bottom",
    legend.text=element_text(size=9),
    axis.title=element_text(size=9)
  ) &
  guides(color=guide_legend(nrow=1))

print((p1|p2)+plot_layout(guides="collect"))
dev.off()
