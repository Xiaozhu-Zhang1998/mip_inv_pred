# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot_all.pdf',width=10,height=5.5)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(latex2exp)
source("plot_left_middle.R",local=TRUE)

tab = read.csv("results/RS_reg.csv", header=FALSE) %>%
  `colnames<-`(c("gamma", "lam", "l", "method", "beta_hat", "loeo", "subset", "causal", "len", "time")) 


# Summarize over trials
df_heat <- tab %>%
  mutate(
    gamma = as.numeric(gamma),
    lam = as.numeric(lam),
    loeo = as.numeric(loeo),
    causal = tolower(as.character(causal)) == "true"
  ) %>%
  group_by(gamma, lam, method) %>%
  summarise(
    loeo = mean(loeo, na.rm=TRUE),
    causal_rate = mean(causal, na.rm=TRUE),
    .groups="drop"
  )

# Heatmap
p <- ggplot(
  df_heat,
  aes(
    x=factor(gamma, levels=sort(unique(gamma))),
    y=factor(lam, levels=sort(unique(lam), decreasing=TRUE)),
    fill=loeo
  )
) +
  geom_tile(color="white", linewidth=.5) +
  geom_text(aes(label=sprintf("%.3f", loeo)), size=3) +
  geom_tile(
    data=df_heat %>% filter(causal_rate > .95),
    fill=NA, color="#E41A1C", linewidth=0.7
  ) +
  scale_fill_gradient(
    low="#EFF3FF",
    high="#3182BD",
    name="Average LOEO"
  ) +
  scale_y_discrete(
    labels=function(x)
      parse(text=paste0("10^", round(log10(as.numeric(x)))))
  ) +
  labs(
    x=TeX("Tuning parameter $\\gamma$"),
    y=TeX("Tuning parameter $\\lambda_n$")
  ) +
  theme_bw(base_size=10) +
  theme(
    panel.grid=element_blank(),
    axis.title=element_text(size=9),
    axis.text=element_text(size=9),
    legend.title=element_text(size=8.5),
    legend.text=element_text(size=8.5),
    plot.margin=margin(6, 8, 6, 8)
  )

wrap_elements(full=final_plot) + p +
  plot_layout(widths=c(2.5, 1))


print(wrap_elements(full=final_plot) + p + plot_layout(widths=c(2.5,1)))
dev.off()
