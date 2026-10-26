# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(tidyverse)
library(patchwork)
library(latex2exp)

tab = read.csv("results/RS1.csv", header = FALSE) %>%
  `colnames<-`(c("l", "method", "beta_diff", "subset", "recovery", "with1", "time")) 

tab[tab == 'ELLIS-s'] = 'EILLS'
tab[tab == 'BIP-s'] = 'BIP'

method_levels <- c("MIP", "EILLS", "ICP", "Causal Dantzig", "BIP")

# cols <- c(
#   "MIP"="#FF7F0E",
#   "EILLS-s"="#D62728",
#   "ICP"="#2CA02C",
#   "Hidden-ICP"="#9467BD",
#   "BIP-s"="#17BECF"
# )

df_plot <- tab %>%
  mutate(
    l=as.numeric(l),
    method=factor(method, levels=method_levels),
    group=floor(l/100)+1,
    recovery=tolower(as.character(recovery))=="true",
    with1=tolower(as.character(with1))=="true"
  ) %>%
  group_by(method, group) %>%
  summarise(
    recovery=mean(recovery, na.rm=TRUE),
    with1=mean(with1, na.rm=TRUE),
    .groups="drop"
  ) %>%
  pivot_longer(
    c(recovery, with1),
    names_to="metric",
    values_to="proportion"
  ) %>%
  mutate(
    metric=recode(
      metric,
      recovery="Perfect recovery",
      with1="Selecting feature 1"
    )
  )

base_theme <- theme_bw(base_size=10) +
  theme(
    legend.position="none",
    axis.text.x=element_text(angle=30, hjust=1),
    axis.title=element_text(size=9),
    panel.border=element_rect(color="grey35", linewidth=.5)
  )

p1 <- df_plot %>%
  filter(metric=="Perfect recovery") %>%
  ggplot(aes(method, proportion, color=method)) +
  geom_boxplot(width=.45, outlier.size=1, linewidth=.5) +
  scale_y_continuous(expand=expansion(mult=c(.03, .08))) +
  labs(
    x=NULL,
    y="Proportion of exact recovery"
  ) +
  base_theme

p2 <- df_plot %>%
  filter(metric=="Selecting feature 1") %>%
  ggplot(aes(method, proportion, color=method)) +
  geom_boxplot(width=.45, outlier.size=1, linewidth=.5) +
  scale_y_continuous(expand=expansion(mult=c(.03, .08))) +
  labs(
    x=NULL,
    y=TeX("Proportion of selecting $X_1$")
  ) +
  base_theme

p1 + p2

print(p1+p2)
dev.off()
