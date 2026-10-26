# Execute with Rscript from any working directory.
script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(tidyverse)
library(patchwork)
library(latex2exp)
library(cowplot)

tab <- rbind(
  read.csv("results/RS0.csv", header=FALSE),
  read.csv("results/RS1.csv", header=FALSE)
) %>%
  `colnames<-`(c("n", "l", "method", "beta_diff", "subset", "recovery", "time"))

methods <- c("MIP-hidden", "EILLS-s", "Causal Dantzig", "BIP-s", "negDRO", "CoCo")
cols <- c(
  "MIP-hidden"="#FF7F0E", "EILLS-s"="#D62728",
  "Causal Dantzig"="#8C564B", "BIP-s"="#17BECF",
  "negDRO"="#1F77B4", "CoCo"="#9467BD"
)

n_grid <- c(1000, 2000, 3000, 4000, 5000)
pointsize <- 2.2
linewidth <- .7

se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)>1) sd(x)/sqrt(length(x)) else 0
}

df <- tab %>%
  filter(method %in% methods) %>%
  mutate(
    n=as.numeric(n),
    method=factor(method, levels=methods),
    beta_diff=as.numeric(beta_diff),
    recovery=case_when(
      tolower(as.character(recovery))=="true" ~ 1,
      tolower(as.character(recovery))=="false" ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  group_by(n, method) %>%
  summarise(
    beta_mean=mean(beta_diff, na.rm=TRUE),
    beta_se=se(beta_diff),
    recovery_mean=mean(recovery, na.rm=TRUE),
    recovery_se=se(recovery),
    .groups="drop"
  ) %>%
  mutate(
    recovery_mean=ifelse(is.nan(recovery_mean), NA, recovery_mean),
    beta_lo=beta_mean-beta_se,
    beta_hi=beta_mean+beta_se,
    rec_lo=pmax(0, recovery_mean-recovery_se),
    rec_hi=pmin(1, recovery_mean+recovery_se)
  )

col_scale <- scale_color_manual(NULL, values=cols, limits=methods, drop=FALSE)
fill_scale <- scale_fill_manual(NULL, values=cols, limits=methods, drop=FALSE)

base_theme <- theme_bw(base_size=9) +
  theme(
    axis.title=element_text(size=9),
    axis.text=element_text(size=9),
    panel.grid.major=element_line(color="grey88", linewidth=.35),
    panel.grid.minor=element_blank(),
    panel.border=element_rect(color="grey30", linewidth=.5),
    plot.margin=margin(5, 8, 5, 8),
    legend.position="none"
  )

add_layers <- function(d, y, lo=NULL, hi=NULL) {
  layers <- list()
  if (!is.null(lo)) {
    layers <- c(layers, list(
      geom_ribbon(
        data=d,
        aes(x=n, ymin=.data[[lo]], ymax=.data[[hi]], fill=method, group=method),
        alpha=.3, color=NA
      )
    ))
  }
  c(layers, list(
    geom_line(
      data=d,
      aes(x=n, y=.data[[y]], color=method, group=method),
      linewidth=linewidth
    ),
    geom_point(
      data=d,
      aes(x=n, y=.data[[y]], fill=method, group=method),
      shape=21, size=pointsize, stroke=.6, color="white"
    )
  ))
}

# Coefficient error
p1 <- ggplot() +
  add_layers(df, "beta_mean", "beta_lo", "beta_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=n_grid) +
  labs(
    x="Sample size",
    y=TeX("Coefficient error $\\|\\widehat{\\beta}-\\beta^*\\|_2$")
  ) +
  base_theme

# Perfect recovery
df_rec <- df %>% filter(is.finite(recovery_mean))

p2 <- ggplot() +
  add_layers(df_rec, "recovery_mean", "rec_lo", "rec_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=n_grid) +
  scale_y_continuous(limits=c(0,1), breaks=seq(0,1,.25)) +
  labs(
    x="Sample size",
    y="Proportion of perfect recovery"
  ) +
  base_theme

# Legend: colored line + filled circle with white border
draw_key_linepoint <- function(data, params, size) {
  grid::grobTree(
    grid::segmentsGrob(
      x0=grid::unit(.08, "npc"), x1=grid::unit(.92, "npc"),
      y0=grid::unit(.5, "npc"), y1=grid::unit(.5, "npc"),
      gp=grid::gpar(col=data$colour, lwd=1.5)
    ),
    grid::pointsGrob(
      x=grid::unit(.5, "npc"), y=grid::unit(.5, "npc"),
      pch=21, size=grid::unit(2.3, "mm"),
      gp=grid::gpar(col="white", fill=data$colour, lwd=1.2)
    )
  )
}

legend_df <- expand_grid(
  method=factor(methods, levels=methods),
  x=c(1,2)
) %>%
  mutate(y=1)

legend <- get_legend(
  ggplot(legend_df, aes(x, y, group=method, color=method)) +
    geom_line(linewidth=linewidth, key_glyph=draw_key_linepoint) +
    col_scale +
    theme_void() +
    theme(
      legend.position="bottom",
      legend.text=element_text(size=9),
      legend.key.width=grid::unit(1.1, "cm"),
      legend.margin=margin(0,0,0,0),
      legend.box.margin=margin(0,0,0,0),
      plot.margin=margin(0,0,0,0)
    ) +
    guides(color=guide_legend(nrow=1, byrow=TRUE))
)

panels <- plot_grid(p1, p2, nrow=1, align="h")

final_plot <- plot_grid(
  panels, legend,
  ncol=1,
  rel_heights=c(1,.08)
)

final_plot
print(final_plot)
dev.off()
