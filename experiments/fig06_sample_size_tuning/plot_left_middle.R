library(tidyverse)
library(patchwork)
library(latex2exp)
library(cowplot)


RS = read.csv("results/RS.csv", header = FALSE) %>%
  `colnames<-`(c("n", "E", "l", "method", "beta_diff", "subset", "recover", "len", "time"))


method_keep <- "CD"

E_levels <- c("2", "3", "5", "10")
cols <- c("2"="#1F77B4", "3"="#9467BD", "5"="#FF7F0E", "10"="#2CA02C")

n_grid <- seq(50, 500, 50)
pointsize <- 2.2
linewidth <- .7

se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) > 1) sd(x)/sqrt(length(x)) else 0
}

# Data
df <- RS %>%
  filter(method == method_keep) %>%
  mutate(
    n=as.numeric(n),
    E=factor(E, levels=E_levels),
    beta_diff=as.numeric(beta_diff),
    recover=case_when(
      tolower(as.character(recover))=="true" ~ 1,
      tolower(as.character(recover))=="false" ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  group_by(n, E) %>%
  summarise(
    beta_diff_mean=mean(beta_diff, na.rm=TRUE),
    beta_diff_se=se(beta_diff),
    recover_mean=mean(recover, na.rm=TRUE),
    recover_se=se(recover),
    .groups="drop"
  ) %>%
  mutate(
    beta_lo=beta_diff_mean-beta_diff_se,
    beta_hi=beta_diff_mean+beta_diff_se,
    rec_lo=pmax(0, recover_mean-recover_se),
    rec_hi=pmin(1, recover_mean+recover_se)
  )

col_scale <- scale_color_manual(NULL, values=cols, limits=E_levels, drop=FALSE)
fill_scale <- scale_fill_manual(NULL, values=cols, limits=E_levels, drop=FALSE)

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

add_layers <- function(d, y, lo, hi) {
  list(
    geom_ribbon(
      data=d,
      aes(x=n, ymin=.data[[lo]], ymax=.data[[hi]], fill=E, group=E),
      alpha=.3, color=NA
    ),
    geom_line(
      data=d,
      aes(x=n, y=.data[[y]], color=E, group=E),
      linewidth=linewidth
    ),
    geom_point(
      data=d,
      aes(x=n, y=.data[[y]], fill=E, group=E),
      shape=21, size=pointsize, stroke=.6, color="white"
    )
  )
}

# Coefficient error
p1 <- ggplot() +
  add_layers(df, "beta_diff_mean", "beta_lo", "beta_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=n_grid) +
  labs(
    x="Sample size for each env",
    y=TeX("Coefficient error $\\|\\widehat{\\beta}-\\beta^*\\|_2$")
  ) +
  base_theme

# Perfect recovery
p2 <- ggplot() +
  add_layers(df, "recover_mean", "rec_lo", "rec_hi") +
  col_scale + fill_scale +
  scale_x_continuous(breaks=n_grid) +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, .25)) +
  labs(
    x=TeX("Sample size for each env"),
    y="Proportion of perfect recovery"
  ) +
  base_theme

# Custom legend glyph: colored line + colored point + white point border
draw_key_linepoint <- function(data, params, size) {
  grid::grobTree(
    grid::segmentsGrob(
      x0=grid::unit(.08, "npc"),
      x1=grid::unit(.92, "npc"),
      y0=grid::unit(.5, "npc"),
      y1=grid::unit(.5, "npc"),
      gp=grid::gpar(
        col=data$colour,
        lwd=1.5
      )
    ),
    grid::pointsGrob(
      x=grid::unit(.5, "npc"),
      y=grid::unit(.5, "npc"),
      pch=21,
      size=grid::unit(2.3, "mm"),
      gp=grid::gpar(
        col="white",
        fill=data$colour,
        lwd=1.2
      )
    )
  )
}

# Dummy data only for common legend
legend_df <- expand_grid(
  E=factor(E_levels, levels=E_levels),
  x=c(1, 2)
) %>%
  mutate(y=1)

legend <- get_legend(
  ggplot(legend_df, aes(x, y, group=E, color=E)) +
    geom_line(
      linewidth=linewidth,
      key_glyph=draw_key_linepoint
    ) +
    col_scale +
    theme_void() +
    theme(
      legend.position="bottom",
      legend.text=element_text(size=9),
      legend.title=element_text(size=9),
      legend.key.width=grid::unit(1.1, "cm"),
      legend.margin=margin(0, 0, 0, 0),
      legend.box.margin=margin(0, 0, 0, 0),
      plot.margin=margin(0, 0, 0, 0)
    ) +
    guides(
      color=guide_legend(
        title="Number of envs E",
        nrow=1, byrow=TRUE
      )
    )
)

panels <- plot_grid(
  p1, p2,
  nrow=1,
  align="h"
)

final_plot <- plot_grid(
  panels, legend,
  ncol=1,
  rel_heights=c(1, .09)
)

# final_plot



