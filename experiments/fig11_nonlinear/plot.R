script_path=sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('figures',showWarnings=FALSE)
pdf('figures/plot.pdf',width=10,height=5.5)
library(tidyverse)
library(cowplot)

# Read and combine
RS_R <- read_csv("results/RS_R.csv", show_col_types=FALSE)

RS_py <- read_csv("results/RS_py.csv", col_names=FALSE, show_col_types=FALSE) %>%
  set_names(names(RS_R))

RS <- bind_rows(RS_R, RS_py) %>%
  mutate(
    n=as.numeric(n),
    l=as.numeric(l),
    TP=as.numeric(TP),
    time=as.numeric(time),
    discover=tolower(as.character(discover))=="true",
    method=recode(method, "Wasserstein"="WVM"),
    method=recode(method, "nonlinearICP"="nonlinear ICP")
  )


method_levels <- c("nonlinear MIP", "nonlinear ICP", "WVM", "FAIR")

cols <- c(
  "nonlinear MIP"="#FF7F0E",
  "nonlinear ICP"="#2CA02C",
  "WVM"="#17BECF",
  "FAIR"="#1F77B4"
)

pointsize <- 2.2
linewidth <- .7

se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) > 1) sd(x)/sqrt(length(x)) else 0
}

df <- RS %>%
  mutate(
    method=factor(method, levels=method_levels),
    discover=as.numeric(discover)
  ) %>%
  group_by(n, method) %>%
  summarise(
    disc_se=se(discover), disc_mean=mean(discover, na.rm=TRUE),
    TP_se=se(TP), TP_mean=mean(TP, na.rm=TRUE),
    time_se=se(time), time_mean=mean(time, na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(
    disc_lo=pmax(0, disc_mean-disc_se),
    disc_hi=pmin(1, disc_mean+disc_se),
    TP_lo=pmax(0, TP_mean-TP_se),
    TP_hi=TP_mean+TP_se,
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

# Draw MIP last
add_layers <- function(d, y, lo, hi) {
  ds <- list(
    d %>% filter(method!="nonlinear MIP"),
    d %>% filter(method=="nonlinear MIP")
  )
  
  c(
    lapply(ds, \(z)
           geom_ribbon(
             data=z, aes(x=n, ymin=.data[[lo]], ymax=.data[[hi]], fill=method),
             alpha=.3, color=NA
           )),
    unlist(lapply(ds, \(z) list(
      geom_line(
        data=z, aes(x=n, y=.data[[y]], color=method, group=method),
        linewidth=linewidth
      ),
      geom_point(
        data=z, aes(x=n, y=.data[[y]], fill=method),
        shape=21, size=pointsize, stroke=.6, color="white"
      )
    )), recursive=FALSE)
  )
}

d1 <- df %>% filter(is.finite(disc_mean))
d2 <- df %>% filter(is.finite(TP_mean))
d3 <- df %>% filter(is.finite(time_mean), time_mean > 0)

p1 <- ggplot() +
  add_layers(d1, "disc_mean", "disc_lo", "disc_hi") +
  col_scale + fill_scale +
  scale_y_continuous(limits=c(0, 1)) +
  labs(x="Sample size", y="Proportion of exact recovery") +
  base_theme

p2 <- ggplot() +
  add_layers(d2, "TP_mean", "TP_lo", "TP_hi") +
  col_scale + fill_scale +
  scale_y_continuous(limits=c(0, 2), breaks=0:2) +
  labs(x="Sample size", y="True discoveries") +
  base_theme

p3 <- ggplot() +
  add_layers(d3, "time_mean", "time_lo", "time_hi") +
  col_scale + fill_scale +
  scale_y_log10(
    breaks=10^(-3:4),
    labels=scales::label_log(base=10)
  ) +
  labs(x="Sample size", y="Runtime (sec)") +
  base_theme

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
