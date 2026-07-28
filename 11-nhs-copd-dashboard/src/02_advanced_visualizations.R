###############################################
# NHS Data Visualization Script
# Based on linked CSV files
###############################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(scales)
  library(ggrepel)
  library(maps)
})

# Setup paths and read data
base <- "."
out_dir <- file.path(base, "NHS_Analysis_Results")

f_full   <- file.path(out_dir, "linked_practice_level_full.csv")
f_icb    <- file.path(out_dir, "agg_icb_level.csv")
f_sub    <- file.path(out_dir, "agg_subicb_level.csv")

df_full  <- readr::read_csv(f_full)  %>% clean_names()
agg_icb  <- if (file.exists(f_icb)) readr::read_csv(f_icb) %>% clean_names() else tibble()
agg_sub  <- if (file.exists(f_sub)) readr::read_csv(f_sub) %>% clean_names() else tibble()

# Output directory
plot_dir <- file.path(base, "NHS_Analysis_Results")
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

save_plot <- function(p, filename, w = 10, h = 6) {
  fp <- file.path(plot_dir, filename)
  ggsave(fp, p, width = w, height = h, dpi = 180)
  message("Saved plot: ", fp)
}

copd_pct_cols <- names(df_full)[str_detect(names(df_full),
                                           regex("copd(007|008|010).*percentage", ignore_case = TRUE))]
if (length(copd_pct_cols) == 0) {
  stop("No COPD percentage columns found in df_full")
}
message("Detected COPD percentage columns: ", paste(copd_pct_cols, collapse = ", "))

# ICB Level COPD Indicators Bar Chart
if (nrow(agg_icb) > 0) {
  icb_long <- agg_icb %>%
    pivot_longer(cols = matches("mean_copd(008|010).*percentage"),
                 names_to = "metric", values_to = "rate") %>%
    mutate(
      rate = as.numeric(rate),
      kpi = case_when(
        str_detect(metric, "copd007") ~ "COPD007",
        str_detect(metric, "copd008") ~ "COPD008",
        str_detect(metric, "copd010") ~ "COPD010",
        TRUE ~ "OTHER"
      )
    ) %>%
    filter(!is.na(rate))

  p_icb <- icb_long %>%
    ggplot(aes(x = reorder(integrated_care_board_name, rate),
               y = rate, fill = kpi)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    coord_flip() +
    scale_y_continuous(labels = function(x) paste0(x, "%")) +
    labs(title = "ICB Level COPD Indicators (Weighted Achievement Rate)",
         x = "Integrated Care Board",
         y = "Weighted Achievement Rate (%)",
         fill = "QOF KPI") +
    theme_minimal(base_size = 12)

  print(p_icb)
  save_plot(p_icb, "01_icb_weighted_rates.png")
} else {
  message("No ICB data available, skipping ICB bar chart.")
}

# Sub-ICB Practice Distribution Boxplot
target_col <- copd_pct_cols[1]
if (!all(c("sub_icb_loc_name", target_col) %in% names(df_full))) {
  warning("Missing sub_icb_loc_name or target KPI column, skipping boxplot.")
} else {
  # First, identify top 15 Sub-ICBs by median performance
  top_subicbs <- df_full %>%
    mutate(target_numeric = as.numeric(.data[[target_col]])) %>%
    filter(!is.na(target_numeric), !is.na(sub_icb_loc_name)) %>%
    group_by(sub_icb_loc_name) %>%
    summarise(
      median_val = median(target_numeric, na.rm = TRUE),
      n_prac = n_distinct(practice_code),
      .groups = 'drop'
    ) %>%
    filter(n_prac >= 10) %>%
    arrange(desc(median_val)) %>%
    slice_head(n = 15) %>%
    pull(sub_icb_loc_name)
  
  # Create boxplot for top 15 Sub-ICBs only
  p_box <- df_full %>%
    mutate(target_numeric = as.numeric(.data[[target_col]])) %>%
    filter(!is.na(target_numeric),
           !is.na(sub_icb_loc_name),
           sub_icb_loc_name %in% top_subicbs) %>%
    ggplot(aes(x = reorder(sub_icb_loc_name, target_numeric,
                           FUN = median, na.rm = TRUE),
               y = target_numeric)) +
    geom_boxplot(outlier.alpha = 0.2) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 0.7) +
    coord_flip() +
    labs(title = paste0("Top 15 Sub-ICB ", target_col, " Distribution"),
         x = "Sub-ICB",
         y = "Achievement Rate (%)") +
    theme_minimal(base_size = 12) +
    theme(axis.text.y = element_text(size = 10))

  print(p_box)
  save_plot(p_box, "02_subicb_boxplot.png", w = 14, h = 10)
}

# Practice Geographic Distribution Map
if (all(c("lon", "lat") %in% names(df_full))) {

  if (!"copd_overall_percentage" %in% names(df_full)) {
    df_full <- df_full %>%
      mutate(copd_overall_percentage = .data[[copd_pct_cols[1]]])
  }

  p_map <- df_full %>%
    filter(!is.na(lon), !is.na(lat)) %>%
    mutate(copd_numeric = as.numeric(copd_overall_percentage)) %>%
    filter(!is.na(copd_numeric)) %>%
    ggplot(aes(x = lon, y = lat, color = copd_numeric)) +
    annotation_map(map_data("world"), xlim = c(-8, 2), ylim = c(49, 61), fill = NA) +
    geom_point(alpha = 0.7, size = 1) +
    scale_color_viridis_c(option = "C") +
    coord_quickmap() +
    labs(title = "Practice Geographic Distribution (COPD Performance)",
         x = NULL, y = NULL,
         color = "COPD (%)") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())

  print(p_map)
  save_plot(p_map, "03_practice_map.png", w = 8, h = 10)
} else {
  warning("No lon/lat columns found in df_full, skipping map visualization.")
}

# PCA vs KPI Scatter Plot
pca_cols   <- names(df_full)[str_detect(names(df_full), "exception")]
denp_cols  <- names(df_full)[str_detect(names(df_full), "denominator_plus_pca")]
rate_col   <- copd_pct_cols[1]

if (length(pca_cols) > 0 && length(denp_cols) > 0) {

  pca_col  <- pca_cols[1]
  denp_col <- denp_cols[1]

  df_scatter <- df_full %>%
    mutate(pca_rate = 100 * (.data[[pca_col]] / pmax(.data[[denp_col]], 1))) %>%
    filter(!is.na(pca_rate), !is.na(.data[[rate_col]]))

  p_pca <- ggplot(df_scatter,
                  aes(x = pca_rate,
                      y = .data[[rate_col]],
                      color = integrated_care_board_name)) +
    geom_point(alpha = 0.6, size = 1.2) +
    geom_hline(yintercept = median(df_scatter[[rate_col]], na.rm = TRUE), linetype = 2) +
    geom_vline(xintercept = median(df_scatter$pca_rate, na.rm = TRUE), linetype = 2) +
    labs(title = paste0("PCA Rate vs ", rate_col, " (Practice Level)"),
         x = "PCA Rate (%)",
         y = "Achievement Rate (%)",
         color = "ICB") +
    theme_minimal(base_size = 12)

  print(p_pca)
  save_plot(p_pca, "04_pca_vs_kpi_scatter.png")
} else {
  warning("No exception or denominator_plus_pca columns found, skipping PCA scatter plot.")
}

# AST vs COPD Correlation Analysis
ast_pct_cols <- names(df_full)[str_detect(names(df_full), "^ast__.*percentage")]
if (length(ast_pct_cols) > 0) {
  ast_col  <- ast_pct_cols[1]
  copd_col <- copd_pct_cols[1]

  df_corr <- df_full %>%
    filter(!is.na(.data[[ast_col]]), !is.na(.data[[copd_col]]))

  p_corr <- ggplot(df_corr,
                   aes(x = .data[[ast_col]],
                       y = .data[[copd_col]])) +
    geom_point(alpha = 0.5, size = 1.2) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(title = "AST vs COPD (Practice Level Correlation)",
         x = paste0(ast_col, " (AST %)"),
         y = paste0(copd_col, " (COPD %)")) +
    theme_minimal(base_size = 12)

  print(p_corr)
  save_plot(p_corr, "05_ast_vs_copd_scatter.png")
} else {
  message("No AST percentage columns detected, skipping AST vs COPD plot.")
}

# PHE Time Series Analysis
if (all(c("phe_indicator_name","phe_time_period","phe_value","sub_icb_loc_name") %in% names(df_full))) {

  top_ind <- df_full %>%
    filter(!is.na(phe_indicator_name)) %>%
    count(phe_indicator_name, sort = TRUE) %>%
    slice(1) %>%
    pull(phe_indicator_name)

  top_area <- df_full %>%
    filter(!is.na(sub_icb_loc_name)) %>%
    count(sub_icb_loc_name, sort = TRUE) %>%
    slice(1) %>%
    pull(sub_icb_loc_name)

  df_ts <- df_full %>%
    filter(phe_indicator_name == top_ind,
           sub_icb_loc_name == top_area) %>%
    mutate(time_ord = factor(phe_time_period, levels = unique(phe_time_period)))

  p_ts <- ggplot(df_ts,
                 aes(x = time_ord,
                     y = phe_value,
                     group = 1)) +
    geom_line() +
    geom_point() +
    labs(title = paste0("PHE Time Series: ", top_ind, " - ", top_area),
         x = "Time Period",
         y = "Value") +
    theme_minimal(base_size = 12)

  print(p_ts)
  save_plot(p_ts, "06_phe_timeseries.png")
} else {
  message("PHE fields incomplete, skipping PHE time series.")
}

message("Visualization complete. All plots saved to: ", plot_dir)
