# ==========================================
# NHS Data Analysis and Visualization Script
# Reads datasets, links data, creates visualizations
# ==========================================
library(readxl)
library(dplyr)
library(stringr)
library(janitor)
library(purrr)
library(tidyr)
library(ggplot2)
library(scales)

base <- "."

# Helper functions
norm_postcode <- function(x) x %>%
  toupper() %>%
  str_replace_all("\\s+", "")

safe_round <- function(x, d=2) ifelse(is.na(x), NA_real_, round(x, d))

# Read datasets
df_pl  <- read_excel(file.path(base, "Practice_Location.xlsx")) %>% clean_names()

df_ll  <- read_excel(file.path(base, "lon_lat_dat.xlsx")) %>% clean_names()

df_copd_clean <- read_excel(file.path(base, "COPD_QoF_HeadingsCleaned.xlsx")) %>%
  clean_names()

qof_xls <- excel_sheets(file.path(base, "Qof Data.xlsx"))
df_qof_ast_raw <- read_excel(file.path(base, "Qof Data.xlsx"), sheet = "AST") %>%
  clean_names()
df_qof_copd_raw <- read_excel(file.path(base, "Qof Data.xlsx"), sheet = "COPD") %>%
  clean_names()

df_phe <- read_excel(file.path(base, "PHE_Fingertips_Cleaned.xlsx")) %>% clean_names()

df_map <- read_excel(file.path(base, "CCG_to_ICB_Mapping.xlsx")) %>% clean_names()

# Key standardization
df_pl <- df_pl %>%
  mutate(postcode_norm = norm_postcode(postcode))

df_ll <- df_ll %>%
  mutate(postcode_norm = norm_postcode(postcode)) %>%
  rename(lon = x, lat = y) %>%
  select(postcode, postcode_norm, everything())
if ("postcode" %in% names(df_copd_clean)) {
  df_copd_clean <- df_copd_clean %>%
    mutate(postcode_norm = norm_postcode(postcode))
}

# Join practice data with coordinates
df_practice <- df_pl %>%
  left_join(df_ll %>% select(postcode_norm, lon, lat, everything()),
            by = "postcode_norm")

# Join COPD data
df_practice_copd <- df_practice %>%
  left_join(df_copd_clean, by = c("practice_code" = "practice_code"))

# Administrative mapping
keep_map_cols <- intersect(names(df_map), c("ccg_code","ccg_name",
                                            "stp_code","stp_name",
                                            "nhse_region_code","nhse_region_name",
                                            "sub_icb_location_name",
                                            "icb_code","integrated_care_board_name",
                                            "icb_non_primary_role",
                                            "region_code","region_name"))
df_practice_copd_map <- df_practice_copd %>%
  left_join(df_map %>% select(all_of(keep_map_cols)),
            by = c("ccg_code" = "ccg_code"))

# Join PHE data
df_full <- df_practice_copd_map %>%
  left_join(
    df_phe %>%
      rename(phe_area_code = area_code,
             phe_area_name = area_name,
             phe_indicator_id = indicator_id,
             phe_indicator_name = indicator_name,
             phe_time_period = time_period,
             phe_value = value,
             phe_lower_ci_95 = lower_ci_95_0_limit,
             phe_upper_ci_95 = upper_ci_95_0_limit),
    by = c("sub_icb_loc_ons_code" = "phe_area_code")
  )

# Optional AST data integration
if ("practice_code" %in% names(df_qof_ast_raw)) {
  df_qof_ast_raw_pref <- df_qof_ast_raw %>%
    rename_with(.fn = ~ paste0("ast__", .x), .cols = -c(practice_code))
  df_full <- df_full %>%
    left_join(df_qof_ast_raw_pref, by = c("practice_code" = "practice_code"))
} else {
  cat("Note: AST data does not contain practice_code column, skipping AST integration\n")
}

# Optional COPD raw data integration
if ("practice_code" %in% names(df_qof_copd_raw)) {
  df_qof_copd_raw_pref <- df_qof_copd_raw %>%
    rename_with(.fn = ~ paste0("copdraw__", .x), .cols = -c(practice_code))
  df_full <- df_full %>%
    left_join(df_qof_copd_raw_pref, by = c("practice_code" = "practice_code"))
} else {
  cat("Note: COPD raw data does not contain practice_code column, skipping COPD raw integration\n")
}

# Calculate overall COPD percentage
pct_cols <- names(df_full)[str_detect(names(df_full), "copd(007|008|010).*percentage")]
if (length(pct_cols) > 0) {
  for(col in pct_cols) {
    if(col %in% names(df_full)) {
      df_full[[col]] <- as.numeric(df_full[[col]])
    }
  }
  
  df_full <- df_full %>%
    rowwise() %>%
    mutate(copd_overall_percentage = safe_round(mean(c_across(all_of(pct_cols)), na.rm = TRUE), 2)) %>%
    ungroup() %>%
    mutate(copd_overall_percentage = ifelse(is.nan(copd_overall_percentage), NA, copd_overall_percentage))
}

# Data aggregation by administrative levels
rate_cols <- names(df_full)[str_detect(names(df_full), "copd.*percentage")]

agg_by <- function(df, by_cols) {
  df_numeric <- df
  for(col in rate_cols) {
    if(col %in% names(df_numeric)) {
      df_numeric[[col]] <- as.numeric(df_numeric[[col]])
    }
  }
  
  df_numeric %>%
    group_by(across(all_of(by_cols))) %>%
    summarize(
      n_practices = n_distinct(practice_code, na.rm = TRUE),
      across(all_of(rate_cols), ~ mean(., na.rm = TRUE), .names = "mean_{.col}"),
      .groups = "drop"
    ) %>%
    mutate(across(starts_with("mean_"), ~ ifelse(is.nan(.), NA, .)))
}

# ICB level
agg_icb <- agg_by(df_full, c("icb_code","integrated_care_board_name"))

# Region level
region_cols <- c("region_code","region_name")
region_cols <- region_cols[region_cols %in% names(df_full)]
if (length(region_cols) == 2) {
  agg_region <- agg_by(df_full, region_cols)
} else {
  agg_region <- tibble()
}

# Sub-ICB level
subicb_cols <- c("sub_icb_loc_ons_code","sub_icb_loc_name")
subicb_cols <- subicb_cols[subicb_cols %in% names(df_full)]
if (length(subicb_cols) >= 1) {
  agg_subicb <- agg_by(df_full, subicb_cols)
} else {
  agg_subicb <- tibble()
}

# Export data files
results_dir <- file.path(base, "NHS_Analysis_Results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir)
  cat("Created results directory:", results_dir, "\n")
}

write.csv(df_full, file.path(results_dir, "linked_practice_level_full.csv"), row.names = FALSE)
if (nrow(agg_icb) > 0)     write.csv(agg_icb,    file.path(results_dir, "agg_icb_level.csv"), row.names = FALSE)
if (nrow(agg_region) > 0)  write.csv(agg_region, file.path(results_dir, "agg_region_level.csv"), row.names = FALSE)
if (nrow(agg_subicb) > 0)  write.csv(agg_subicb, file.path(results_dir, "agg_subicb_level.csv"), row.names = FALSE)

# Data Visualization
cat("Creating visualizations...\n")

copd_pct_cols <- names(df_full)[str_detect(names(df_full), "copd.*percentage")]

# ICB Level Comparison Chart
if(nrow(agg_icb) > 0) {
  mean_cols <- names(agg_icb)[str_detect(names(agg_icb), "mean_")]
  
  if(length(mean_cols) > 0) {
    target_col <- mean_cols[1]
    
    p_icb <- agg_icb %>%
      filter(!is.na(.data[[target_col]])) %>%
      arrange(desc(.data[[target_col]])) %>%
      slice_head(n = 20) %>%
      ggplot(aes(x = reorder(integrated_care_board_name, .data[[target_col]]), 
                 y = .data[[target_col]])) +
      geom_col(fill = "steelblue", alpha = 0.8) +
      coord_flip() +
      labs(title = "ICB Level COPD Indicators Comparison",
           x = "ICB Name",
           y = "Mean Value (%)") +
      theme_minimal(base_size = 10) +
      theme(axis.text.y = element_text(size = 8))
    
    ggsave(file.path(results_dir, "icb_comparison.png"), p_icb, 
           width = 14, height = 10, dpi = 150)
    cat("Saved: icb_comparison.png\n")
  }
}

# Practice Geographic Distribution
if(all(c("lon", "lat") %in% names(df_full))) {
  color_col <- if("copd_overall_percentage" %in% names(df_full)) {
    "copd_overall_percentage"
  } else if(length(copd_pct_cols) > 0) {
    copd_pct_cols[1]
  } else {
    NULL
  }
  
  if(!is.null(color_col)) {
    p_map <- df_full %>%
      filter(!is.na(lon), !is.na(lat), !is.na(.data[[color_col]])) %>%
      ggplot(aes(x = lon, y = lat, color = .data[[color_col]])) +
      geom_point(alpha = 0.6, size = 0.8) +
      scale_color_viridis_c(name = "COPD %") +
      coord_quickmap() +
      labs(title = "Practice Geographic Distribution by COPD Performance",
           x = "Longitude", y = "Latitude") +
      theme_minimal()
    
    ggsave(file.path(results_dir, "practice_geographic_distribution.png"), p_map, 
           width = 12, height = 8, dpi = 150)
    cat("Saved: practice_geographic_distribution.png\n")
  }
}

# Sub-ICB Boxplot Analysis
if(nrow(agg_subicb) > 0 && length(copd_pct_cols) > 0) {
  target_col <- copd_pct_cols[1]
  
  df_box <- df_full %>%
    mutate(target_numeric = as.numeric(.data[[target_col]])) %>%
    filter(!is.na(target_numeric), !is.na(sub_icb_loc_name)) %>%
    group_by(sub_icb_loc_name) %>%
    mutate(n_practices = n()) %>%
    ungroup() %>%
    filter(n_practices >= 10) %>%
    slice_sample(n = min(5000, nrow(.)))
  
  if(nrow(df_box) > 0) {
    top_subicbs <- df_box %>%
      group_by(sub_icb_loc_name) %>%
      summarise(median_val = median(target_numeric, na.rm = TRUE), .groups = 'drop') %>%
      filter(!is.na(median_val)) %>%
      arrange(desc(median_val)) %>%
      slice_head(n = 15) %>%
      pull(sub_icb_loc_name)
    
    if(length(top_subicbs) > 0) {
      p_box <- df_box %>%
        filter(sub_icb_loc_name %in% top_subicbs) %>%
        ggplot(aes(x = reorder(sub_icb_loc_name, target_numeric, median, na.rm = TRUE),
                   y = target_numeric)) +
        geom_boxplot(alpha = 0.7) +
        geom_jitter(width = 0.2, alpha = 0.3, size = 0.5) +
        coord_flip() +
        labs(title = "Sub-ICB Practice Distribution Analysis",
             x = "Sub-ICB", y = "COPD Indicator Value (%)") +
        theme_minimal(base_size = 10) +
        theme(axis.text.y = element_text(size = 8))
      
      ggsave(file.path(results_dir, "subicb_boxplot.png"), p_box, 
             width = 14, height = 10, dpi = 150)
      cat("Saved: subicb_boxplot.png\n")
    }
  }
}

# Top Practices Ranking
if(length(copd_pct_cols) > 0) {
  target_col <- copd_pct_cols[1]
  
  practice_name_col <- if("practice_name.x" %in% names(df_full)) {
    "practice_name.x"
  } else if("practice_name" %in% names(df_full)) {
    "practice_name"
  } else {
    NULL
  }
  
  if(!is.null(practice_name_col)) {
    top_practices <- df_full %>%
      filter(!is.na(.data[[target_col]]), !is.na(.data[[practice_name_col]])) %>%
      arrange(desc(.data[[target_col]])) %>%
      slice_head(n = 15)
    
    p_top <- ggplot(top_practices, 
                    aes(x = reorder(.data[[practice_name_col]], .data[[target_col]]), 
                        y = .data[[target_col]])) +
      geom_col(fill = "darkgreen", alpha = 0.8) +
      coord_flip() +
      labs(title = "Top 15 Best Performing Practices",
           x = "Practice Name", y = "COPD Indicator Value (%)") +
      theme_minimal(base_size = 10) +
      theme(axis.text.y = element_text(size = 8))
    
    ggsave(file.path(results_dir, "top_practices_ranking.png"), p_top, 
           width = 14, height = 8, dpi = 150)
    cat("Saved: top_practices_ranking.png\n")
  }
}

# Analysis Summary Report
cat("Generating analysis summary report...\n")

report_content <- paste0(
  "# NHS Data Analysis Report\n",
  "Generated: ", Sys.time(), "\n\n",
  "## Data Overview\n",
  "- Total Practices: ", n_distinct(df_full$practice_code, na.rm = TRUE), "\n",
  "- Total ICBs: ", n_distinct(df_full$integrated_care_board_name, na.rm = TRUE), "\n",
  "- Total Sub-ICBs: ", n_distinct(df_full$sub_icb_loc_name, na.rm = TRUE), "\n",
  "- Total Records: ", nrow(df_full), "\n",
  "- Total Variables: ", ncol(df_full), "\n\n",
  "## COPD Indicators Statistics\n"
)

if(length(copd_pct_cols) > 0) {
  target_col <- copd_pct_cols[1]
  summary_stats <- df_full %>%
    mutate(target_numeric = as.numeric(.data[[target_col]])) %>%
    filter(!is.na(target_numeric)) %>%
    summarise(
      count = n(),
      mean_val = mean(target_numeric, na.rm = TRUE),
      median_val = median(target_numeric, na.rm = TRUE),
      min_val = min(target_numeric, na.rm = TRUE),
      max_val = max(target_numeric, na.rm = TRUE)
    )
  
  if(summary_stats$count > 0 && !is.na(summary_stats$mean_val)) {
    report_content <- paste0(report_content,
      "Primary Indicator: ", target_col, "\n",
      "- Sample Size: ", summary_stats$count, "\n",
      "- Mean: ", round(summary_stats$mean_val, 2), "%\n",
      "- Median: ", round(summary_stats$median_val, 2), "%\n",
      "- Minimum: ", round(summary_stats$min_val, 2), "%\n",
      "- Maximum: ", round(summary_stats$max_val, 2), "%\n\n"
    )
  } else {
    report_content <- paste0(report_content,
      "Primary Indicator: ", target_col, "\n",
      "- Note: No valid numeric data available for statistical analysis\n\n"
    )
  }
}

report_content <- paste0(report_content,
  "## Generated Files\n",
  "### Data Files\n",
  "- linked_practice_level_full.csv: Complete practice-level dataset\n",
  "- agg_icb_level.csv: ICB-level aggregated data\n",
  "- agg_subicb_level.csv: Sub-ICB-level aggregated data\n\n",
  "### Visualization Files\n",
  "- icb_comparison.png: ICB-level COPD indicators comparison\n",
  "- practice_geographic_distribution.png: Geographic distribution of practices\n",
  "- subicb_boxplot.png: Sub-ICB practice distribution analysis\n",
  "- top_practices_ranking.png: Top 15 best performing practices\n"
)

writeLines(report_content, file.path(results_dir, "analysis_summary_report.txt"))

cat("\n=== Analysis Complete ===\n")
cat("All results saved to directory:", results_dir, "\n")
cat("Files written:\n",
    file.path(results_dir, "linked_practice_level_full.csv"), "\n",
    file.path(results_dir, "agg_icb_level.csv"), "\n",
    file.path(results_dir, "agg_region_level.csv"), "\n",
    file.path(results_dir, "agg_subicb_level.csv"), "\n",
    file.path(results_dir, "analysis_summary_report.txt"), "\n")

cat("Generated files:\n")
list.files(results_dir) %>% paste0("- ", .) %>% cat(sep = "\n")
