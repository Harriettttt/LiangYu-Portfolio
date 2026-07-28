library(tidyverse)
library(readxl)
library(janitor)
library(plotly)
library(patchwork)
library(DT)
library(sf)
library(leaflet)
library(shiny)
library(shinydashboard)
library(htmltools)
library(viridis)  # 色盲友好调色板

# 创建www文件夹（如果不存在）
if (!dir.exists("www")) dir.create("www")
# shiny::addResourcePath("static", normalizePath("www"))  # 暂时注释掉，因为不需要这个功能

#read data
# 读取Excel文件，将所有列作为文本读取
data_raw = read_xlsx('Full_data1.xlsx',na = 'NA',col_types = 'text')

# 打印所有列名，检查是否有经纬度相关列
cat("Available columns in data:\n")
print(names(data_raw))

# Clean Name（清洗列名为下划线风格，自动处理重复列名）
data_raw = data_raw %>% clean_names()

# 检查是否有经纬度列
has_coords <- any(grepl("lat|lng|long|latitude|longitude|easting|northing", 
                        names(data_raw), ignore.case = TRUE))
cat("\nData has coordinate columns:", has_coords, "\n")

# 检查是否有邮编列
has_postcode <- any(grepl("postcode|post_code", names(data_raw), ignore.case = TRUE))
cat("Data has postcode column:", has_postcode, "\n")

# 选择需要的列（包括可能的坐标列）
coord_cols <- names(data_raw)[grepl("lat|lng|long|latitude|longitude|easting|northing|postcode", 
                                     names(data_raw), ignore.case = TRUE)]
cat("Coordinate-related columns found:", paste(coord_cols, collapse = ", "), "\n")

data = data_raw %>%
  # 直接选择并重命名需要的列，避免重复列名问题
  select(
    ccg_code = sub_icb_loc_ods_code,
    ccg_name = sub_icb_loc_name,
    practice_code,
    practice_name,
    prevalence_2020_21,
    prevalence_2021_22,
    year_on_year_change_prevalence,
    achievement_2020_21,
    achievement_2021_22,
    year_on_year_change_achievement,
    pca_rate_2021_22,
    copd008_achievement_score,
    copd008_achievement_net_of_pca,
    copd008_patients_receiving_intervention_percentage,
    # 尝试选择坐标列（如果存在）
    any_of(c("latitude", "longitude", "lat", "lng", "long", 
             "easting", "northing", "postcode", "post_code"))
  ) %>%
  arrange(ccg_code,ccg_name) %>% 
  group_by(ccg_code) %>% 
  fill(ccg_name,.direction = 'down') %>% 
  ungroup() %>% 
  mutate(
    ccg = str_c(ccg_code,ccg_name,sep = '-'),
    # 将数值列转换为正确的类型
    prevalence_2020_21 = as.numeric(prevalence_2020_21),
    prevalence_2021_22 = as.numeric(prevalence_2021_22),
    achievement_2020_21 = as.numeric(achievement_2020_21),
    achievement_2021_22 = as.numeric(achievement_2021_22),
    pca_rate_2021_22 = as.numeric(pca_rate_2021_22),
    copd008_achievement_score = as.numeric(copd008_achievement_score),
    copd008_achievement_net_of_pca = as.numeric(copd008_achievement_net_of_pca),
    copd008_patients_receiving_intervention_percentage = as.numeric(copd008_patients_receiving_intervention_percentage),
    year_on_year_change_prevalence = as.numeric(year_on_year_change_prevalence),
    year_on_year_change_achievement = as.numeric(year_on_year_change_achievement)
  )

# 创建年度变化数据
yoy_change_data <- data %>%
  select(practice_code, practice_name, ccg_code, ccg_name, ccg,
         year_on_year_change_prevalence, year_on_year_change_achievement) %>%
  filter(!is.na(year_on_year_change_prevalence) | !is.na(year_on_year_change_achievement))

# Data for Prevalence Visualization (CCG)（构建 CCG 维度的患病率长表：把 2020/21 与 2021/22 两列压成长格式并标注年份因子。）
prevalence_data = data %>% 
  select(practice_code,
         practice_name,
         ccg_code,
         ccg,
         prevalence_2020_21, 
         prevalence_2021_22) %>% 
  pivot_longer(cols = c(prevalence_2020_21, prevalence_2021_22),
               names_to = 'year') %>% 
  mutate(year = case_when(year=='prevalence_2020_21' ~ '2020-2021',
                          year=='prevalence_2021_22' ~ '2021-2022'),
         year = as.factor(year))

# Data for Achievement Visualization (GP)（构建 GP/Practice 维度的达标率长表：同样把两年达标率列压成长格式并标注年份因子。）
achievement_data = data %>% 
  select(ccg_code,
         ccg,
         practice_code,
         practice_name,
         achievement_2020_21, 
         achievement_2021_22) %>% 
  pivot_longer(cols = c(achievement_2020_21, achievement_2021_22),
               names_to = 'year') %>% 
  mutate(year = case_when(year=='achievement_2020_21' ~ '2020-2021',
                          year=='achievement_2021_22' ~ '2021-2022'),
         year = as.factor(year))

# Data for COPD 008 (GP)（提取 COPD 指标 008 的明细（得分、净得分、接受干预比例）按 practice 与 CCG 输出。）
copd008_data = data %>% 
  select(practice_code,
         ccg_code,
         copd008_achievement_score,
         copd008_achievement_net_of_pca,
         copd008_patients_receiving_intervention_percentage)

# GP Map - 使用新的Sub-ICB Location边界
cat("Loading Sub-ICB Location map data...\n")

# 读取Sub-ICB Location shapefile
sicbl_map <- st_read("ccg map/SICBL_JUL_2022_EN_BFC.shp", quiet = TRUE)

# 转换坐标系统为WGS84 (leaflet需要的格式)
sicbl_map_wgs84 <- st_transform(sicbl_map, crs = 4326)

cat("✓ Coordinate system conversion completed:", st_crs(sicbl_map_wgs84)$input, "\n")

# 检查地图字段和内容
cat("Map fields:", paste(names(sicbl_map_wgs84), collapse = ", "), "\n")
cat("\nMap information for the first 5 regions:\n")
print(as.data.frame(sicbl_map_wgs84[1:5, ]) %>% select(-geometry))

# 尝试从名称中提取ODS代码
cat("\nTrying to extract code from name...\n")
sicbl_map_wgs84 <- sicbl_map_wgs84 %>%
  mutate(
    # 从名称末尾提取ODS代码（格式如 "NHS xxx ICB - 00L" 或 "NHS xxx ICB - M1J4Y"）
    # 支持2-5位的字母数字代码
    ods_code = str_extract(SICBL22NM, "[A-Z0-9]{2,5}$")
  )

cat("Extracted code examples:\n")
print(head(sicbl_map_wgs84 %>% 
  as.data.frame() %>% 
  select(SICBL22NM, ods_code) %>% 
  head(10)
))

# 统计每个Sub-ICB Location的GP数量和平均指标
gp_stats <- data %>%
  group_by(ccg_code, ccg_name) %>%
  summarise(
    n_gp = n(),
    avg_prevalence_2021 = round(mean(prevalence_2021_22, na.rm = TRUE), 2),
    avg_achievement_2021 = round(mean(achievement_2021_22, na.rm = TRUE), 1),
    .groups = 'drop'
  )

cat("Data regions number:", nrow(gp_stats), "\n")
cat("Data ccg_code examples:\n")
print(head(gp_stats %>% select(ccg_code, ccg_name, n_gp), 10))

# 合并地图和数据（尝试通过提取的ODS代码匹配）
# 首先尝试直接匹配
map_data <- sicbl_map_wgs84 %>%
  left_join(gp_stats, by = c("ods_code" = "ccg_code"))

# 如果匹配率太低，尝试通过名称模糊匹配
matched_count <- sum(!is.na(map_data$n_gp))
cat("Direct matching result:", matched_count, "/", nrow(map_data), "regions\n")

# 如果直接匹配失败，尝试通过名称部分匹配
if(matched_count < nrow(map_data) * 0.5) {
  cat("Trying fuzzy name matching...\n")
  
  # 创建一个简化的名称用于匹配
  gp_stats <- gp_stats %>%
    mutate(
      # 从ccg_name中提取关键词用于匹配
      name_key = str_to_upper(str_replace_all(ccg_name, "NHS |ICB.*| - .*", ""))
    )
  
  sicbl_map_wgs84 <- sicbl_map_wgs84 %>%
    mutate(
      name_key = str_to_upper(str_replace_all(SICBL22NM, "NHS |ICB.*| - .*", ""))
    )
  
  # 重新尝试匹配
  map_data <- sicbl_map_wgs84 %>%
    left_join(gp_stats, by = "name_key")
  
  matched_count <- sum(!is.na(map_data$n_gp))
  cat("Fuzzy name matching result:", matched_count, "/", nrow(map_data), "regions\n")
}

cat("✓ Data matching completed - successfully matched:", sum(!is.na(map_data$n_gp)), "/", nrow(map_data), "regions\n")

# 准备GP点数据（用于在地图上显示单个GP位置）
# 检查是否有经纬度数据
gp_points <- NULL
if("latitude" %in% names(data) && "longitude" %in% names(data)) {
  gp_points <- data %>%
    filter(!is.na(latitude) & !is.na(longitude)) %>%
    mutate(
      latitude = as.numeric(latitude),
      longitude = as.numeric(longitude)
    ) %>%
    filter(!is.na(latitude) & !is.na(longitude))
  cat("✓ GP points with coordinates:", nrow(gp_points), "\n")
} else if("easting" %in% names(data) && "northing" %in% names(data)) {
  # 如果有英国国家网格坐标，转换为WGS84
  cat("Converting British National Grid to WGS84...\n")
  gp_points <- data %>%
    filter(!is.na(easting) & !is.na(northing)) %>%
    mutate(
      easting = as.numeric(easting),
      northing = as.numeric(northing)
    ) %>%
    filter(!is.na(easting) & !is.na(northing))
  
  if(nrow(gp_points) > 0) {
    # 转换坐标
    gp_sf <- st_as_sf(gp_points, coords = c("easting", "northing"), crs = 27700)
    gp_sf <- st_transform(gp_sf, crs = 4326)
    coords <- st_coordinates(gp_sf)
    gp_points$longitude <- coords[, 1]
    gp_points$latitude <- coords[, 2]
    cat("✓ GP points converted:", nrow(gp_points), "\n")
  }
} else {
  cat("⚠ No coordinate data found. GP points will use region centroids.\n")
  # 使用区域中心点作为GP的近似位置
  # 计算每个区域的中心点
  region_centroids <- map_data %>%
    st_centroid() %>%
    st_coordinates() %>%
    as.data.frame() %>%
    rename(longitude = X, latitude = Y)
  
  region_centroids$ods_code <- map_data$ods_code
  
  # 为每个GP分配其所属区域的中心点（带随机偏移以避免重叠）
  set.seed(123)
  gp_points <- data %>%
    left_join(region_centroids, by = c("ccg_code" = "ods_code")) %>%
    filter(!is.na(latitude) & !is.na(longitude)) %>%
    mutate(
      # 添加小的随机偏移（约0.01-0.02度，约1-2公里）
      latitude = latitude + runif(n(), -0.015, 0.015),
      longitude = longitude + runif(n(), -0.02, 0.02)
    )
  cat("✓ GP points using region centroids with jitter:", nrow(gp_points), "\n")
}

# 显示匹配成功的示例
if(sum(!is.na(map_data$n_gp)) > 0) {
  matched_sample <- map_data %>% 
    filter(!is.na(n_gp)) %>% 
    select(SICBL22CD, SICBL22NM, n_gp, avg_prevalence_2021) %>%
    head(5)
  cat("Matching examples:\n")
  print(as.data.frame(matched_sample))
}

# 显示未匹配的区域
unmatched <- map_data %>% 
  filter(is.na(n_gp)) %>% 
  select(SICBL22CD, SICBL22NM) %>%
  head(5)
if(nrow(unmatched) > 0) {
  cat("\nUnmatched regions (first 5):\n")
  print(as.data.frame(unmatched))
}

cat("✓ Map data preparation completed!\n")

#UI - 使用紫色主题和现代化设计
ui = dashboardPage(
  header = dashboardHeader(title = 'Healthcare Analytics Dashboard'),
  
  sidebar = dashboardSidebar(
    sidebarMenu(
      menuItem(text = 'Regional Analysis', tabName = 'ccg', icon = icon('hospital')),
      
      selectInput(inputId = 'ccg_ccg_code',
                  label = 'Select Region Codes:',
                  choices = unique(na.omit(data$ccg_code)),
                  selected = c('00L','00N','00P','00Q'),
                  multiple = TRUE),
      
      menuItem(text = 'Practice Performance', tabName = 'gp', icon = icon('user-doctor')),
      
      selectInput(inputId = 'gp_ccgs_code',
                  label = 'Multi-Region Performance Metrics:',
                  choices = unique(na.omit(data$ccg_code)),
                  selected = c('00L','00N','00P','00Q'),
                  multiple = TRUE),
      
      selectInput(inputId = 'gp_ccg_code',
                  label = 'Single Region COPD Quality Indicators:',
                  choices = unique(na.omit(data$ccg_code)),
                  selected = c('00L'),
                  multiple = FALSE),
      
      menuItem(text = 'Year-on-Year Change', tabName = 'yoy', icon = icon('chart-line')),
      menuItem(text = 'Geographic View', tabName = 'map', icon = icon('map-location-dot')),
      menuItem(text = 'Data Explorer', tabName = 'table', icon = icon('database'))
    ),
    collapsed = FALSE),
  
  body = dashboardBody(
    tags$head(tags$style(HTML('
      .content-wrapper { background-color: #f4f6f9; }
      .box { border-top-color: #9b59b6 !important; }
    '))),
    
    tabItems(
      # Regional prevalence analysis
      tabItem(tabName = 'ccg',
              box(
                title = 'Disease Prevalence Distribution',
                status = 'primary',
                solidHeader = TRUE,
                width = 12,
                div(
                  style = "max-height: 700px; overflow-y: auto; overflow-x: hidden;",
                  plotlyOutput(outputId = 'prevalence', height = 'auto')
                )
              )
      ),
      
      # GP Performance metrics
      tabItem(tabName = 'gp',
              fluidRow(
                box(
                  title = 'Performance Metrics Over Time',
                  status = 'warning',
                  solidHeader = TRUE,
                  collapsible = TRUE,
                  width = 12,
                  div(
                    style = "max-height: 600px; overflow-y: auto; overflow-x: hidden;",
                    plotlyOutput(outputId = 'achivement_score', height = 'auto')
                  )
                )
              ),
              fluidRow(
                box(
                  title = 'COPD Quality Indicators',
                  status = 'info',
                  solidHeader = TRUE,
                  collapsible = TRUE,
                  width = 12,
                  div(
                    style = "max-height: 650px; overflow-y: auto; overflow-x: hidden;",
                    plotOutput(outputId = 'copd008', height = '650px')
                  )
                )
              )
      ),
      
      # Year-on-Year Change Analysis
      tabItem(tabName = 'yoy',
              fluidRow(
                box(
                  title = 'Prevalence Year-on-Year Change',
                  status = 'primary',
                  solidHeader = TRUE,
                  collapsible = TRUE,
                  width = 6,
                  div(
                    style = "max-height: 550px; overflow-y: auto; overflow-x: hidden;",
                    plotlyOutput(outputId = 'yoy_prevalence', height = '500px')
                  )
                ),
                box(
                  title = 'Achievement Year-on-Year Change',
                  status = 'success',
                  solidHeader = TRUE,
                  collapsible = TRUE,
                  width = 6,
                  div(
                    style = "max-height: 550px; overflow-y: auto; overflow-x: hidden;",
                    plotlyOutput(outputId = 'yoy_achievement', height = '500px')
                  )
                )
              ),
              fluidRow(
                box(
                  title = 'Top Improvers & Decliners - Prevalence',
                  status = 'info',
                  solidHeader = TRUE,
                  width = 12,
                  div(
                    style = "max-height: 450px; overflow-y: auto; overflow-x: hidden;",
                    plotOutput(outputId = 'yoy_top_bottom', height = '400px')
                  )
                )
              )
      ),
      
      # Geographic distribution
      tabItem(tabName = 'map',
              fluidRow(
                box(
                  title = 'Spatial Distribution of Practices',
                  status = 'danger',
                  solidHeader = TRUE,
                  collapsible = TRUE,
                  width = 12,
                      leafletOutput("gp_map", height = "600px")
                )
              )
      ),
      
      # Data table view
      tabItem(tabName = 'table',
              box(
                title = 'Comprehensive Data Table',
                status = 'primary',
                solidHeader = TRUE,
                width = 12,
                DTOutput(outputId = 'table')
              ))
    )
  ),
  title = 'Healthcare Analytics Dashboard',
  skin = 'purple')

#Server
server = function(input, output){
  
  # Regional prevalence visualization with boxplot
  output$prevalence = renderPlotly({
    filtered_prev_data = prevalence_data %>% 
      filter(ccg_code %in% input$ccg_ccg_code)
    
    # 根据选择的区域数量动态调整列数
    n_regions <- length(unique(filtered_prev_data$ccg))
    ncol_dynamic <- case_when(
      n_regions == 1 ~ 1,
      n_regions == 2 ~ 2,
      n_regions <= 4 ~ 2,
      n_regions <= 6 ~ 3,
      TRUE ~ 4
    )
    
    p = ggplot(data = filtered_prev_data,
               mapping = aes(x = year, y = value)) +
      geom_boxplot(aes(fill = year), 
                   alpha = 0.7, 
                   outlier.shape = NA,
                   width = 0.5,
                   color = "#2c3e50") +
      geom_jitter(aes(color = year, text = paste("Practice:", practice_code)),
                  position = position_jitter(width = 0.25, seed = 456),
                  alpha = 0.7,
                  size = 1.8) +
      stat_summary(fun = median, 
                   geom = "crossbar",
                   width = 0.5,
                   color = "#34495e",
                   size = 0.4,
                   linetype = "solid") +
      facet_wrap(~ccg, ncol = ncol_dynamic) +
      scale_fill_manual(values = c("#9b59b6", "#1abc9c")) +
      scale_color_manual(values = c("#8e44ad", "#16a085")) +
      labs(x = 'Time Period',
           y = 'Prevalence Rate (%)',
           title = 'COPD Prevalence Distribution Across Practices') +
      theme_minimal() +
      theme(
        legend.position = 'top',
        legend.title = element_blank(),
        strip.background = element_rect(fill = "#ecf0f1", color = NA),
        strip.text = element_text(face = "bold", size = 10),
        strip.text.x = element_text(margin = margin(b = 5, t = 5)),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "grey80", fill = NA),
        panel.spacing = unit(1, "lines")
      )
    
    # 根据区域数量和行数动态调整图表高度
    n_rows <- ceiling(n_regions / ncol_dynamic)
    plot_height <- max(350, n_rows * 280)  # 每行约280px高度
    
    ggplotly(p, tooltip = c("text", "y"), height = plot_height) %>%
      layout(margin = list(t = 50))
  })
  
  # Practice achievement trends
  output$achivement_score = renderPlotly({
    filtered_achv_data = achievement_data %>%
      filter(ccg_code %in% input$gp_ccgs_code)
    
    # 根据选择的区域数量动态调整列数
    n_regions <- length(unique(filtered_achv_data$ccg))
    ncol_dynamic <- case_when(
      n_regions == 1 ~ 1,
      n_regions == 2 ~ 2,
      n_regions <= 4 ~ 2,
      n_regions <= 6 ~ 3,
      TRUE ~ 4
    )
    
    p = ggplot(data = filtered_achv_data,
               mapping = aes(x = year, y = value, group = practice_code)) +
      geom_line(aes(text = paste("Practice:", practice_name)), 
                alpha = 0.4, 
                color = "#f39c12",
                size = 0.8) +
      geom_point(aes(text = paste("Practice:", practice_name)),
                 alpha = 0.6,
                 size = 2,
                 color = "#e67e22") +
      stat_summary(fun = median, geom = "line", 
                   aes(group = 1), 
                   color = "#c0392b", 
                   size = 1.5,
                   linetype = "dashed") +
      facet_wrap(~ccg, scales = "free_y", ncol = ncol_dynamic) +
      labs(x = 'Project Period',
           y = 'Achievement Score (%)') +
      theme_minimal() +
      theme(
        strip.background = element_rect(fill = "#fff3cd", color = NA),
        strip.text = element_text(face = "bold", color = "#856404", size = 9),
        strip.text.x = element_text(margin = margin(b = 5, t = 5)),
        panel.grid.major = element_line(color = "#dee2e6"),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(1, "lines")
      )
    
    # 根据区域数量和行数动态调整图表高度
    n_rows <- ceiling(n_regions / ncol_dynamic)
    plot_height <- max(300, n_rows * 250)  # 每行约250px高度
    
    ggplotly(p, tooltip = c("text", "y"), height = plot_height) %>%
      layout(margin = list(t = 30))
  })
  
  # COPD quality indicators
  output$copd008 = renderPlot({
    copd_subset = copd008_data %>%
      filter(ccg_code == input$gp_ccg_code) %>% 
      pivot_longer(cols = -c(practice_code, ccg_code)) %>% 
      mutate(indicator = case_when(
        name == 'copd008_achievement_score' ~ 'Overall Achievement',
        name == 'copd008_achievement_net_of_pca' ~ 'Net Achievement (PCA Adjusted)',
        name == 'copd008_patients_receiving_intervention_percentage' ~ 'Intervention Coverage Rate'
      )) %>% 
      group_by(indicator) %>% 
      slice_min(order_by = value, n = 5) %>% 
      ungroup()
    
    # First metric - gradient purple
    metric1 = ggplot(data = copd_subset %>% 
                       filter(indicator == 'Overall Achievement') %>% 
                       mutate(practice_code = fct_reorder(practice_code, value)),
                     mapping = aes(x = practice_code, y = value, fill = value)) +
      geom_col(width = 0.7) +
      scale_fill_gradient(low = "#d5a6bd", high = "#8e44ad") +
      facet_wrap(~indicator, scales = 'free') +
      labs(x = 'Practice', y = 'Score') +
      coord_flip() +
      theme_minimal() +
      theme(
        legend.position = 'none',
        strip.text = element_text(face = "bold", size = 10),
        panel.grid.major.y = element_blank()
      )
    
    # Second metric - gradient teal
    metric2 = ggplot(data = copd_subset %>% 
                       filter(indicator == 'Net Achievement (PCA Adjusted)') %>% 
                       mutate(practice_code = fct_reorder(practice_code, value)),
                     mapping = aes(x = practice_code, y = value, fill = value)) +
      geom_col(width = 0.7) +
      scale_fill_gradient(low = "#a8e6cf", high = "#16a085") +
      facet_wrap(~indicator, scales = 'free') +
      labs(x = 'Practice', y = 'Score') +
      coord_flip() +
      theme_minimal() +
      theme(
        legend.position = 'none',
        strip.text = element_text(face = "bold", size = 10),
        panel.grid.major.y = element_blank()
      )
    
    # Third metric - gradient coral
    metric3 = ggplot(data = copd_subset %>% 
                       filter(indicator == 'Intervention Coverage Rate') %>% 
                       mutate(practice_code = fct_reorder(practice_code, value)),
                     mapping = aes(x = practice_code, y = value, fill = value)) +
      geom_col(width = 0.7) +
      scale_fill_gradient(low = "#ffd3b6", high = "#e67e22") +
      facet_wrap(~indicator, scales = 'free') +
      labs(x = 'Practice', y = 'Percentage') +
      coord_flip() +
      theme_minimal() +
      theme(
        legend.position = 'none',
        strip.text = element_text(face = "bold", size = 10),
        panel.grid.major.y = element_blank()
      )
    
    metric1 | metric2 | metric3
  })
  
  # Year-on-Year Prevalence Change - 只显示前20个变化最大的
  output$yoy_prevalence = renderPlotly({
    yoy_data <- yoy_change_data %>%
      filter(ccg_code %in% input$ccg_ccg_code) %>%
      filter(!is.na(year_on_year_change_prevalence)) %>%
      arrange(desc(abs(year_on_year_change_prevalence))) %>%
      head(20)  # 只显示变化最大的前20个
    
    p <- ggplot(yoy_data, aes(x = year_on_year_change_prevalence, 
                              y = reorder(practice_name, year_on_year_change_prevalence),
                              fill = year_on_year_change_prevalence,
                              text = paste("Practice:", practice_name,
                                          "<br>Change:", round(year_on_year_change_prevalence, 2), "%"))) +
      geom_col(width = 0.7) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray40", size = 0.8) +
      scale_fill_viridis_c(option = "D", name = "Change %") +
      labs(x = "Year-on-Year Change (%)", y = NULL,
           title = "Top 20 Prevalence Changes (2020-21 to 2021-22)") +
      theme_minimal() +
      theme(
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 11)
      )
    
    ggplotly(p, tooltip = "text") %>%
      layout(title = list(text = "Top 20 Prevalence Changes (2020-21 to 2021-22)", 
                          font = list(size = 14, face = "bold"),
                          y = 0.98),
             margin = list(t = 80))
  })
  
  # Year-on-Year Achievement Change - 只显示前20个变化最大的
  output$yoy_achievement = renderPlotly({
    yoy_data <- yoy_change_data %>%
      filter(ccg_code %in% input$ccg_ccg_code) %>%
      filter(!is.na(year_on_year_change_achievement)) %>%
      arrange(desc(abs(year_on_year_change_achievement))) %>%
      head(20)  # 只显示变化最大的前20个
    
    p <- ggplot(yoy_data, aes(x = year_on_year_change_achievement, 
                              y = reorder(practice_name, year_on_year_change_achievement),
                              fill = year_on_year_change_achievement,
                              text = paste("Practice:", practice_name,
                                          "<br>Change:", round(year_on_year_change_achievement, 2), "%"))) +
      geom_col(width = 0.7) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray40", size = 0.8) +
      scale_fill_viridis_c(option = "C", name = "Change %") +
      labs(x = "Year-on-Year Change (%)", y = NULL,
           title = "Top 20 Achievement Changes (2020-21 to 2021-22)") +
      theme_minimal() +
      theme(
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(size = 9),
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 11)
      )
    
    ggplotly(p, tooltip = "text") %>%
      layout(title = list(text = "Top 20 Achievement Changes (2020-21 to 2021-22)", 
                          font = list(size = 14, face = "bold"),
                          y = 0.98),
             margin = list(t = 80))
  })
  
  # Top Improvers and Decliners
  output$yoy_top_bottom = renderPlot({
    yoy_data <- yoy_change_data %>%
      filter(ccg_code %in% input$ccg_ccg_code) %>%
      filter(!is.na(year_on_year_change_prevalence))
    
    # Top 5 improvers (biggest decrease in prevalence is good)
    top_improvers <- yoy_data %>%
      slice_min(order_by = year_on_year_change_prevalence, n = 5) %>%
      mutate(category = "Top Improvers\n(Decreased Prevalence)")
    
    # Top 5 decliners (biggest increase in prevalence is concerning)
    top_decliners <- yoy_data %>%
      slice_max(order_by = year_on_year_change_prevalence, n = 5) %>%
      mutate(category = "Needs Attention\n(Increased Prevalence)")
    
    p1 <- ggplot(top_improvers, 
                 aes(x = reorder(practice_name, -year_on_year_change_prevalence), 
                     y = year_on_year_change_prevalence,
                     fill = year_on_year_change_prevalence)) +
      geom_col(width = 0.7) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      scale_fill_viridis_c(option = "D", direction = -1) +
      labs(x = NULL, y = "Change (%)", 
           title = "🏆 Top Improvers (Decreased Prevalence)") +
      coord_flip() +
      theme_minimal() +
      theme(legend.position = "none",
            plot.title = element_text(face = "bold", color = "#2e7d32"))
    
    p2 <- ggplot(top_decliners, 
                 aes(x = reorder(practice_name, year_on_year_change_prevalence), 
                     y = year_on_year_change_prevalence,
                     fill = year_on_year_change_prevalence)) +
      geom_col(width = 0.7) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      scale_fill_viridis_c(option = "A", direction = 1) +
      labs(x = NULL, y = "Change (%)", 
           title = "⚠️ Needs Attention (Increased Prevalence)") +
      coord_flip() +
      theme_minimal() +
      theme(legend.position = "none",
            plot.title = element_text(face = "bold", color = "#c62828"))
    
    p1 | p2
  })
  
  # Table - 包含诊所名称和年度变化
  output$table = renderDT({
    data %>% 
      select(
        `Practice Code` = practice_code,
        `Practice Name` = practice_name,
        `Region Code` = ccg_code,
        `Region Name` = ccg_name,
        `Prevalence 2021-22` = prevalence_2021_22,
        `Prevalence YoY Change` = year_on_year_change_prevalence,
        `Achievement 2021-22` = achievement_2021_22,
        `Achievement YoY Change` = year_on_year_change_achievement,
        `PCA Rate 2021-22` = pca_rate_2021_22,
        `COPD008 Score` = copd008_achievement_score,
        `COPD008 Net Score` = copd008_achievement_net_of_pca,
        `Intervention %` = copd008_patients_receiving_intervention_percentage
      ) %>% 
      datatable(
        filter = 'top',
        options = list(
          lengthMenu = c('5','10','15','20','30'),
          lengthChange = TRUE,
          pageLength = 10,
          autoWidth = FALSE,
          scrollX = TRUE,
          columnDefs = list(
            list(className = 'dt-center', targets = '_all')
          )
        ),
        rownames = FALSE
      ) %>%
      formatRound(columns = c('Prevalence 2021-22', 'Prevalence YoY Change', 
                              'Achievement 2021-22', 'Achievement YoY Change',
                              'PCA Rate 2021-22', 'COPD008 Score', 
                              'COPD008 Net Score', 'Intervention %'), 
                  digits = 2)
  })
  
  # Geographic map - Sub-ICB Location 交互式地图
  output$gp_map = renderLeaflet({
    # 创建区域颜色调色板（黄-橙-红渐变，基于GP数量）
    pal_region <- colorNumeric(
      palette = "YlOrRd",
      domain = map_data$n_gp,
      na.color = "#e0e0e0"
    )
    
    # 创建GP点颜色调色板（基于患病率）
    pal_gp <- colorNumeric(
      palette = "RdYlGn",  # 红-黄-绿，低患病率为绿色
      domain = as.numeric(gp_points$prevalence_2021_22),
      na.color = "#808080",
      reverse = TRUE  # 反转：低值绿色，高值红色
    )
    
    # 创建区域信息弹窗
    region_labels <- sprintf(
      "<div style='font-size:13px;'>
        <strong style='font-size:14px; color:#2c3e50;'>%s</strong><br/>
        <hr style='margin:5px 0;'/>
        <b>Region Code:</b> %s<br/>
        <b>GP Practices:</b> <span style='color:#e74c3c; font-weight:bold;'>%s</span><br/>
        <b>Avg Prevalence 2021-22:</b> %s%%<br/>
        <b>Avg Achievement 2021-22:</b> %s%%
      </div>",
      map_data$SICBL22NM,
      map_data$SICBL22CD,
      ifelse(is.na(map_data$n_gp), "N/A", as.character(map_data$n_gp)),
      ifelse(is.na(map_data$avg_prevalence_2021), "N/A", 
             sprintf("%.2f", map_data$avg_prevalence_2021)),
      ifelse(is.na(map_data$avg_achievement_2021), "N/A", 
             sprintf("%.1f", map_data$avg_achievement_2021))
    ) %>% lapply(htmltools::HTML)
    
    # 创建GP点信息弹窗
    gp_labels <- sprintf(
      "<div style='font-size:12px; min-width:200px;'>
        <strong style='font-size:13px; color:#9b59b6;'>%s</strong><br/>
        <span style='color:#666; font-size:11px;'>%s</span>
        <hr style='margin:5px 0;'/>
        <b>Practice Code:</b> %s<br/>
        <b>Region:</b> %s<br/>
        <b>Prevalence 2021-22:</b> <span style='color:#e74c3c;'>%s%%</span><br/>
        <b>Achievement 2021-22:</b> <span style='color:#27ae60;'>%s%%</span>
      </div>",
      gp_points$practice_name,
      gp_points$ccg_name,
      gp_points$practice_code,
      gp_points$ccg_code,
      ifelse(is.na(gp_points$prevalence_2021_22), "N/A", 
             sprintf("%.2f", as.numeric(gp_points$prevalence_2021_22))),
      ifelse(is.na(gp_points$achievement_2021_22), "N/A", 
             sprintf("%.1f", as.numeric(gp_points$achievement_2021_22)))
    ) %>% lapply(htmltools::HTML)
    
    # 创建leaflet地图
    leaflet() %>%
      addProviderTiles(providers$CartoDB.Positron, group = "Light") %>%
      addProviderTiles(providers$OpenStreetMap, group = "Street") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satellite") %>%
      
      # 添加区域多边形图层
      addPolygons(
        data = map_data,
        fillColor = ~pal_region(n_gp),
        weight = 1.5,
        opacity = 1,
        color = "#ffffff",
        dashArray = "2",
        fillOpacity = 0.6,
        highlightOptions = highlightOptions(
          weight = 3,
          color = "#ff6b6b",
          dashArray = "",
          fillOpacity = 0.8,
          bringToFront = FALSE
        ),
        label = region_labels,
        labelOptions = labelOptions(
          style = list(
            "font-weight" = "normal",
            "padding" = "8px 12px",
            "border-radius" = "4px",
            "box-shadow" = "0 2px 6px rgba(0,0,0,0.3)"
          ),
          textsize = "13px",
          direction = "auto"
        ),
        group = "Regions"
      ) %>%
      
      # 添加GP点标记图层
      addCircleMarkers(
        data = gp_points,
        lng = ~longitude,
        lat = ~latitude,
        radius = 5,
        color = "#2c3e50",
        weight = 1,
        fillColor = ~pal_gp(as.numeric(prevalence_2021_22)),
        fillOpacity = 0.8,
        popup = gp_labels,
        label = ~practice_name,
        labelOptions = labelOptions(
          style = list("font-size" = "11px"),
          direction = "auto"
        ),
        clusterOptions = markerClusterOptions(
          spiderfyOnMaxZoom = TRUE,
          showCoverageOnHover = TRUE,
          zoomToBoundsOnClick = TRUE,
          maxClusterRadius = 50
        ),
        group = "GP Practices"
      ) %>%
      
      # 添加图层控制器
      addLayersControl(
        baseGroups = c("Light", "Street", "Satellite"),
        overlayGroups = c("Regions", "GP Practices"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      
      # 区域图例
      addLegend(
        pal = pal_region,
        values = map_data$n_gp,
        opacity = 0.7,
        title = "GP Count<br/>per Region",
        position = "bottomright",
        labFormat = labelFormat(digits = 0)
      ) %>%
      
      # GP患病率图例
      addLegend(
        pal = pal_gp,
        values = as.numeric(gp_points$prevalence_2021_22),
        opacity = 0.7,
        title = "GP Prevalence<br/>Rate (%)",
        position = "bottomleft",
        labFormat = labelFormat(digits = 1)
      ) %>%
      
      addControl(
        html = paste0(
          '<div style="background:white; padding:10px; border-radius:5px; box-shadow:0 2px 6px rgba(0,0,0,0.2);">',
          '<h5 style="margin:0 0 5px 0; color:#9b59b6;">🗺️ NHS Healthcare Map</h5>',
          '<p style="margin:0; font-size:11px; color:#666;">',
          '<b>Regions:</b> Colored by GP count<br/>',
          '<b>Points:</b> Individual GP practices<br/>',
          'Click points for details | Use layer control<br/>',
          'Data: July 2022 NHS boundaries</p>',
          '</div>'
        ),
        position = "topleft"
      ) %>%
      setView(lng = -2, lat = 53.5, zoom = 6.2)
  })
}

# 启动Shiny应用并自动打开浏览器
cat("\nstarting Shiny application...\n")
app <- shinyApp(ui = ui, server = server)
runApp(app, launch.browser = TRUE)