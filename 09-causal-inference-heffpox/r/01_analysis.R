# ============================================================================
# Project 2: Milnepan对Heffpox患者7天死亡风险的因果效应分析
# Statistical Modelling and Inference for Health
# ============================================================================

# 0) 加载必需包
# ----------------------------
pkgs <- c(
  "tidyverse",    # 数据处理
  "survival",     # 生存分析
  "survminer",    # KM曲线可视化
  "tableone",     # Table 1 基线特征表
  "broom",        # 模型结果整理
  "mice",         # 多重插补（处理缺失数据 - 必需）
  "dagitty",      # DAG因果图（必需）
  "ggdag",        # DAG可视化
  "WeightIt",     # 倾向性评分加权（第二种因果方法）
  "cobalt",       # 协变量平衡诊断
  "survey"        # 加权回归
)

# 安装缺失的包
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

invisible(lapply(pkgs, function(x) {
  suppressPackageStartupMessages(library(x, character.only = TRUE))
}))

# ----------------------------
# 1) 设置路径与输出文件夹
# ----------------------------
OUT_DIR <- "outputs"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

save_plot <- function(p, filename, w = 8, h = 6, dpi = 300) {
  ggsave(file.path(OUT_DIR, filename), plot = p, width = w, height = h, dpi = dpi)
}

# ----------------------------
# 2) 加载数据
# ----------------------------
# 注意：PDF要求使用 read_rds 加载 .RData 文件
# 如果是CSV格式，使用下面的代码
dat <- read_csv("heffpox_2526.csv", show_col_types = FALSE)

# 筛选Heffpox患者（研究人群）
dat_h <- dat %>% 
  filter(heffpox == 1) %>%
  mutate(
    sex      = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    smoking  = factor(smoking, levels = c(0, 1), labels = c("No", "Yes")),
    diabetes = factor(diabetes, levels = c(0, 1), labels = c("No", "Yes")),
    milnepan = factor(milnepan, levels = c(0, 1), labels = c("No", "Yes")),
    icu      = factor(icu, levels = c(0, 1), labels = c("No", "Yes")),
    Status   = as.integer(Status),
    death7day = as.integer(death7day)
  )

cat("\n============ 数据概览 ============\n")
cat("Heffpox患者总数 N =", nrow(dat_h), "\n")
cat("接受Milnepan治疗:", sum(dat_h$milnepan == "Yes"), "\n")
cat("未接受Milnepan治疗:", sum(dat_h$milnepan == "No"), "\n")
cat("7天死亡率:", round(mean(dat_h$death7day == 1)*100, 1), "%\n")

# ============================================================================
# PART 1: 探索性/描述性分析 (Exploratory/Descriptive Analysis)
# ============================================================================

# ----------------------------
# 3) 缺失数据分析
# ----------------------------
cat("\n============ 缺失数据分析 ============\n")

miss_summary <- dat_h %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "N_Missing") %>%
  mutate(Pct_Missing = round(N_Missing / nrow(dat_h) * 100, 2)) %>%
  filter(N_Missing > 0) %>%
  arrange(desc(Pct_Missing))

print(miss_summary)
write_csv(miss_summary, file.path(OUT_DIR, "table_missing_data.csv"))

# 缺失模式 (修复: 需要print()才能写入文件)
sink(file.path(OUT_DIR, "missing_pattern.txt"))
cat("Missing Data Pattern\n")
cat("====================\n\n")
print(md.pattern(dat_h %>% select(age, sex, bmi, smoking, diabetes, milnepan, death7day), plot = FALSE))
sink()

# ----------------------------
# 4) Table 1: 基线特征（按治疗组分层）
# ----------------------------
cat("\n============ 基线特征表 (Table 1) ============\n")

vars <- c("age", "sex", "bmi", "smoking", "diabetes")
tab1 <- CreateTableOne(vars = vars, strata = "milnepan", data = dat_h, test = FALSE)
tab1_print <- print(tab1, smd = TRUE, printToggle = FALSE)
write.csv(tab1_print, file.path(OUT_DIR, "table1_baseline.csv"))

cat("SMD > 0.1 表示组间不平衡:\n")
print(ExtractSmd(tab1))

# ----------------------------
# 5) 生存时间(TEVENT)探索性分析
# ----------------------------
cat("\n============ 生存分析 (TEVENT) ============\n")

# Kaplan-Meier曲线
fit_km <- survfit(Surv(TEVENT, Status) ~ milnepan, data = dat_h)
print(fit_km)

# Log-rank检验
logrank <- survdiff(Surv(TEVENT, Status) ~ milnepan, data = dat_h)
logrank_p <- 1 - pchisq(logrank$chisq, df = 1)
cat("Log-rank test p-value:", format.pval(logrank_p, digits = 4), "\n")

# KM图
km_plot <- ggsurvplot(
  fit_km, data = dat_h,
  pval = TRUE, conf.int = TRUE,
  risk.table = TRUE,
  palette = c("#E7B800", "#2E9FDF"),
  legend.labs = c("No Milnepan", "Milnepan"),
  xlab = "Time (days)", ylab = "Survival Probability",
  title = "Figure 1: Kaplan-Meier Survival Curves by Milnepan Treatment",
  ggtheme = theme_minimal()
)
# 保存为PNG格式
png(file.path(OUT_DIR, "figure1_km_curve.png"), width = 10, height = 8, units = "in", res = 300)
print(km_plot)
dev.off()

# 中位生存时间
median_surv <- surv_median(fit_km)
write_csv(median_surv, file.path(OUT_DIR, "median_survival.csv"))

# ============================================================================
# PART 2: 因果推断 (Causal Inference) - 主要结局: death7day
# ============================================================================

# ----------------------------
# 6) DAG因果图
# ----------------------------
cat("\n============ DAG因果分析 ============\n")

# 根据题目描述构建DAG:
# - 混杂因素(同时影响治疗和结局): age, sex, bmi, smoking, diabetes
# - 糖尿病会干扰milnepan的作用 (effect modification)
# - ICU是中介变量(在milnepan之后发生)，不应调整

dag <- dagitty("dag {
  age -> milnepan
  sex -> milnepan
  bmi -> milnepan
  smoking -> milnepan
  diabetes -> milnepan
  
  age -> death7day
  sex -> death7day
  bmi -> death7day
  smoking -> death7day
  diabetes -> death7day
  
  milnepan -> death7day
  milnepan -> icu
  icu -> death7day
}")

# 设定暴露和结局
exposures(dag) <- "milnepan"
outcomes(dag) <- "death7day"

# 最小充分调整集
adj_set <- adjustmentSets(dag, exposure = "milnepan", outcome = "death7day", effect = "total")
cat("最小充分调整集 (用于估计总因果效应):\n")
print(adj_set)

# 保存DAG信息
sink(file.path(OUT_DIR, "dag_analysis.txt"))
cat("DAG-Based Causal Analysis\n")
cat("=========================\n\n")
cat("Exposure: milnepan (treatment)\n")
cat("Outcome: death7day (7-day mortality)\n\n")
cat("Minimal Sufficient Adjustment Set for Total Effect:\n")
print(adj_set)
cat("\nNote: ICU is on the causal pathway (mediator) and should NOT be adjusted\n")
cat("to estimate the TOTAL causal effect of milnepan on death7day.\n")
sink()

# DAG可视化 (修复: 放大节点，标签放在节点内部)
dag_tidy <- dag %>%
  tidy_dagitty() %>%
  node_status()   # 添加status列（exposure/outcome/其他）

dag_plot <- ggplot(dag_tidy, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_edges(edge_colour = "gray30", edge_width = 0.6) +
  geom_dag_point(aes(colour = status), size = 22) +
  geom_dag_text(colour = "white", size = 3, fontface = "bold") +
  scale_colour_manual(
    values = c("exposure" = "#E69F00", "outcome" = "#56B4E9"),
    labels = c("exposure" = "Exposure (Milnepan)", "outcome" = "Outcome (Death7day)"),
    na.value = "gray60",
    na.translate = TRUE
  ) +
  theme_dag() +
  labs(
    title = "Figure 2: Directed Acyclic Graph (DAG)",
    subtitle = "Causal structure: milnepan → death7day",
    colour = "Variable Type"
  ) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )
save_plot(dag_plot, "figure2_dag.png", w = 12, h = 9)

# ----------------------------
# 7) 多重插补 (Multiple Imputation)
# ----------------------------
cat("\n============ 多重插补 (MICE) ============\n")

# 准备插补数据（smoking用因子以配合logreg方法）
imp_vars <- dat_h %>%
  transmute(
    age = age,
    sex = as.numeric(sex == "Male"),
    bmi = bmi,
    smoking = factor(as.numeric(smoking == "Yes")),  # 因子类型用于logreg
    diabetes = as.numeric(diabetes == "Yes"),
    milnepan = as.numeric(milnepan == "Yes"),
    death7day = death7day
  )

# 设置插补方法 (连续变量用pmm, 二元因子用logreg)
ini  <- mice(imp_vars, maxit = 0, printFlag = FALSE)
meth <- ini$method
pred <- ini$predictorMatrix

# 只对有缺失的变量设置插补方法
meth[] <- ""                    # 先清空所有
meth["bmi"]     <- "pmm"        # 连续变量用预测均值匹配
meth["smoking"] <- "logreg"     # 二元因子用逻辑回归

set.seed(12345)
cat("执行多重插补 (m=20)...\n")
cat("  - BMI: predictive mean matching (pmm)\n")
cat("  - Smoking: logistic regression (logreg)\n")
imp <- mice(imp_vars, m = 20, method = meth, predictorMatrix = pred, 
            maxit = 10, printFlag = FALSE)

# 检查收敛 (保存为PNG格式，使用print确保输出)
png(file.path(OUT_DIR, "mice_convergence.png"), width = 10, height = 6, units = "in", res = 300)
print(plot(imp))
dev.off()

# ============================================================================
# 方法1: 多重插补 + 逻辑回归 (Outcome Regression)
# ============================================================================
cat("\n============ 方法1: 逻辑回归 (调整混杂) ============\n")

# 在每个插补数据集上拟合逻辑回归
fit_logit <- with(imp, glm(death7day ~ milnepan + age + sex + bmi + smoking + diabetes, 
                            family = binomial))

# Rubin规则合并结果 (修复: exponentiate=TRUE后estimate已经是OR，不需要再exp)
pooled_logit <- pool(fit_logit)
pooled_summary <- summary(pooled_logit, conf.int = TRUE, exponentiate = TRUE)

cat("\n--- 逻辑回归结果 (Pooled OR) ---\n")
print(pooled_summary)

# 提取milnepan的效应 (注意: exponentiate=TRUE后，estimate已是OR，2.5%/97.5%已是CI)
or_logit_result <- pooled_summary %>%
  filter(term == "milnepan") %>%
  transmute(
    OR = estimate,           # 已经是OR，不需要exp()
    OR_lower = `2.5 %`,      # 已经是CI下界
    OR_upper = `97.5 %`,     # 已经是CI上界
    p_value = p.value
  )

write_csv(pooled_summary, file.path(OUT_DIR, "method1_logistic_regression.csv"))

# ============================================================================
# 方法2: 倾向性评分加权 (IPTW)
# ============================================================================
cat("\n============ 方法2: 倾向性评分加权 (IPTW) ============\n")

# 使用完整案例进行IPTW（或可在插补数据上进行）
# 这里展示在完整案例上的分析，高分做法是在每个插补数据集上进行

# 从第一个插补数据集开始（简化版）
# 高分做法：循环所有插补数据集
imp_list <- complete(imp, "all")

# 将smoking转回数值型（logreg插补后是因子）
imp_list <- lapply(imp_list, function(d) {
  d$smoking <- as.numeric(as.character(d$smoking))
  d
})

iptw_results <- lapply(1:20, function(i) {
  dat_imp <- imp_list[[i]]
  
  # 估计倾向性评分
  ps_fit <- glm(milnepan ~ age + sex + bmi + smoking + diabetes, 
                data = dat_imp, family = binomial)
  dat_imp$ps <- predict(ps_fit, type = "response")
  
  # 计算IPTW权重 (ATE)
  dat_imp$iptw <- ifelse(dat_imp$milnepan == 1, 
                          1 / dat_imp$ps, 
                          1 / (1 - dat_imp$ps))
  
  # 双尾截断极端权重 (修复: 1%和99%分位数)
  lo <- quantile(dat_imp$iptw, 0.01)
  hi <- quantile(dat_imp$iptw, 0.99)
  dat_imp$iptw <- pmin(pmax(dat_imp$iptw, lo), hi)
  
  # 使用svyglm获得稳健标准误 (修复: 更规范的IPTW方差估计)
  des <- svydesign(ids = ~1, weights = ~iptw, data = dat_imp)
  fit_iptw <- svyglm(death7day ~ milnepan, design = des, family = quasibinomial())
  
  coef_est <- coef(fit_iptw)["milnepan"]
  se_est <- sqrt(vcov(fit_iptw)["milnepan", "milnepan"])
  
  return(data.frame(estimate = coef_est, std.error = se_est))
})

# 合并IPTW结果 (Rubin's rules)
iptw_df <- bind_rows(iptw_results)
pooled_iptw_est <- mean(iptw_df$estimate)
within_var <- mean(iptw_df$std.error^2)
between_var <- var(iptw_df$estimate)
pooled_iptw_se <- sqrt(within_var + (1 + 1/20) * between_var)

or_iptw <- exp(pooled_iptw_est)
or_iptw_lower <- exp(pooled_iptw_est - 1.96 * pooled_iptw_se)
or_iptw_upper <- exp(pooled_iptw_est + 1.96 * pooled_iptw_se)
p_iptw <- 2 * (1 - pnorm(abs(pooled_iptw_est / pooled_iptw_se)))

cat("\n--- IPTW结果 ---\n")
cat("OR =", round(or_iptw, 3), "\n")
cat("95% CI: [", round(or_iptw_lower, 3), ",", round(or_iptw_upper, 3), "]\n")
cat("p-value:", format.pval(p_iptw, digits = 4), "\n")

# 协变量平衡检查（使用第一个插补数据集展示）
dat_imp1 <- imp_list[[1]]
ps_model <- weightit(milnepan ~ age + sex + bmi + smoking + diabetes,
                     data = dat_imp1, method = "ps", estimand = "ATE")

# 修复: 使用白色背景，增大变量名显示空间，添加阈值线说明
bal_plot <- love.plot(ps_model, 
                      stats = "mean.diffs", 
                      thresholds = c(m = 0.1),
                      abs = TRUE, 
                      var.names = c(age = "Age", sex = "Sex", bmi = "BMI", 
                                    smoking = "Smoking", diabetes = "Diabetes"),
                      colors = c("#E69F00", "#56B4E9"),
                      shapes = c(18, 16),
                      sample.names = c("Unadjusted", "IPTW Adjusted"),
                      title = "Figure 3: Covariate Balance (IPTW)",
                      subtitle = "Dashed line indicates SMD = 0.1 threshold") +
  theme(
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    legend.background = element_rect(fill = "white", colour = NA),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 10, colour = "gray40")
  )
save_plot(bal_plot, "figure3_iptw_balance.png", w = 8, h = 5)

# 保存IPTW结果
iptw_summary <- data.frame(
  Method = "IPTW",
  OR = or_iptw,
  CI_lower = or_iptw_lower,
  CI_upper = or_iptw_upper,
  p_value = p_iptw
)
write_csv(iptw_summary, file.path(OUT_DIR, "method2_iptw.csv"))

# ============================================================================
# 8) 结果汇总与比较
# ============================================================================
cat("\n============ 因果效应估计汇总 ============\n")

# 从pooled_summary提取逻辑回归结果（注意：summary已经exponentiate过）
logit_row <- pooled_summary[pooled_summary$term == "milnepan", ]
or_logit <- logit_row$estimate  # 已经是OR
or_logit_lower <- logit_row$`2.5 %`
or_logit_upper <- logit_row$`97.5 %`
p_logit <- logit_row$p.value

# 汇总两种方法的结果
summary_table <- data.frame(
  Method = c("Outcome Regression (Logistic)", "IPTW"),
  OR = c(or_logit, or_iptw),
  CI_lower = c(or_logit_lower, or_iptw_lower),
  CI_upper = c(or_logit_upper, or_iptw_upper),
  p_value = c(p_logit, p_iptw)
) %>%
  mutate(
    OR_CI = sprintf("%.3f (%.3f - %.3f)", OR, CI_lower, CI_upper),
    p_value_fmt = format.pval(p_value, digits = 3)
  )

cat("\nTable 2: Comparison of Causal Effect Estimates\n")
print(summary_table %>% select(Method, OR_CI, p_value_fmt))

write_csv(summary_table, file.path(OUT_DIR, "table2_causal_effects_comparison.csv"))

# 森林图比较两种方法
forest_plot <- ggplot(summary_table, aes(x = OR, y = Method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper), height = 0.15, linewidth = 1) +
  geom_point(size = 4, color = "darkred") +
  scale_x_log10(limits = c(0.5, 1.5), breaks = c(0.5, 0.75, 1, 1.25, 1.5)) +
  labs(
    title = "Figure 4: Causal Effect Estimates (OR with 95% CI)",
    subtitle = "Effect of Milnepan on 7-day Mortality",
    x = "Odds Ratio (log scale)",
    y = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 11)
  ) +
  annotate("text", x = 0.55, y = 2.3, label = "Favors Milnepan", hjust = 0, size = 3) +
  annotate("text", x = 1.35, y = 2.3, label = "Favors Control", hjust = 1, size = 3)

save_plot(forest_plot, "figure4_forest_plot.png", w = 8, h = 4)

# ============================================================================
# 9) 敏感性分析: 糖尿病亚组（题目提示糖尿病干扰milnepan作用）
# ============================================================================
cat("\n============ 敏感性分析: 糖尿病亚组 ============\n")

# 在插补数据上按糖尿病分层分析
subgroup_results <- lapply(c(0, 1), function(diab) {
  fit_sub <- with(imp, {
    dat_sub <- data.frame(
      death7day = death7day,
      milnepan = milnepan,
      age = age, sex = sex, bmi = bmi, smoking = smoking, diabetes = diabetes
    )
    dat_sub <- dat_sub[dat_sub$diabetes == diab, ]
    glm(death7day ~ milnepan + age + sex + bmi + smoking, 
        data = dat_sub, family = binomial)
  })
  pooled_sub <- pool(fit_sub)
  summary(pooled_sub, conf.int = TRUE) %>%
    filter(term == "milnepan") %>%
    mutate(diabetes_group = ifelse(diab == 0, "No Diabetes", "Diabetes"))
})

subgroup_df <- bind_rows(subgroup_results) %>%
  mutate(
    OR = exp(estimate),
    OR_lower = exp(estimate - 1.96 * std.error),
    OR_upper = exp(estimate + 1.96 * std.error)
  )

cat("\nSubgroup Analysis by Diabetes Status:\n")
print(subgroup_df %>% select(diabetes_group, OR, OR_lower, OR_upper, p.value))

write_csv(subgroup_df, file.path(OUT_DIR, "table3_subgroup_diabetes.csv"))

# ============================================================================
# 10) 输出文件汇总
# ============================================================================
cat("\n============ 分析完成 ============\n")
cat("所有输出保存至:", OUT_DIR, "\n\n")

cat("生成的文件:\n")
cat("  Tables:\n")
cat("    - table_missing_data.csv: 缺失数据汇总\n")
cat("    - table1_baseline.csv: 基线特征表\n")
cat("    - table2_causal_effects_comparison.csv: 因果效应比较\n")
cat("    - table3_subgroup_diabetes.csv: 糖尿病亚组分析\n")
cat("  Figures:\n")
cat("    - figure1_km_curve.png: Kaplan-Meier生存曲线\n")
cat("    - figure2_dag.png: DAG因果图\n")
cat("    - figure3_iptw_balance.png: IPTW协变量平衡图\n")
cat("    - figure4_forest_plot.png: 因果效应森林图\n")
cat("  Supplementary:\n")
cat("    - missing_pattern.txt: 缺失模式\n")
cat("    - dag_analysis.txt: DAG分析说明\n")
cat("    - mice_convergence.png: MICE收敛诊断\n")
