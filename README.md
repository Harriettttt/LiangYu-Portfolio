# Liang Yu — Algorithm & Health Data Science Portfolio

面向算法 / 健康数据科学岗位的代码作品集。每个编号目录为独立项目，可直接打开源码与结果图。

> 大数据集与模型权重未入库（见各项目 `data/.gitkeep` 与根目录 `.gitignore`）。仓库体积已控制在可网页一键上传的范围。

**Author:** Liang Yu

---

## Projects Overview

### A. Computer Vision & Deep Learning

| # | Project | Highlights |
|---|---------|------------|
| 01 | [semantic-segmentation-camvid](01-semantic-segmentation-camvid) | U-Net 街景分割；sunny→cloudy 域微调；小目标 object-level F1 |
| 02 | [reinforcement-learning-agents](02-reinforcement-learning-agents) | 从零实现 DQN / A2C / PPO（Atari + LunarLander） |
| 05 | [classic-ml-classifiers](05-classic-ml-classifiers) | PaviaU 高光谱地物分类：KNN / RF / Gaussian NB |
| 07 | [pathmnist-cnn-classification](07-pathmnist-cnn-classification) | PathMNIST 病理组织分类：CNN vs 逻辑回归基线 |
| 08 | [multimodal-ai-healthcare](08-multimodal-ai-healthcare) | SpaGCN 空间转录组聚类 + PathVQA 多模态视觉问答 |

### B. NLP & Classical ML

| # | Project | Highlights |
|---|---------|------------|
| 03 | [text-classification-benchmark](03-text-classification-benchmark) | 新闻 / 谣言 / 垃圾短信：CNN · XGBoost · Random Forest |
| 04 | [document-clustering](04-document-clustering) | TF-IDF + KMeans / DBSCAN / GMM + t-SNE |

### C. Causal Inference & Biostatistics

| # | Project | Highlights |
|---|---------|------------|
| 09 | [causal-inference-heffpox](09-causal-inference-heffpox) | 多重插补 + 倾向评分 / IPTW；R 与 Python 双实现 |
| 10 | [ra-pain-longitudinal-modeling](10-ra-pain-longitudinal-modeling) | 类风湿疼痛纵向混合效应模型（lme4） |

### D. Data Systems & Analytics Apps

| # | Project | Highlights |
|---|---------|------------|
| 06 | [auto-retail-database](06-auto-retail-database) | 汽车零售库 DDL / 种子数据 / 分析查询 |
| 11 | [nhs-copd-dashboard](11-nhs-copd-dashboard) | NHS COPD QoF 数据处理 + 可视化 + Shiny 仪表盘 |

---

## Project Notes (brief)

### 01 Semantic Segmentation (CamVid)
PyTorch U-Net：全局精度 / mIoU、损失与微调策略对比、小目标增强。代码在 `src/`，实验在 `notebooks/`，结果图在 `results/`。

### 02 Reinforcement Learning
`src/dqn.py` · `a2c.py` · `ppo.py`；`notebooks/` 含 LunarLander 实验与 Stable-Baselines3 对照。

### 03 Text Classification
`src/{news,rumors,spam}/` 下各含 CNN / RF / XGBoost。指标见 `results/*.csv`。CNN 使用字符级数值映射作为基线特征。

### 04 Document Clustering
无监督文本聚类流水线；语料需自备至 `data/`。

### 05 Classic ML (Hyperspectral)
`notebooks/`；将 `PaviaU.mat` / `PaviaU_gt.mat` 放入 `data/`。

### 06 Auto Retail Database
`sql/01_schema.sql` · `02_seed_data.sql` · `03_queries.sql`。

### 07 PathMNIST CNN
病理 5 类组织分类可复现实验；可通过 `medmnist` 或本地 `.npz` 加载数据。详见 `report.txt`。

### 08 Multimodal AI for Healthcare
- Part A：SpaGCN 融合表达 / 空间 / 组织学（Visium DLPFC）
- Part B：病理视觉问答（多模态 LLM）
原始 Visium / 大模型权重不入库。

### 09 Causal Inference (Heffpox)
评估治疗对生存结局的效应：缺失机制、Kaplan–Meier、倾向评分与 IPTW。`python/` 与 `r/` 对照；`data/heffpox_2526.csv` 已附（较小）。

### 10 RA Pain Longitudinal Modeling
DAS28 与 VAS 疼痛轨迹的线性混合模型；`src/01_analysis.R`，图在 `outputs/`。

### 11 NHS COPD Dashboard
`src/01_data_processing.R` · `02_advanced_visualizations.R` · `03_shiny_dashboard.R`。大体积原始 Excel / shapefile / 明细 CSV 未入库，请按脚本说明自行准备 `data/`。

---

## Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

R 项目需本地安装：`tidyverse`, `lme4`, `lmerTest`, `shiny`, `sf` 等（见各脚本 `library()`）。

---

## Upload checklist (GitHub web)

1. 打开 [https://github.com/new](https://github.com/new) 创建空仓库（不要勾选自动 README）
2. 选择 **uploading an existing file**
3. 将本目录 **全部内容**（11 个项目文件夹 + `README.md` + `requirements.txt` + `.gitignore`）拖入
4. Commit

当前总大小约 **30MB 量级**，适合网页上传。请勿把本地备份中的原始大数据、`.pt` 权重、课程 PDF、学号材料一并拖入。

---

## Repository policy

- 已移除学号、课程号、本机绝对路径与作业模板口吻
- 大文件由 `.gitignore` 排除
- 仅保留根目录唯一 `README.md` 作为作品集说明
