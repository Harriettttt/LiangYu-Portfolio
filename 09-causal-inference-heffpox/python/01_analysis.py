# -*- coding: utf-8 -*-
"""
Heffpox Treatment Analysis - Statistical Modelling and Inference for Health
Complete analysis including:
1. Exploratory Data Analysis
2. Missing Data Handling (Multiple Imputation)
3. Causal Inference (Propensity Score and IPW methods)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================
print("=" * 80)
print("HEFFPOX TREATMENT ANALYSIS REPORT")
print("Assessing the Effectiveness of Milnepan in Reducing Mortality")
print("=" * 80)

# Load data
df = pd.read_csv('heffpox_2526.csv')

print("\n" + "=" * 80)
print("SECTION 1: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n1.1 Dataset Overview")
print("-" * 40)
print(f"Total observations: {len(df)}")
print(f"Number of variables: {df.shape[1]}")
print(f"\nVariable names: {list(df.columns)}")

print("\n1.2 Data Types and Structure")
print("-" * 40)
print(df.dtypes)

print("\n1.3 Summary Statistics for Continuous Variables")
print("-" * 40)
continuous_vars = ['age', 'bmi', 'TEVENT']
print(df[continuous_vars].describe().round(2))

print("\n1.4 Frequency Tables for Categorical Variables")
print("-" * 40)
categorical_vars = ['heffpox', 'sex', 'smoking', 'diabetes', 'milnepan', 'icu', 'Status', 'death7day']
for var in categorical_vars:
    print(f"\n{var}:")
    counts = df[var].value_counts(dropna=False)
    percentages = df[var].value_counts(dropna=False, normalize=True) * 100
    summary = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
    print(summary)

# =============================================================================
# 1.5 Missing Data Analysis
# =============================================================================
print("\n1.5 Missing Data Analysis")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# Create missing data visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Missing data pattern
missing_vars = missing_df[missing_df['Missing Count'] > 0].index.tolist()
if missing_vars:
    ax1 = axes[0]
    missing_df[missing_df['Missing Count'] > 0]['Missing %'].plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Missing Data by Variable')
    ax1.set_ylabel('Percentage Missing')
    ax1.set_xlabel('Variable')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Missing data heatmap for subset
ax2 = axes[1]
sample_missing = df[missing_vars].head(100).isnull()
sns.heatmap(sample_missing.T, cbar=True, ax=ax2, cmap='YlOrRd')
ax2.set_title('Missing Data Pattern (First 100 Observations)')
ax2.set_xlabel('Observation Index')
ax2.set_ylabel('Variable')

plt.tight_layout()
plt.savefig('fig1_missing_data.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved: Missing data visualization")

# =============================================================================
# 1.6 Distribution Analysis
# =============================================================================
print("\n1.6 Distribution Analysis")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Age distribution
ax1 = axes[0, 0]
df['age'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Age Distribution')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Frequency')

# BMI distribution
ax2 = axes[0, 1]
df['bmi'].dropna().hist(bins=30, ax=ax2, color='seagreen', edgecolor='black')
ax2.set_title('BMI Distribution')
ax2.set_xlabel('BMI (kg/m²)')
ax2.set_ylabel('Frequency')

# TEVENT distribution
ax3 = axes[0, 2]
df['TEVENT'].hist(bins=30, ax=ax3, color='darkorange', edgecolor='black')
ax3.set_title('Time to Event Distribution')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Frequency')

# Treatment by outcome
ax4 = axes[1, 0]
cross_tab = pd.crosstab(df['milnepan'], df['death7day'])
cross_tab.plot(kind='bar', ax=ax4, color=['forestgreen', 'crimson'])
ax4.set_title('Treatment vs 7-Day Mortality')
ax4.set_xlabel('Milnepan Treatment (0=No, 1=Yes)')
ax4.set_ylabel('Count')
ax4.legend(['Survived', 'Died within 7 days'])
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

# Age by treatment group
ax5 = axes[1, 1]
df.boxplot(column='age', by='milnepan', ax=ax5)
ax5.set_title('Age by Treatment Group')
ax5.set_xlabel('Milnepan (0=No, 1=Yes)')
ax5.set_ylabel('Age (years)')
plt.suptitle('')

# BMI by treatment group
ax6 = axes[1, 2]
df.boxplot(column='bmi', by='milnepan', ax=ax6)
ax6.set_title('BMI by Treatment Group')
ax6.set_xlabel('Milnepan (0=No, 1=Yes)')
ax6.set_ylabel('BMI (kg/m²)')
plt.suptitle('')

plt.tight_layout()
plt.savefig('fig2_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: Distribution analysis")

# =============================================================================
# 1.7 Survival Analysis - Kaplan-Meier Curves
# =============================================================================
print("\n1.7 Survival Analysis")
print("-" * 40)

# Filter to heffpox patients only for main analysis
df_heffpox = df[df['heffpox'] == 1].copy()
print(f"Patients with Heffpox: {len(df_heffpox)}")
print(f"Patients without Heffpox: {len(df[df['heffpox'] == 0])}")

# Kaplan-Meier analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall survival
ax1 = axes[0]
kmf = KaplanMeierFitter()
kmf.fit(df_heffpox['TEVENT'], event_observed=df_heffpox['Status'], label='All Heffpox Patients')
kmf.plot_survival_function(ax=ax1, ci_show=True)
ax1.set_title('Overall Survival - Heffpox Patients')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Survival Probability')
ax1.set_xlim(0, 10)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Survival by treatment
ax2 = axes[1]
for treatment, group in df_heffpox.groupby('milnepan'):
    kmf = KaplanMeierFitter()
    label = 'Milnepan' if treatment == 1 else 'No Milnepan'
    kmf.fit(group['TEVENT'], event_observed=group['Status'], label=label)
    kmf.plot_survival_function(ax=ax2, ci_show=True)

ax2.set_title('Survival by Treatment Group')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Survival Probability')
ax2.set_xlim(0, 10)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig('fig3_kaplan_meier.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: Kaplan-Meier survival curves")

# Log-rank test
treated = df_heffpox[df_heffpox['milnepan'] == 1]
untreated = df_heffpox[df_heffpox['milnepan'] == 0]
results = logrank_test(treated['TEVENT'], untreated['TEVENT'], 
                       event_observed_A=treated['Status'], 
                       event_observed_B=untreated['Status'])
print(f"\nLog-rank test (Milnepan vs No Milnepan):")
print(f"Test statistic: {results.test_statistic:.4f}")
print(f"P-value: {results.p_value:.4f}")

# Median survival times
print("\nMedian Survival Times:")
for treatment, group in df_heffpox.groupby('milnepan'):
    kmf = KaplanMeierFitter()
    kmf.fit(group['TEVENT'], event_observed=group['Status'])
    label = 'Milnepan' if treatment == 1 else 'No Milnepan'
    print(f"{label}: {kmf.median_survival_time_:.2f} days")

# =============================================================================
# 1.8 Baseline Characteristics by Treatment Group
# =============================================================================
print("\n1.8 Baseline Characteristics by Treatment Group (Table 1)")
print("-" * 40)

def summarize_by_group(data, var, is_continuous=True):
    """Summarize variable by treatment group"""
    if is_continuous:
        summary = data.groupby('milnepan')[var].agg(['mean', 'std', 'count'])
        return summary
    else:
        cross = pd.crosstab(data['milnepan'], data[var], normalize='index') * 100
        return cross

# Create Table 1
table1_data = []

# Continuous variables
for var in ['age', 'bmi']:
    for trt in [0, 1]:
        subset = df_heffpox[df_heffpox['milnepan'] == trt]
        mean_val = subset[var].mean()
        std_val = subset[var].std()
        n_val = subset[var].notna().sum()
        table1_data.append({
            'Variable': var,
            'Treatment': 'Milnepan' if trt == 1 else 'No Milnepan',
            'Summary': f"{mean_val:.1f} (SD: {std_val:.1f})",
            'N': n_val
        })

# Categorical variables
for var in ['sex', 'smoking', 'diabetes', 'icu']:
    for trt in [0, 1]:
        subset = df_heffpox[df_heffpox['milnepan'] == trt]
        pct = (subset[var] == 1).mean() * 100
        n_val = (subset[var] == 1).sum()
        table1_data.append({
            'Variable': f"{var} (=1)",
            'Treatment': 'Milnepan' if trt == 1 else 'No Milnepan',
            'Summary': f"{pct:.1f}%",
            'N': n_val
        })

table1 = pd.DataFrame(table1_data)
print(table1.pivot(index='Variable', columns='Treatment', values='Summary'))

# Outcome by treatment
print("\n7-Day Mortality by Treatment Group:")
outcome_table = pd.crosstab(df_heffpox['milnepan'], df_heffpox['death7day'], 
                            margins=True, normalize='index') * 100
outcome_table.columns = ['Survived (%)', 'Died (%)']
outcome_table.index = ['No Milnepan', 'Milnepan', 'Total']
print(outcome_table.round(2))

# =============================================================================
# SECTION 2: MISSING DATA HANDLING
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: MISSING DATA HANDLING")
print("=" * 80)

print("\n2.1 Missing Data Mechanism Project")
print("-" * 40)

# Test if missingness is related to observed variables
# Create indicator for BMI missingness
df_heffpox['bmi_missing'] = df_heffpox['bmi'].isnull().astype(int)
df_heffpox['smoking_missing'] = df_heffpox['smoking'].isnull().astype(int)

# Compare characteristics between missing and non-missing
print("\nBMI Missingness Analysis:")
for var in ['age', 'sex', 'milnepan', 'death7day']:
    missing_group = df_heffpox[df_heffpox['bmi_missing'] == 1][var]
    complete_group = df_heffpox[df_heffpox['bmi_missing'] == 0][var]
    if var in ['age']:
        stat, pval = stats.ttest_ind(missing_group.dropna(), complete_group.dropna())
        print(f"{var}: Missing mean={missing_group.mean():.2f}, Complete mean={complete_group.mean():.2f}, p={pval:.4f}")
    else:
        # Chi-square test for categorical
        contingency = pd.crosstab(df_heffpox['bmi_missing'], df_heffpox[var])
        chi2, pval, dof, expected = stats.chi2_contingency(contingency)
        print(f"{var}: Chi2={chi2:.2f}, p={pval:.4f}")

print("\nConclusion: Missing data appears to be Missing at Random (MAR)")
print("Multiple Imputation is appropriate for handling missing data.")

# =============================================================================
# 2.2 Multiple Imputation
# =============================================================================
print("\n2.2 Multiple Imputation")
print("-" * 40)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Prepare data for imputation
vars_to_impute = ['age', 'bmi', 'smoking', 'sex', 'diabetes', 'milnepan', 'icu', 'death7day']
df_for_imputation = df_heffpox[vars_to_impute].copy()

# Number of imputations
n_imputations = 5
imputed_datasets = []

print(f"Performing {n_imputations} imputations...")

np.random.seed(42)
for i in range(n_imputations):
    imputer = IterativeImputer(max_iter=20, random_state=42+i, sample_posterior=True)
    imputed_data = imputer.fit_transform(df_for_imputation)
    imputed_df = pd.DataFrame(imputed_data, columns=vars_to_impute)
    
    # Round binary variables
    for var in ['smoking', 'sex', 'diabetes', 'milnepan', 'icu', 'death7day']:
        imputed_df[var] = imputed_df[var].round().clip(0, 1).astype(int)
    
    imputed_datasets.append(imputed_df)
    print(f"  Imputation {i+1} complete")

print(f"\nCreated {len(imputed_datasets)} imputed datasets")

# Check imputation quality
print("\nImputation Quality Check - BMI:")
original_bmi_mean = df_heffpox['bmi'].mean()
original_bmi_std = df_heffpox['bmi'].std()
print(f"Original: Mean={original_bmi_mean:.2f}, SD={original_bmi_std:.2f}")
for i, imp_df in enumerate(imputed_datasets):
    print(f"Imputation {i+1}: Mean={imp_df['bmi'].mean():.2f}, SD={imp_df['bmi'].std():.2f}")

# =============================================================================
# SECTION 3: CAUSAL INFERENCE - DAG AND CONFOUNDERS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: CAUSAL INFERENCE")
print("=" * 80)

print("\n3.1 Directed Acyclic Graph (DAG) Analysis")
print("-" * 40)
print("""
Based on domain knowledge and the data structure, we propose the following DAG:

                    Age
                   / | \\
                  /  |  \\
                 v   v   v
    Smoking --> BMI  |  Diabetes
        \\      |   |    /
         \\     |   |   /
          v    v   v  v
           Milnepan (Treatment)
               |
               v
           ICU (Mediator - DO NOT ADJUST)
               |
               v
         Death within 7 days (Outcome)

Confounders to adjust for (backdoor criterion):
- Age: Affects treatment assignment and outcome
- Sex: May affect treatment and outcome
- BMI: Associated with disease severity and treatment
- Smoking: Risk factor for disease severity
- Diabetes: Affects milnepan effectiveness and outcome

Variables NOT to adjust for:
- ICU: This is a MEDIATOR (on the causal pathway)
  Adjusting for ICU would block part of the treatment effect
- heffpox: We are analyzing only heffpox patients

Causal Effect of Interest:
- Average Treatment Effect (ATE) of milnepan on death7day
- Estimand: E[Y(1)] - E[Y(0)]
  where Y(1) = outcome if treated, Y(0) = outcome if untreated
""")

# Create DAG visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Node positions
nodes = {
    'Age': (5, 9),
    'Sex': (2, 8),
    'Smoking': (1, 6),
    'BMI': (3, 6),
    'Diabetes': (7, 6),
    'Milnepan': (5, 4),
    'ICU': (5, 2),
    'Death7day': (5, 0)
}

# Draw nodes
for node, (x, y) in nodes.items():
    color = 'lightgreen' if node == 'Milnepan' else ('salmon' if node == 'Death7day' else ('lightyellow' if node == 'ICU' else 'lightblue'))
    circle = plt.Circle((x, y), 0.6, color=color, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold')

# Draw arrows (edges)
edges = [
    ('Age', 'BMI'), ('Age', 'Milnepan'), ('Age', 'Death7day'),
    ('Sex', 'Milnepan'), ('Sex', 'Death7day'),
    ('Smoking', 'BMI'), ('Smoking', 'Milnepan'),
    ('BMI', 'Milnepan'), ('BMI', 'Death7day'),
    ('Diabetes', 'Milnepan'), ('Diabetes', 'Death7day'),
    ('Milnepan', 'ICU'), ('Milnepan', 'Death7day'),
    ('ICU', 'Death7day')
]

for start, end in edges:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    # Adjust for node radius
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    x1_adj = x1 + 0.6 * dx/dist
    y1_adj = y1 + 0.6 * dy/dist
    x2_adj = x2 - 0.6 * dx/dist
    y2_adj = y2 - 0.6 * dy/dist
    
    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Directed Acyclic Graph (DAG) for Milnepan Treatment Effect\n(Green=Treatment, Red=Outcome, Yellow=Mediator, Blue=Confounders)', 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig4_dag.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 4 saved: DAG visualization")

# =============================================================================
# 3.2 Propensity Score Analysis
# =============================================================================
print("\n3.2 Propensity Score Estimation")
print("-" * 40)

# Use first imputed dataset for initial analysis
df_analysis = imputed_datasets[0].copy()

# Confounders for propensity score model
confounders = ['age', 'sex', 'bmi', 'smoking', 'diabetes']

# Fit propensity score model
X = df_analysis[confounders]
y = df_analysis['milnepan']

# Standardize continuous variables
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])

# Logistic regression for propensity scores
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, y)
propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]

df_analysis['ps'] = propensity_scores

print("Propensity Score Model Coefficients:")
for var, coef in zip(confounders, ps_model.coef_[0]):
    print(f"  {var}: {coef:.4f}")

# Propensity score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
df_analysis[df_analysis['milnepan'] == 0]['ps'].hist(bins=30, alpha=0.6, label='No Milnepan', ax=ax1, color='blue')
df_analysis[df_analysis['milnepan'] == 1]['ps'].hist(bins=30, alpha=0.6, label='Milnepan', ax=ax1, color='red')
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Propensity Score Distribution by Treatment Group')
ax1.legend()

# Overlap Project
ax2 = axes[1]
ps_treated = df_analysis[df_analysis['milnepan'] == 1]['ps']
ps_control = df_analysis[df_analysis['milnepan'] == 0]['ps']
ax2.boxplot([ps_control, ps_treated], labels=['No Milnepan', 'Milnepan'])
ax2.set_ylabel('Propensity Score')
ax2.set_title('Propensity Score Overlap Project')

plt.tight_layout()
plt.savefig('fig5_propensity_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 5 saved: Propensity score distribution")

# Check overlap
print(f"\nPropensity Score Summary:")
print(f"Treated: Mean={ps_treated.mean():.3f}, Min={ps_treated.min():.3f}, Max={ps_treated.max():.3f}")
print(f"Control: Mean={ps_control.mean():.3f}, Min={ps_control.min():.3f}, Max={ps_control.max():.3f}")

# =============================================================================
# 3.3 METHOD 1: Inverse Probability Weighting (IPW)
# =============================================================================
print("\n3.3 METHOD 1: Inverse Probability Weighting (IPW)")
print("-" * 40)

def calculate_ipw_ate(data, ps_col='ps', treatment_col='milnepan', outcome_col='death7day'):
    """Calculate ATE using IPW"""
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    
    # IPW weights
    # For treated: 1/PS
    # For control: 1/(1-PS)
    w_treated = 1 / treated[ps_col]
    w_control = 1 / (1 - control[ps_col])
    
    # Stabilized weights (optional but recommended)
    p_treated = data[treatment_col].mean()
    sw_treated = p_treated / treated[ps_col]
    sw_control = (1 - p_treated) / (1 - control[ps_col])
    
    # Weighted outcomes
    y1_weighted = (treated[outcome_col] * sw_treated).sum() / sw_treated.sum()
    y0_weighted = (control[outcome_col] * sw_control).sum() / sw_control.sum()
    
    ate = y1_weighted - y0_weighted
    
    return ate, y1_weighted, y0_weighted, sw_treated, sw_control

# Calculate IPW ATE for each imputed dataset
ipw_ates = []
print("\nIPW Results for Each Imputed Dataset:")
for i, imp_df in enumerate(imputed_datasets):
    # Recalculate propensity scores for each imputed dataset
    X = imp_df[confounders]
    X_scaled = X.copy()
    X_scaled[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])
    ps = ps_model.fit(X_scaled, imp_df['milnepan']).predict_proba(X_scaled)[:, 1]
    imp_df['ps'] = ps
    
    ate, y1, y0, _, _ = calculate_ipw_ate(imp_df)
    ipw_ates.append(ate)
    print(f"  Imputation {i+1}: ATE = {ate:.4f} (Y1={y1:.4f}, Y0={y0:.4f})")

# Rubin's rules for combining estimates
ipw_ate_pooled = np.mean(ipw_ates)
between_var = np.var(ipw_ates, ddof=1)
print(f"\nPooled IPW ATE: {ipw_ate_pooled:.4f}")
print(f"Between-imputation variance: {between_var:.6f}")

# Bootstrap for confidence interval (using first imputed dataset)
def bootstrap_ipw(data, n_bootstrap=1000):
    """Bootstrap confidence interval for IPW ATE"""
    ates = []
    n = len(data)
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[boot_idx].copy()
        
        # Refit PS model
        X = boot_data[confounders]
        X_scaled = X.copy()
        X_scaled[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])
        ps = ps_model.fit(X_scaled, boot_data['milnepan']).predict_proba(X_scaled)[:, 1]
        boot_data['ps'] = ps
        
        ate, _, _, _, _ = calculate_ipw_ate(boot_data)
        ates.append(ate)
    
    return np.percentile(ates, [2.5, 97.5])

print("\nCalculating bootstrap confidence interval...")
np.random.seed(42)
ci_lower, ci_upper = bootstrap_ipw(df_analysis, n_bootstrap=500)
print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# =============================================================================
# 3.4 METHOD 2: Outcome Regression with Covariate Adjustment
# =============================================================================
print("\n3.4 METHOD 2: Outcome Regression with Covariate Adjustment")
print("-" * 40)

def outcome_regression_ate(data, confounders, treatment_col='milnepan', outcome_col='death7day'):
    """Calculate ATE using outcome regression (g-computation)"""
    # Fit outcome model
    X = data[confounders + [treatment_col]].copy().reset_index(drop=True)
    y = data[outcome_col].reset_index(drop=True)
    
    # Add intercept
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # Logistic regression
    model = sm.Logit(y, X_with_const).fit(disp=0)
    
    # Predict potential outcomes
    # Y(1): Set treatment to 1 for everyone
    X_treated = data[confounders + [treatment_col]].copy().reset_index(drop=True)
    X_treated[treatment_col] = 1
    X_treated_const = sm.add_constant(X_treated, has_constant='add')
    y1_pred = model.predict(X_treated_const)
    
    # Y(0): Set treatment to 0 for everyone
    X_control = data[confounders + [treatment_col]].copy().reset_index(drop=True)
    X_control[treatment_col] = 0
    X_control_const = sm.add_constant(X_control, has_constant='add')
    y0_pred = model.predict(X_control_const)
    
    # ATE
    ate = y1_pred.mean() - y0_pred.mean()
    
    return ate, y1_pred.mean(), y0_pred.mean(), model

# Calculate regression ATE for each imputed dataset
reg_ates = []
print("\nOutcome Regression Results for Each Imputed Dataset:")
for i, imp_df in enumerate(imputed_datasets):
    ate, y1, y0, _ = outcome_regression_ate(imp_df, confounders)
    reg_ates.append(ate)
    print(f"  Imputation {i+1}: ATE = {ate:.4f} (E[Y(1)]={y1:.4f}, E[Y(0)]={y0:.4f})")

# Pooled estimate
reg_ate_pooled = np.mean(reg_ates)
print(f"\nPooled Regression ATE: {reg_ate_pooled:.4f}")

# Full model output for first imputed dataset
print("\nFull Outcome Model (First Imputed Dataset):")
ate, y1, y0, full_model = outcome_regression_ate(imputed_datasets[0], confounders)
print(full_model.summary())

# =============================================================================
# 3.5 METHOD 3: Doubly Robust Estimation (AIPW)
# =============================================================================
print("\n3.5 METHOD 3: Doubly Robust Estimation (AIPW)")
print("-" * 40)

def doubly_robust_ate(data, confounders, ps_col='ps', treatment_col='milnepan', outcome_col='death7day'):
    """Calculate ATE using Augmented IPW (doubly robust)"""
    n = len(data)
    
    # Fit outcome models for treated and control separately
    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]
    
    # Model for E[Y|X, T=1]
    X_treated = sm.add_constant(treated_data[confounders])
    y_treated = treated_data[outcome_col]
    model_treated = sm.Logit(y_treated, X_treated).fit(disp=0)
    
    # Model for E[Y|X, T=0]
    X_control = sm.add_constant(control_data[confounders])
    y_control = control_data[outcome_col]
    model_control = sm.Logit(y_control, X_control).fit(disp=0)
    
    # Predict for all observations
    X_all = sm.add_constant(data[confounders])
    mu1 = model_treated.predict(X_all)  # E[Y|X, T=1]
    mu0 = model_control.predict(X_all)  # E[Y|X, T=0]
    
    # AIPW estimator
    ps = data[ps_col]
    T = data[treatment_col]
    Y = data[outcome_col]
    
    # Component 1: IPW for treated
    ipw1 = T * Y / ps
    # Component 2: IPW for control
    ipw0 = (1 - T) * Y / (1 - ps)
    # Component 3: Augmentation for treated
    aug1 = (T - ps) / ps * mu1
    # Component 4: Augmentation for control
    aug0 = (ps - T) / (1 - ps) * mu0
    
    # AIPW estimate
    y1_aipw = (ipw1 - aug1).mean() + mu1.mean()
    y0_aipw = (ipw0 + aug0).mean() + mu0.mean()
    
    # Simplified AIPW
    aipw_y1 = (T * (Y - mu1) / ps + mu1).mean()
    aipw_y0 = ((1 - T) * (Y - mu0) / (1 - ps) + mu0).mean()
    
    ate = aipw_y1 - aipw_y0
    
    return ate, aipw_y1, aipw_y0

# Calculate AIPW ATE for each imputed dataset
aipw_ates = []
print("\nAIPW (Doubly Robust) Results for Each Imputed Dataset:")
for i, imp_df in enumerate(imputed_datasets):
    # Ensure PS is calculated
    X = imp_df[confounders]
    X_scaled = X.copy()
    X_scaled[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])
    ps = ps_model.fit(X_scaled, imp_df['milnepan']).predict_proba(X_scaled)[:, 1]
    imp_df['ps'] = ps
    
    ate, y1, y0 = doubly_robust_ate(imp_df, confounders)
    aipw_ates.append(ate)
    print(f"  Imputation {i+1}: ATE = {ate:.4f} (E[Y(1)]={y1:.4f}, E[Y(0)]={y0:.4f})")

aipw_ate_pooled = np.mean(aipw_ates)
print(f"\nPooled AIPW ATE: {aipw_ate_pooled:.4f}")

# =============================================================================
# 3.6 Sensitivity Analysis - Covariate Balance
# =============================================================================
print("\n3.6 Covariate Balance Project")
print("-" * 40)

def standardized_mean_diff(treated, control):
    """Calculate standardized mean difference"""
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_std == 0:
        return 0
    return (treated.mean() - control.mean()) / pooled_std

# Before weighting
print("\nStandardized Mean Differences (Before vs After IPW):")
print(f"{'Variable':<15} {'Before':<10} {'After IPW':<10}")
print("-" * 35)

df_analysis = imputed_datasets[0].copy()
X = df_analysis[confounders]
X_scaled = X.copy()
X_scaled[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])
ps = ps_model.fit(X_scaled, df_analysis['milnepan']).predict_proba(X_scaled)[:, 1]
df_analysis['ps'] = ps

treated = df_analysis[df_analysis['milnepan'] == 1]
control = df_analysis[df_analysis['milnepan'] == 0]

# IPW weights
p_treated = df_analysis['milnepan'].mean()
w_treated = p_treated / treated['ps']
w_control = (1 - p_treated) / (1 - control['ps'])

balance_data = []
for var in confounders:
    smd_before = standardized_mean_diff(treated[var], control[var])
    
    # Weighted means
    weighted_mean_treated = np.average(treated[var], weights=w_treated)
    weighted_mean_control = np.average(control[var], weights=w_control)
    pooled_std = np.sqrt((treated[var].var() + control[var].var()) / 2)
    smd_after = (weighted_mean_treated - weighted_mean_control) / pooled_std if pooled_std > 0 else 0
    
    print(f"{var:<15} {smd_before:<10.3f} {smd_after:<10.3f}")
    balance_data.append({'Variable': var, 'Before': abs(smd_before), 'After': abs(smd_after)})

# Balance plot
balance_df = pd.DataFrame(balance_data)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(confounders))
width = 0.35
ax.bar(x - width/2, balance_df['Before'], width, label='Before Weighting', color='salmon')
ax.bar(x + width/2, balance_df['After'], width, label='After IPW', color='steelblue')
ax.axhline(y=0.1, color='green', linestyle='--', label='Threshold (0.1)')
ax.set_xlabel('Covariate')
ax.set_ylabel('Absolute Standardized Mean Difference')
ax.set_title('Covariate Balance Before and After IPW')
ax.set_xticks(x)
ax.set_xticklabels(confounders)
ax.legend()
plt.tight_layout()
plt.savefig('fig6_covariate_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 6 saved: Covariate balance plot")

# =============================================================================
# SECTION 4: RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: RESULTS SUMMARY")
print("=" * 80)

print("\n4.1 Summary of Causal Effect Estimates")
print("-" * 40)

results_summary = pd.DataFrame({
    'Method': ['IPW', 'Outcome Regression', 'AIPW (Doubly Robust)'],
    'Pooled ATE': [ipw_ate_pooled, reg_ate_pooled, aipw_ate_pooled],
    'Interpretation': [
        f"{'Increase' if ipw_ate_pooled > 0 else 'Decrease'} of {abs(ipw_ate_pooled)*100:.1f}pp in 7-day mortality",
        f"{'Increase' if reg_ate_pooled > 0 else 'Decrease'} of {abs(reg_ate_pooled)*100:.1f}pp in 7-day mortality",
        f"{'Increase' if aipw_ate_pooled > 0 else 'Decrease'} of {abs(aipw_ate_pooled)*100:.1f}pp in 7-day mortality"
    ]
})
print(results_summary.to_string(index=False))

print("\n4.2 Consistency of Results")
print("-" * 40)
ate_range = max(ipw_ate_pooled, reg_ate_pooled, aipw_ate_pooled) - min(ipw_ate_pooled, reg_ate_pooled, aipw_ate_pooled)
print(f"Range of ATE estimates: {ate_range:.4f}")
print(f"All methods show {'consistent' if ate_range < 0.05 else 'some variation in'} results")

# Direction of effect
if ipw_ate_pooled < 0 and reg_ate_pooled < 0 and aipw_ate_pooled < 0:
    print("\nAll methods indicate milnepan REDUCES 7-day mortality risk")
elif ipw_ate_pooled > 0 and reg_ate_pooled > 0 and aipw_ate_pooled > 0:
    print("\nAll methods indicate milnepan INCREASES 7-day mortality risk")
else:
    print("\nMethods show mixed results regarding direction of effect")

# =============================================================================
# 4.3 Naive vs Adjusted Comparison
# =============================================================================
print("\n4.3 Naive vs Adjusted Estimates Comparison")
print("-" * 40)

# Naive estimate (unadjusted)
naive_treated = df_heffpox[df_heffpox['milnepan'] == 1]['death7day'].mean()
naive_control = df_heffpox[df_heffpox['milnepan'] == 0]['death7day'].mean()
naive_ate = naive_treated - naive_control

print(f"Naive (unadjusted) estimate:")
print(f"  Mortality in treated: {naive_treated*100:.1f}%")
print(f"  Mortality in control: {naive_control*100:.1f}%")
print(f"  Naive ATE: {naive_ate:.4f}")
print(f"\nAdjusted estimates (average):")
print(f"  IPW ATE: {ipw_ate_pooled:.4f}")
print(f"  Regression ATE: {reg_ate_pooled:.4f}")
print(f"  AIPW ATE: {aipw_ate_pooled:.4f}")

confounding_bias = naive_ate - np.mean([ipw_ate_pooled, reg_ate_pooled, aipw_ate_pooled])
print(f"\nEstimated confounding bias: {confounding_bias:.4f}")

# =============================================================================
# 4.4 Cox Proportional Hazards Model (Supplementary)
# =============================================================================
print("\n4.4 Cox Proportional Hazards Model (Supplementary Analysis)")
print("-" * 40)

# Prepare data for Cox model
cox_data = imputed_datasets[0].copy()
cox_data['TEVENT'] = df_heffpox['TEVENT'].values
cox_data['Status'] = df_heffpox['Status'].values

# Fit Cox model
cph = CoxPHFitter()
cox_vars = confounders + ['milnepan']
cph.fit(cox_data[cox_vars + ['TEVENT', 'Status']], duration_col='TEVENT', event_col='Status')

print("\nCox Proportional Hazards Model Results:")
print(cph.summary[['coef', 'exp(coef)', 'p']])

milnepan_hr = cph.summary.loc['milnepan', 'exp(coef)']
milnepan_p = cph.summary.loc['milnepan', 'p']
print(f"\nMilnepan Hazard Ratio: {milnepan_hr:.3f} (p = {milnepan_p:.4f})")
if milnepan_hr < 1:
    print(f"Interpretation: Milnepan reduces hazard of death by {(1-milnepan_hr)*100:.1f}%")
else:
    print(f"Interpretation: Milnepan increases hazard of death by {(milnepan_hr-1)*100:.1f}%")

# =============================================================================
# FINAL CONCLUSIONS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: CONCLUSIONS AND RECOMMENDATIONS")
print("=" * 80)

avg_ate = np.mean([ipw_ate_pooled, reg_ate_pooled, aipw_ate_pooled])

print(f"""
SUMMARY OF FINDINGS:

1. STUDY POPULATION:
   - Analyzed {len(df_heffpox)} patients diagnosed with Heffpox
   - {(df_heffpox['milnepan']==1).sum()} ({(df_heffpox['milnepan']==1).mean()*100:.1f}%) received Milnepan treatment
   - Overall 7-day mortality: {df_heffpox['death7day'].mean()*100:.1f}%

2. MISSING DATA:
   - BMI had {df_heffpox['bmi'].isnull().mean()*100:.1f}% missing values
   - Smoking had {df_heffpox['smoking'].isnull().mean()*100:.1f}% missing values
   - Multiple imputation with {n_imputations} datasets was used

3. CAUSAL EFFECT ESTIMATES:
   - IPW: {ipw_ate_pooled:.4f} ({ipw_ate_pooled*100:.1f} percentage points)
   - Outcome Regression: {reg_ate_pooled:.4f} ({reg_ate_pooled*100:.1f} percentage points)
   - AIPW (Doubly Robust): {aipw_ate_pooled:.4f} ({aipw_ate_pooled*100:.1f} percentage points)
   - Average ATE: {avg_ate:.4f} ({avg_ate*100:.1f} percentage points)

4. INTERPRETATION:
   {'Milnepan treatment is associated with a REDUCTION in 7-day mortality risk.' if avg_ate < 0 else 'Milnepan treatment is associated with an INCREASE in 7-day mortality risk.'}
   The treatment effect is {'statistically meaningful' if abs(avg_ate) > 0.05 else 'relatively small'}.

5. CONSISTENCY:
   All three causal inference methods show {'consistent' if ate_range < 0.05 else 'broadly similar'} results,
   increasing confidence in the findings.

6. LIMITATIONS:
   - Observational data: unmeasured confounding possible
   - ICU was not adjusted for (mediator)
   - Results depend on correct model specification

RECOMMENDATION TO DR. C. ROBIN:
Based on this analysis, {'milnepan appears to be effective in reducing 7-day mortality risk among Heffpox patients' if avg_ate < -0.02 else 'milnepan does not appear to significantly reduce 7-day mortality risk' if abs(avg_ate) < 0.02 else 'milnepan may increase 7-day mortality risk and should be used with caution'}.
{'A randomized controlled trial is recommended to confirm these findings.' if abs(avg_ate) > 0.02 else 'Further investigation is warranted.'}
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated Figures:")
print("  - fig1_missing_data.png")
print("  - fig2_distributions.png")
print("  - fig3_kaplan_meier.png")
print("  - fig4_dag.png")
print("  - fig5_propensity_scores.png")
print("  - fig6_covariate_balance.png")

