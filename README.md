# Sales Win Rate Analysis - SkyGeni Challenge

**A comprehensive machine learning analysis to diagnose and address declining win rates in B2B SaaS sales**

---

## Executive Summary

This project analyzes 5,000 B2B SaaS deals (January 2023 - July 2024) to identify the root causes of a 1.3 percentage point decline in win rates during Q2-Q3 2024. Through advanced statistical analysis, custom metrics development, and machine learning modeling, we provide actionable recommendations to reverse this trend.

### Key Findings

- **Win Rate Decline**: 45.6% → 44.2% (-1.3pp)
- **Pipeline Volume Crisis**: -61.6% decline (not initially disclosed)
- **Revenue Impact**: $11.6M annualized revenue at risk
- **Primary Drivers**: Inbound lead quality degradation (-6.6pp), Proposal stage breakdown (-6.9pp), FinTech vertical challenges (-4.9pp)

### Recommended Actions

1. **Immediate**: Audit Inbound lead generation, intervene in Proposal stage deals, coach underperforming reps
2. **Short-term**: Reallocate budget to Referral programs, review FinTech strategy
3. **Medium-term**: Deploy ML-based deal risk scoring system
4. **Long-term**: Build decision intelligence platform

---

## Project Structure

```
SkyGeni/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions
│
├── src/                               # Source code (numbered execution order)
│   ├── 01_calculate_metrics.py        # Calculate 7 custom metrics
│   ├── 02_train_models.py             # Train 8 ML models with SHAP
│   ├── 03_analyze_drivers.py          # Win rate driver analysis
│   ├── 04_ensemble_model.py           # Ensemble modeling
│   ├── 05_create_charts.py            # Generate visualizations
│   └── 06_validate_metrics.py         # Validate metric calculations
│
├── docs/                              # Documentation (assignment parts)
│   ├── 01_problem_framing.md          # Part 1: Problem framing
│   ├── 02_insights_and_analysis.md    # Part 2: Insights & custom metrics
│   ├── 03_system_design.md            # Part 3: System architecture
│   └── 04_reflection.md               # Part 4: Reflection & limitations
│
├── data/                              # Data files
│   ├── sales_data.csv                 # Original dataset (5,000 deals)
│   ├── sales_data_with_metrics.csv    # With 7 custom metrics
│   └── model_comparison.csv           # Model performance results
│
├── visualizations/                    # Generated charts
│   ├── correlation_heatmap.png
│   ├── distribution_comparison.png
│   ├── time_series_analysis.png
│   ├── segmentation_heatmap.png
│   ├── funnel_analysis.png
│   ├── cohort_analysis.png
│   └── ... (11 total visualizations)
│
└── notebooks/                         # Jupyter notebooks
    └── 02_comprehensive_eda.ipynb     # Exploratory data analysis
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+ (tested on Python 3.14)
- Virtual environment (recommended)

### Installation

```bash
# Clone or navigate to project directory
cd /path/to/SkyGeni

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# For macOS users (required for XGBoost)
brew install libomp
```

---

## Usage

### Quick Start (5 minutes)

Run the complete analysis pipeline:

```bash
# 1. Calculate custom metrics
python src/01_calculate_metrics.py

# 2. Train ML models
python src/02_train_models.py

# 3. Create visualizations
python src/05_create_charts.py
```

### Individual Components

#### 1. Custom Metrics Calculation

```bash
python src/01_calculate_metrics.py
```

**Output**: `data/sales_data_with_metrics.csv` with 7 custom metrics:
- Pipeline Quality Score
- Deal Velocity Index
- Rep Efficiency Ratio
- Lead Source ROI
- Segment Attractiveness Index
- Pipeline Decay Rate
- Deal Complexity Score

#### 2. Machine Learning Models

```bash
python src/02_train_models.py
```

**Output**: 
- 8 trained models (LightGBM best: 54.56% accuracy)
- SHAP interpretability visualizations
- Model comparison results

#### 3. Win Rate Driver Analysis

```bash
python src/03_analyze_drivers.py
```

**Output**: Identifies key drivers of win rate decline

#### 4. Visualizations

```bash
python src/05_create_charts.py
```

**Output**: 11 publication-quality visualizations in `visualizations/`

---

## Documentation

### Assignment Parts

1. **Part 1 - Problem Framing**: [`docs/01_problem_framing.md`](docs/01_problem_framing.md)
   - Business problem analysis
   - Key questions framework
   - Metrics framework
   - Critical assumptions

2. **Part 2 - Insights & Analysis**: [`docs/02_insights_and_analysis.md`](docs/02_insights_and_analysis.md)
   - 3 key business insights
   - 7 custom metrics (exceeded requirement of 2)
   - Actionable recommendations

3. **Part 3 - System Design**: [`docs/03_system_design.md`](docs/03_system_design.md)
   - Architecture diagram
   - Data flow
   - Alert system
   - Execution schedule
   - Failure modes

4. **Part 4 - Reflection**: [`docs/04_reflection.md`](docs/04_reflection.md)
   - Weakest assumptions
   - Production failure modes
   - 1-month roadmap
   - Areas of least confidence

---

## Key Results

### Model Performance

| Model | Accuracy | AUC-ROC | F1 Score |
|-------|----------|---------|----------|
| **LightGBM** | **54.56%** | **0.5891** | **0.5512** |
| XGBoost | 54.24% | 0.5876 | 0.5489 |
| Gradient Boosting | 53.92% | 0.5845 | 0.5467 |
| Random Forest | 53.68% | 0.5823 | 0.5445 |
| Ensemble | 53.84% | 0.5834 | 0.5456 |

### Custom Metrics Impact

All 7 custom metrics show statistically significant correlation with win rate (p < 0.05):
- Pipeline Quality Score: r = 0.42
- Deal Velocity Index: r = 0.38
- Rep Efficiency Ratio: r = 0.45

---

## Methodology

### 1. Problem Framing
- Multi-layered problem decomposition
- Key questions framework
- Metrics framework design

### 2. Exploratory Data Analysis
- 5,000 deals analyzed
- 15+ visualizations created
- Statistical significance testing

### 3. Custom Metrics Development
- 7 custom metrics designed
- Validated for predictive power
- Business-aligned interpretations

### 4. Machine Learning
- 8 algorithms tested
- SHAP for interpretability
- Revenue-optimized thresholds

### 5. System Design
- Production-ready architecture
- Real-time alert system
- Scalability considerations

---

## Visualizations

### Sample Visualizations

- **Correlation Heatmap**: Feature relationships
- **Time Series Analysis**: Win rate trends over time
- **Segmentation Heatmap**: Performance by Industry × Region
- **Funnel Analysis**: Conversion rates by stage
- **Model Comparison**: Performance across 8 algorithms
- **SHAP Summary**: Feature importance with directionality

All visualizations are in the `visualizations/` directory.

---

## Limitations

### Data Constraints
- **Missing Features**: Competitor presence, customer engagement signals, budget confirmation
- **CRM-Only Data**: Limited to internal sales data
- **Historical Bias**: Past patterns may not predict future behavior

### Model Limitations
- **54% Accuracy**: Appropriate for CRM-only data, but higher accuracy requires additional data sources
- **Correlation vs. Causation**: Recommendations based on correlations, require A/B testing for validation
- **Class Imbalance**: 55/45 split handled with SMOTE

### Business Limitations
- **Adoption Risk**: Sales teams may be skeptical of AI predictions
- **Change Management**: Requires training and process changes
- **Maintenance**: Model requires retraining as business evolves

---

## Technologies Used

- **Python 3.14**: Core language
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML algorithms
- **XGBoost, LightGBM**: Gradient boosting
- **SHAP**: Model interpretability
- **matplotlib, seaborn**: Visualizations
- **Jupyter**: Interactive analysis

---

## Acknowledgments

- **SkyGeni Team**: For providing a realistic and challenging dataset

--- 

**Author**: Vivek Rajbansh    
**GitHub**: [github.com/vivekrajbansh](https://github.com/vivekrajbansh08)
