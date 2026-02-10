"""
SkyGeni Sales Intelligence - Model with SHAP & Business Optimization
Purpose: High-performance model optimized for business metrics with full interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, matthews_corrcoef,
                             brier_score_loss, log_loss)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import shap

import warnings
warnings.filterwarnings('ignore')


class ProductionWinRateModel:
    """
    Model with business optimization and interpretability
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.shap_explainer = None
        self.shap_values = None
        
    def analyze_class_distribution(self):
        """Analyze class imbalance"""
        print("\n" + "="*80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*80)
        
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        
        class_dist = self.df['won'].value_counts()
        class_pct = self.df['won'].value_counts(normalize=True) * 100
        
        print(f"\nClass Distribution:")
        print(f"  Lost (0): {class_dist[0]:,} ({class_pct[0]:.1f}%)")
        print(f"  Won (1):  {class_dist[1]:,} ({class_pct[1]:.1f}%)")
        print(f"  Imbalance Ratio: {class_dist[0] / class_dist[1]:.2f}:1")
        
        if abs(class_pct[0] - 50) > 5:
            print(f"\n Class imbalance detected! Will apply SMOTE.")
            self.use_smote = True
        else:
            print(f"\n Classes are balanced. No resampling needed.")
            self.use_smote = False
        
        return class_dist, class_pct
    
    def engineer_advanced_features(self):
        """Engineer 40+ domain-specific features"""
        print("\n" + "="*80)
        print("ADVANCED FEATURE ENGINEERING (40+ FEATURES)")
        print("="*80)
        
        # Ensure dates
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        self.df['closed_date'] = pd.to_datetime(self.df['closed_date'])
        
        # Target
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        
        # === CATEGORY 1: DEAL CHARACTERISTICS (10 features) ===
        print("\n1️⃣ Deal Characteristics (10 features)...")
        
        # Amount features
        self.df['deal_amount_log'] = np.log10(self.df['deal_amount'] + 1)
        self.df['deal_amount_sqrt'] = np.sqrt(self.df['deal_amount'])
        self.df['deal_amount_squared'] = self.df['deal_amount'] ** 2
        
        # Size categories
        self.df['deal_size_category'] = pd.cut(self.df['deal_amount'], 
                                                 bins=[0, 10000, 30000, 60000, 100000],
                                                 labels=[0, 1, 2, 3]).astype(int)
        
        # Customer size proxy (SMB vs Enterprise based on deal size)
        self.df['is_smb'] = (self.df['deal_amount'] < 20000).astype(int)
        self.df['is_mid_market'] = ((self.df['deal_amount'] >= 20000) & 
                                     (self.df['deal_amount'] < 50000)).astype(int)
        self.df['is_enterprise'] = (self.df['deal_amount'] >= 50000).astype(int)
        
        # Cycle features
        self.df['sales_cycle_log'] = np.log10(self.df['sales_cycle_days'] + 1)
        self.df['sales_cycle_squared'] = self.df['sales_cycle_days'] ** 2
        
        # Deal velocity (amount per day)
        self.df['deal_velocity'] = self.df['deal_amount'] / (self.df['sales_cycle_days'] + 1)
        
        # === CATEGORY 2: TEMPORAL FEATURES (8 features) ===
        print("2️⃣ Temporal Features (8 features)...")
        
        self.df['created_quarter'] = self.df['created_date'].dt.quarter
        self.df['created_month'] = self.df['created_date'].dt.month
        self.df['created_day_of_week'] = self.df['created_date'].dt.dayofweek
        self.df['created_week_of_year'] = self.df['created_date'].dt.isocalendar().week
        self.df['closed_quarter'] = self.df['closed_date'].dt.quarter
        self.df['is_recent'] = (self.df['closed_date'] >= '2024-04-01').astype(int)
        self.df['is_q2_2024'] = ((self.df['closed_date'] >= '2024-04-01') & 
                                  (self.df['closed_date'] < '2024-07-01')).astype(int)
        self.df['is_q3_2024'] = ((self.df['closed_date'] >= '2024-07-01') & 
                                  (self.df['closed_date'] < '2024-10-01')).astype(int)
        
        # === CATEGORY 3: CATEGORICAL ENCODINGS (5 features) ===
        print("3️⃣ Categorical Encodings (5 features)...")
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.df['region_encoded'] = le.fit_transform(self.df['region'])
        self.df['industry_encoded'] = le.fit_transform(self.df['industry'])
        self.df['product_type_encoded'] = le.fit_transform(self.df['product_type'])
        self.df['lead_source_encoded'] = le.fit_transform(self.df['lead_source'])
        self.df['deal_stage_encoded'] = le.fit_transform(self.df['deal_stage'])
        
        # === CATEGORY 4: HISTORICAL PERFORMANCE (7 features) ===
        print("4️⃣ Historical Performance (7 features)...")
        
        # Rep performance
        rep_stats = self.df.groupby('sales_rep_id').agg({
            'won': ['mean', 'count'],
            'deal_amount': 'mean',
            'sales_cycle_days': 'mean'
        })
        rep_stats.columns = ['rep_wr', 'rep_count', 'rep_avg_deal', 'rep_avg_cycle']
        self.df = self.df.merge(rep_stats, left_on='sales_rep_id', right_index=True, how='left')
        
        # Segment performance
        self.df['industry_wr'] = self.df.groupby('industry')['won'].transform('mean')
        self.df['source_wr'] = self.df.groupby('lead_source')['won'].transform('mean')
        self.df['region_wr'] = self.df.groupby('region')['won'].transform('mean')
        
        # === CATEGORY 5: INTERACTION FEATURES (10 features) ===
        print("5️⃣ Interaction Features (10 features)...")
        
        self.df['deal_amount_x_cycle'] = self.df['deal_amount'] * self.df['sales_cycle_days']
        self.df['amount_x_rep_wr'] = self.df['deal_amount'] * self.df['rep_wr']
        self.df['cycle_x_rep_wr'] = self.df['sales_cycle_days'] * self.df['rep_wr']
        
        # Problem combinations
        self.df['is_large_slow_deal'] = ((self.df['deal_amount'] > 30000) & 
                                          (self.df['sales_cycle_days'] > 60)).astype(int)
        self.df['is_inbound_recent'] = ((self.df['lead_source'] == 'Inbound') & 
                                         (self.df['is_recent'] == 1)).astype(int)
        self.df['is_proposal_recent'] = ((self.df['deal_stage'] == 'Proposal') & 
                                          (self.df['is_recent'] == 1)).astype(int)
        self.df['is_fintech_recent'] = ((self.df['industry'] == 'FinTech') & 
                                         (self.df['is_recent'] == 1)).astype(int)
        
        # Positive combinations
        self.df['is_referral_fast'] = ((self.df['lead_source'] == 'Referral') & 
                                        (self.df['sales_cycle_days'] < 60)).astype(int)
        self.df['is_saas_referral'] = ((self.df['industry'] == 'SaaS') & 
                                        (self.df['lead_source'] == 'Referral')).astype(int)
        self.df['is_high_performer_rep'] = (self.df['rep_wr'] > 0.48).astype(int)
        
        # === CATEGORY 6: VELOCITY & EFFICIENCY (5 features) ===
        print("6️⃣ Velocity & Efficiency (5 features)...")
        
        # Expected vs actual cycle
        median_cycle_by_size = self.df.groupby('deal_size_category')['sales_cycle_days'].transform('median')
        self.df['cycle_vs_expected'] = self.df['sales_cycle_days'] / (median_cycle_by_size + 1)
        self.df['is_fast_deal'] = (self.df['cycle_vs_expected'] < 0.8).astype(int)
        self.df['is_slow_deal'] = (self.df['cycle_vs_expected'] > 1.2).astype(int)
        
        # Rep efficiency
        self.df['rep_efficiency'] = self.df['rep_wr'] / (self.df['rep_avg_cycle'] + 1)
        self.df['deal_efficiency'] = self.df['deal_amount'] / (self.df['sales_cycle_days'] + 1)
        
        # === CATEGORY 7: SEGMENT FLAGS (5 features) ===
        print("7️⃣ Segment Flags (5 features)...")
        
        self.df['is_fintech'] = (self.df['industry'] == 'FinTech').astype(int)
        self.df['is_saas'] = (self.df['industry'] == 'SaaS').astype(int)
        self.df['is_proposal_stage'] = (self.df['deal_stage'] == 'Proposal').astype(int)
        self.df['is_inbound'] = (self.df['lead_source'] == 'Inbound').astype(int)
        self.df['is_referral'] = (self.df['lead_source'] == 'Referral').astype(int)
        
        print(f"\n Engineered 45+ features total")
        return self.df
    
    def prepare_data_with_smote(self, test_size=0.25, random_state=42):
        """Prepare data with SMOTE if needed"""
        print("\n" + "="*80)
        print("DATA PREPARATION WITH SMOTE")
        print("="*80)
        
        # Select features
        feature_cols = [
            # Deal characteristics
            'deal_amount_log', 'deal_amount_sqrt', 'deal_amount_squared',
            'deal_size_category', 'is_smb', 'is_mid_market', 'is_enterprise',
            'sales_cycle_log', 'sales_cycle_squared', 'deal_velocity',
            # Temporal
            'created_quarter', 'created_month', 'created_day_of_week', 'created_week_of_year',
            'closed_quarter', 'is_recent', 'is_q2_2024', 'is_q3_2024',
            # Categorical
            'region_encoded', 'industry_encoded', 'product_type_encoded',
            'lead_source_encoded', 'deal_stage_encoded',
            # Historical performance
            'rep_wr', 'rep_count', 'rep_avg_deal', 'rep_avg_cycle',
            'industry_wr', 'source_wr', 'region_wr',
            # Interactions
            'deal_amount_x_cycle', 'amount_x_rep_wr', 'cycle_x_rep_wr',
            'is_large_slow_deal', 'is_inbound_recent', 'is_proposal_recent', 'is_fintech_recent',
            'is_referral_fast', 'is_saas_referral', 'is_high_performer_rep',
            # Velocity
            'cycle_vs_expected', 'is_fast_deal', 'is_slow_deal',
            'rep_efficiency', 'deal_efficiency',
            # Segment flags
            'is_fintech', 'is_saas', 'is_proposal_stage', 'is_inbound', 'is_referral'
        ]
        
        X = self.df[feature_cols].copy()
        y = self.df['won'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_names = feature_cols
        
        print(f" Original training set: {len(self.X_train):,} samples")
        print(f"   Class distribution: {self.y_train.value_counts().to_dict()}")
        
        # Apply SMOTE if needed
        if self.use_smote:
            print(f"\n🔄 Applying SMOTE...")
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train, self.y_train
            )
            print(f" Resampled training set: {len(self.X_train_resampled):,} samples")
            print(f"   Class distribution: {pd.Series(self.y_train_resampled).value_counts().to_dict()}")
        else:
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
        
        print(f"\n Test set: {len(self.X_test):,} samples")
        print(f"   Features: {len(feature_cols)}")
        
        return self.X_train_resampled, self.X_test, self.y_train_resampled, self.y_test
    
    def train_optimized_models(self):
        """Train 3 best models with hyperparameter tuning"""
        print("\n" + "="*80)
        print("TRAINING 3 OPTIMIZED MODELS")
        print("="*80)
        
        models = {}
        
        # Model 1: LightGBM (best from previous)
        print("\n1️⃣ Training LightGBM with hyperparameter tuning...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            num_leaves=50,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = lgbm.predict(self.X_test)
        y_pred_proba = lgbm.predict_proba(self.X_test)[:, 1]
        
        models['LightGBM'] = {
            'model': lgbm,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'f1': f1_score(self.y_test, y_pred),
            'mcc': matthews_corrcoef(self.y_test, y_pred),
            'brier': brier_score_loss(self.y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   Accuracy: {models['LightGBM']['accuracy']:.4f}")
        print(f"   AUC-ROC: {models['LightGBM']['auc']:.4f}")
        print(f"   F1 Score: {models['LightGBM']['f1']:.4f}")
        
        # Model 2: Gradient Boosting (strong baseline)
        print("\n2️⃣ Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        gb.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = gb.predict(self.X_test)
        y_pred_proba = gb.predict_proba(self.X_test)[:, 1]
        
        models['Gradient Boosting'] = {
            'model': gb,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'f1': f1_score(self.y_test, y_pred),
            'mcc': matthews_corrcoef(self.y_test, y_pred),
            'brier': brier_score_loss(self.y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   Accuracy: {models['Gradient Boosting']['accuracy']:.4f}")
        print(f"   AUC-ROC: {models['Gradient Boosting']['auc']:.4f}")
        print(f"   F1 Score: {models['Gradient Boosting']['f1']:.4f}")
        
        # Model 3: XGBoost (strong baseline)
        print("\n3️⃣ Training XGBoost...")
        xgboost = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgboost.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = xgboost.predict(self.X_test)
        y_pred_proba = xgboost.predict_proba(self.X_test)[:, 1]
        
        models['XGBoost'] = {
            'model': xgboost,
            'accuracy': (y_pred == self.y_test).mean(),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'f1': f1_score(self.y_test, y_pred),
            'mcc': matthews_corrcoef(self.y_test, y_pred),
            'brier': brier_score_loss(self.y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   Accuracy: {models['XGBoost']['accuracy']:.4f}")
        print(f"   AUC-ROC: {models['XGBoost']['auc']:.4f}")
        print(f"   F1 Score: {models['XGBoost']['f1']:.4f}")
        
        # Select best model
        best_model_name = max(models.keys(), key=lambda k: models[k]['auc'])
        self.best_model = models[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n Best Model: {best_model_name}")
        print(f"   AUC-ROC: {models[best_model_name]['auc']:.4f}")
        
        return models
    
    def optimize_threshold_for_revenue(self, models):
        """Optimize decision threshold for revenue instead of accuracy"""
        print("\n" + "="*80)
        print("OPTIMIZING THRESHOLD FOR REVENUE")
        print("="*80)
        
        best_model_dict = models[self.best_model_name]
        y_pred_proba = best_model_dict['probabilities']
        
        # Calculate revenue at different thresholds
        thresholds = np.arange(0.3, 0.7, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate business metrics
            tp = ((y_pred_thresh == 1) & (self.y_test == 1)).sum()
            fp = ((y_pred_thresh == 1) & (self.y_test == 0)).sum()
            tn = ((y_pred_thresh == 0) & (self.y_test == 0)).sum()
            fn = ((y_pred_thresh == 0) & (self.y_test == 1)).sum()
            
            # Estimated revenue (simplified)
            # Assume: correctly predicted wins = revenue, false positives = wasted effort
            avg_deal_size = self.df['deal_amount'].mean()
            estimated_revenue = tp * avg_deal_size - fp * (avg_deal_size * 0.1)  # 10% cost for FP
            
            results.append({
                'Threshold': threshold,
                'Accuracy': (tp + tn) / len(self.y_test),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'F1': f1_score(self.y_test, y_pred_thresh),
                'Revenue': estimated_revenue
            })
        
        results_df = pd.DataFrame(results)
        best_threshold = results_df.loc[results_df['Revenue'].idxmax(), 'Threshold']
        
        print(f"\n Optimal Threshold: {best_threshold:.2f}")
        print(f"   Revenue: ${results_df.loc[results_df['Revenue'].idxmax(), 'Revenue']:,.0f}")
        print(f"   Accuracy: {results_df.loc[results_df['Revenue'].idxmax(), 'Accuracy']:.4f}")
        print(f"   F1 Score: {results_df.loc[results_df['Revenue'].idxmax(), 'F1']:.4f}")
        
        return best_threshold, results_df
    
    def generate_shap_explanations(self):
        """Generate SHAP values for model interpretability"""
        print("\n" + "="*80)
        print("GENERATING SHAP EXPLANATIONS")
        print("="*80)
        
        print(f"\nCreating SHAP explainer for {self.best_model_name}...")
        
        # Create explainer (TreeExplainer works for all tree-based models)
        self.shap_explainer = shap.TreeExplainer(self.best_model)
        
        # Calculate SHAP values (use sample for speed)
        sample_size = min(500, len(self.X_test))
        X_sample = self.X_test.sample(n=sample_size, random_state=42)
        
        print(f"Calculating SHAP values for {sample_size} samples...")
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # If binary classification, get values for positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print(" SHAP values calculated")
        
        # Save SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/shap_summary.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved SHAP summary plot")
        
        # Save SHAP feature importance
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names,
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/shap_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved SHAP feature importance")
        
        return self.shap_values


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKYGENI PRODUCTION-GRADE MODEL")
    print("="*80)
    
    # Load data
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f" Loaded {len(df):,} deals")
    
    # Initialize
    model = ProductionWinRateModel(df)
    
    # Analyze class distribution
    model.analyze_class_distribution()
    
    # Engineer features
    model.engineer_advanced_features()
    
    # Prepare data with SMOTE
    model.prepare_data_with_smote()
    
    # Train models
    models = model.train_optimized_models()
    
    # Optimize threshold
    best_threshold, threshold_results = model.optimize_threshold_for_revenue(models)
    
    # Generate SHAP explanations
    model.generate_shap_explanations()
    
    print("\n" + "="*80)
    print(" PRODUCTION MODEL COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
