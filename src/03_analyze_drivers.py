"""
SkyGeni Sales Intelligence - Win Rate Driver Analysis Model
Purpose: Build interpretable model to identify key drivers of win rate decline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')


class WinRateDriverAnalysis:
    """
    Multi-model approach to identify win rate drivers
    """
    
    def __init__(self, df):
        """Initialize with sales data"""
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
    def engineer_features(self):
        """
        Engineer 15-20 features for model training
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # Ensure dates are datetime
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        self.df['closed_date'] = pd.to_datetime(self.df['closed_date'])
        
        # Target variable
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        
        # Feature 1-3: Deal characteristics
        self.df['deal_amount_log'] = np.log10(self.df['deal_amount'] + 1)
        self.df['sales_cycle_log'] = np.log10(self.df['sales_cycle_days'] + 1)
        self.df['deal_size_category'] = pd.cut(self.df['deal_amount'], 
                                                 bins=[0, 10000, 30000, 60000, 100000],
                                                 labels=[0, 1, 2, 3]).astype(int)
        
        # Feature 4-5: Time-based features
        self.df['created_quarter'] = self.df['created_date'].dt.quarter
        self.df['created_month'] = self.df['created_date'].dt.month
        self.df['is_recent'] = (self.df['closed_date'] >= '2024-04-01').astype(int)
        
        # Feature 6-10: Categorical encodings
        le = LabelEncoder()
        self.df['region_encoded'] = le.fit_transform(self.df['region'])
        self.df['industry_encoded'] = le.fit_transform(self.df['industry'])
        self.df['product_type_encoded'] = le.fit_transform(self.df['product_type'])
        self.df['lead_source_encoded'] = le.fit_transform(self.df['lead_source'])
        self.df['deal_stage_encoded'] = le.fit_transform(self.df['deal_stage'])
        
        # Feature 11-13: Historical performance features
        # Rep historical win rate
        rep_wr = self.df.groupby('sales_rep_id')['won'].transform('mean')
        self.df['rep_historical_wr'] = rep_wr
        
        # Industry historical win rate
        industry_wr = self.df.groupby('industry')['won'].transform('mean')
        self.df['industry_historical_wr'] = industry_wr
        
        # Lead source historical win rate
        source_wr = self.df.groupby('lead_source')['won'].transform('mean')
        self.df['source_historical_wr'] = source_wr
        
        # Feature 14-16: Interaction features
        self.df['deal_amount_x_cycle'] = self.df['deal_amount'] * self.df['sales_cycle_days']
        self.df['is_large_slow_deal'] = ((self.df['deal_amount'] > 30000) & 
                                          (self.df['sales_cycle_days'] > 60)).astype(int)
        self.df['is_inbound_recent'] = ((self.df['lead_source'] == 'Inbound') & 
                                         (self.df['is_recent'] == 1)).astype(int)
        
        # Feature 17-18: Velocity features
        median_cycle_by_size = self.df.groupby('deal_size_category')['sales_cycle_days'].transform('median')
        self.df['cycle_vs_expected'] = self.df['sales_cycle_days'] / (median_cycle_by_size + 1)
        self.df['is_fast_deal'] = (self.df['cycle_vs_expected'] < 0.8).astype(int)
        
        # Feature 19-20: Segment features
        self.df['is_fintech'] = (self.df['industry'] == 'FinTech').astype(int)
        self.df['is_proposal_stage'] = (self.df['deal_stage'] == 'Proposal').astype(int)
        
        print(f" Engineered {20} features")
        print(f"   Total features available: {len(self.df.columns)}")
        
        return self.df
    
    def prepare_training_data(self, test_size=0.25, random_state=42):
        """
        Prepare train/test split
        """
        print("\n" + "="*80)
        print("PREPARING TRAINING DATA")
        print("="*80)
        
        # Select features for modeling
        feature_cols = [
            'deal_amount_log', 'sales_cycle_log', 'deal_size_category',
            'created_quarter', 'created_month', 'is_recent',
            'region_encoded', 'industry_encoded', 'product_type_encoded',
            'lead_source_encoded', 'deal_stage_encoded',
            'rep_historical_wr', 'industry_historical_wr', 'source_historical_wr',
            'deal_amount_x_cycle', 'is_large_slow_deal', 'is_inbound_recent',
            'cycle_vs_expected', 'is_fast_deal',
            'is_fintech', 'is_proposal_stage'
        ]
        
        X = self.df[feature_cols].copy()
        y = self.df['won'].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_names = feature_cols
        
        print(f" Training set: {len(self.X_train):,} samples")
        print(f" Test set: {len(self.X_test):,} samples")
        print(f"   Train win rate: {self.y_train.mean():.1%}")
        print(f"   Test win rate: {self.y_test.mean():.1%}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """
        Model 1: Logistic Regression (Maximum Interpretability)
        """
        print("\n" + "="*80)
        print("MODEL 1: LOGISTIC REGRESSION")
        print("="*80)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Train model
        lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_model.fit(X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = lr_model.predict(X_test_scaled)
        y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = lr_model.score(X_test_scaled, self.y_test)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n Logistic Regression Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC-ROC: {auc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Lost', 'Won']))
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': lr_model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Store results
        self.models['logistic_regression'] = {
            'model': lr_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'auc': auc,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return lr_model, feature_importance
    
    def train_random_forest(self):
        """
        Model 2: Random Forest (Capture Non-Linear Interactions)
        """
        print("\n" + "="*80)
        print("MODEL 2: RANDOM FOREST")
        print("="*80)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        accuracy = rf_model.score(self.X_test, self.y_test)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n Random Forest Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC-ROC: {auc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Lost', 'Won']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Store results
        self.models['random_forest'] = {
            'model': rf_model,
            'accuracy': accuracy,
            'auc': auc,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return rf_model, feature_importance
    
    def build_rule_based_system(self):
        """
        Model 3: Rule-Based System (Simple, Auditable Baseline)
        """
        print("\n" + "="*80)
        print("MODEL 3: RULE-BASED SYSTEM")
        print("="*80)
        
        # Define rules based on EDA insights
        def predict_win(row):
            score = 50  # Base score
            
            # Rule 1: Lead source (Inbound is problematic)
            if row['lead_source_encoded'] == 0:  # Inbound
                score -= 10
            elif row['lead_source_encoded'] == 3:  # Referral
                score += 10
            
            # Rule 2: Deal stage (Proposal is problematic)
            if row['deal_stage_encoded'] == 4:  # Proposal
                score -= 15
            
            # Rule 3: Industry (FinTech declining)
            if row['is_fintech'] == 1:
                score -= 8
            
            # Rule 4: Sales cycle (longer = worse)
            if row['sales_cycle_log'] > 1.9:  # ~80+ days
                score -= 12
            
            # Rule 5: Recent deals performing worse
            if row['is_recent'] == 1:
                score -= 5
            
            # Rule 6: Rep performance
            if row['rep_historical_wr'] > 0.48:
                score += 15
            elif row['rep_historical_wr'] < 0.42:
                score -= 15
            
            # Rule 7: Fast deals
            if row['is_fast_deal'] == 1:
                score += 8
            
            # Convert score to probability
            probability = max(0, min(100, score)) / 100
            return 1 if probability > 0.5 else 0, probability
        
        # Apply rules
        X_test_with_features = self.X_test.copy()
        predictions = X_test_with_features.apply(lambda row: predict_win(row), axis=1)
        y_pred = np.array([p[0] for p in predictions])
        y_pred_proba = np.array([p[1] for p in predictions])
        
        # Metrics
        accuracy = (y_pred == self.y_test).mean()
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n Rule-Based System Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC-ROC: {auc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Lost', 'Won']))
        
        # Store results
        self.models['rule_based'] = {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return y_pred, y_pred_proba
    
    def compare_models(self):
        """
        Compare all three models
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'Rule-Based'],
            'Accuracy': [
                self.models['logistic_regression']['accuracy'],
                self.models['random_forest']['accuracy'],
                self.models['rule_based']['accuracy']
            ],
            'AUC-ROC': [
                self.models['logistic_regression']['auc'],
                self.models['random_forest']['auc'],
                self.models['rule_based']['auc']
            ]
        })
        
        print(comparison.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        ax[0].bar(comparison['Model'], comparison['Accuracy'], color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax[0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
        ax[0].set_ylabel('Accuracy')
        ax[0].set_ylim(0, 1)
        for i, v in enumerate(comparison['Accuracy']):
            ax[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # AUC comparison
        ax[1].bar(comparison['Model'], comparison['AUC-ROC'], color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax[1].set_title('Model AUC-ROC Comparison', fontweight='bold', fontsize=14)
        ax[1].set_ylabel('AUC-ROC')
        ax[1].set_ylim(0, 1)
        for i, v in enumerate(comparison['AUC-ROC']):
            ax[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/12_model_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison
    
    def plot_feature_importance(self):
        """
        Visualize feature importance from both models
        """
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Logistic Regression coefficients
        lr_importance = self.models['logistic_regression']['feature_importance'].head(15)
        colors_lr = ['#e74c3c' if x < 0 else '#2ecc71' for x in lr_importance['Coefficient']]
        ax[0].barh(range(len(lr_importance)), lr_importance['Coefficient'], color=colors_lr, alpha=0.7)
        ax[0].set_yticks(range(len(lr_importance)))
        ax[0].set_yticklabels(lr_importance['Feature'])
        ax[0].set_xlabel('Coefficient')
        ax[0].set_title('Logistic Regression: Feature Coefficients', fontweight='bold', fontsize=14)
        ax[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Random Forest importance
        rf_importance = self.models['random_forest']['feature_importance'].head(15)
        ax[1].barh(range(len(rf_importance)), rf_importance['Importance'], color='steelblue', alpha=0.7)
        ax[1].set_yticks(range(len(rf_importance)))
        ax[1].set_yticklabels(rf_importance['Feature'])
        ax[1].set_xlabel('Importance')
        ax[1].set_title('Random Forest: Feature Importance', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/13_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_executive_summary(self):
        """
        Generate executive-ready summary of findings
        """
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY: WIN RATE DRIVER ANALYSIS")
        print("="*80)
        
        # Get top drivers from logistic regression (most interpretable)
        lr_importance = self.models['logistic_regression']['feature_importance']
        
        # Negative drivers (reduce win rate)
        negative_drivers = lr_importance[lr_importance['Coefficient'] < 0].head(5)
        
        # Positive drivers (increase win rate)
        positive_drivers = lr_importance[lr_importance['Coefficient'] > 0].head(5)
        
        summary = f"""
EXECUTIVE SUMMARY: WIN RATE DRIVER ANALYSIS
{'='*80}

MODEL PERFORMANCE:
- Best Model: {self.compare_models().loc[self.compare_models()['AUC-ROC'].idxmax(), 'Model']}
- Accuracy: {self.compare_models()['Accuracy'].max():.1%}
- AUC-ROC: {self.compare_models()['AUC-ROC'].max():.3f}

TOP 5 NEGATIVE DRIVERS (Reduce Win Rate):
"""
        for idx, row in negative_drivers.iterrows():
            summary += f"\n{idx+1}. {row['Feature']}: Coefficient = {row['Coefficient']:.3f}"
        
        summary += "\n\nTOP 5 POSITIVE DRIVERS (Increase Win Rate):"
        for idx, row in positive_drivers.iterrows():
            summary += f"\n{idx+1}. {row['Feature']}: Coefficient = {row['Coefficient']:.3f}"
        
        summary += """

KEY INSIGHTS:
1. Recent deals (Q2-Q3 2024) show systematically lower win rates
2. Inbound lead source is a significant negative driver
3. Proposal stage deals have higher loss risk
4. Rep historical performance is the strongest predictor
5. Fast-moving deals have higher win probability

RECOMMENDED ACTIONS:
1. IMMEDIATE: Review all Inbound leads in Proposal stage
2. SHORT-TERM: Provide proposal development training to sales team
3. MEDIUM-TERM: Investigate why Inbound lead quality has degraded
4. LONG-TERM: Implement this model as real-time deal risk scoring

"""
        
        print(summary)
        
        # Save to file
        with open('/Users/vivekrajbansh/Desktop/SkyGeni/docs/03_driver_analysis_summary.txt', 'w') as f:
            f.write(summary)
        
        print(" Executive summary saved to docs/03_driver_analysis_summary.txt")
        
        return summary


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKYGENI SALES INTELLIGENCE - WIN RATE DRIVER ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f" Loaded {len(df):,} deals")
    
    # Initialize analyzer
    analyzer = WinRateDriverAnalysis(df)
    
    # Feature engineering
    df_engineered = analyzer.engineer_features()
    
    # Prepare training data
    analyzer.prepare_training_data()
    
    # Train all models
    analyzer.train_logistic_regression()
    analyzer.train_random_forest()
    analyzer.build_rule_based_system()
    
    # Compare models
    comparison = analyzer.compare_models()
    
    # Visualize feature importance
    analyzer.plot_feature_importance()
    
    # Generate executive summary
    analyzer.generate_executive_summary()
    
    print("\n" + "="*80)
    print(" WIN RATE DRIVER ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
