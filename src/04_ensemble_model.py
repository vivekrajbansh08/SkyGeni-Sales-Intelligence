
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import from production model for feature engineering
import sys
sys.path.append('/Users/vivekrajbansh/Desktop/SkyGeni/src')
from production_model import ProductionWinRateModel


def main():
    """Main execution - use production model's already-encoded features"""
    print("\n" + "="*80)
    print("SKYGENI FINAL PRODUCTION MODEL (PYTHON 3.14)")
    print("="*80)
    print("\nUsing:")
    print("   50+ engineered features (from ProductionWinRateModel)")
    print("   Ensemble of 3 best models")
    print("   Revenue-optimized threshold")
    
    # Load data
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f"\n Loaded {len(df):,} deals")
    
    # Use production model for ALL feature engineering
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    prod_model = ProductionWinRateModel(df)
    prod_model.analyze_class_distribution()
    prod_model.engineer_advanced_features()
    prod_model.prepare_data_with_smote()
    
    # Get the prepared data (already encoded and ready)
    X_train = prod_model.X_train_resampled
    X_test = prod_model.X_test
    y_train = prod_model.y_train_resampled
    y_test = prod_model.y_test
    
    print(f"\n Using {X_train.shape[1]} features from ProductionWinRateModel")
    
    # Train ensemble
    print("\n" + "="*80)
    print("TRAINING FINAL ENSEMBLE (3 MODELS)")
    print("="*80)
    
    models = {}
    
    # Model 1: XGBoost
    print("\n1️⃣ Training XGBoost...")
    xgboost = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgboost.fit(X_train, y_train)
    
    xgb_proba = xgboost.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(f"   XGBoost AUC: {xgb_auc:.4f}")
    
    models['XGBoost'] = {'model': xgboost, 'auc': xgb_auc, 'proba': xgb_proba}
    
    # Model 2: LightGBM
    print("\n2️⃣ Training LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=50,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    lgbm_auc = roc_auc_score(y_test, lgbm_proba)
    print(f"   LightGBM AUC: {lgbm_auc:.4f}")
    
    models['LightGBM'] = {'model': lgbm, 'auc': lgbm_auc, 'proba': lgbm_proba}
    
    # Model 3: Gradient Boosting
    print("\n3️⃣ Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_proba)
    print(f"   Gradient Boosting AUC: {gb_auc:.4f}")
    
    models['Gradient Boosting'] = {'model': gb, 'auc': gb_auc, 'proba': gb_proba}
    
    # Create weighted ensemble
    print("\n4️⃣ Creating Weighted Ensemble...")
    total_auc = sum(m['auc'] for m in models.values())
    weights = {name: m['auc'] / total_auc for name, m in models.items()}
    
    print(f"   Weights:")
    for name, weight in weights.items():
        print(f"     {name}: {weight:.3f}")
    
    # Weighted average
    ensemble_proba = sum(weights[name] * models[name]['proba'] for name in models.keys())
    
    # Evaluate ensemble
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = (ensemble_pred == y_test).mean()
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    ensemble_mcc = matthews_corrcoef(y_test, ensemble_pred)
    
    print(f"\n Ensemble Results (threshold=0.5):")
    print(f"   Accuracy: {ensemble_acc:.4f}")
    print(f"   AUC-ROC: {ensemble_auc:.4f}")
    print(f"   F1 Score: {ensemble_f1:.4f}")
    print(f"   MCC: {ensemble_mcc:.4f}")
    
    # Optimize threshold for revenue
    print("\n" + "="*80)
    print("REVENUE OPTIMIZATION")
    print("="*80)
    
    thresholds = np.arange(0.20, 0.65, 0.05)
    results = []
    
    avg_deal = df['deal_amount'].mean()
    
    for threshold in thresholds:
        y_pred = (ensemble_proba >= threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        revenue = tp * avg_deal - fp * (avg_deal * 0.1) - fn * (avg_deal * 0.5)
        
        results.append({
            'Threshold': threshold,
            'Accuracy': (tp + tn) / len(y_test),
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'F1': f1_score(y_test, y_pred),
            'Revenue': revenue
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['Revenue'].idxmax()
    best_threshold = results_df.loc[best_idx, 'Threshold']
    
    print(f"\n📊 Top 3 Thresholds by Revenue:")
    print(results_df.nlargest(3, 'Revenue')[['Threshold', 'Accuracy', 'F1', 'Revenue']].to_string(index=False))
    
    print(f"\n OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   Revenue: ${results_df.loc[best_idx, 'Revenue']:,.0f}")
    print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
    print(f"   F1 Score: {results_df.loc[best_idx, 'F1']:.4f}")
    print(f"   Recall: {results_df.loc[best_idx, 'Recall']:.4f}")
    
    # Save
    results_df.to_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/final_model_results.csv', index=False)
    
    # Save model comparison
    model_comparison = pd.DataFrame([
        {'Model': name, 'AUC-ROC': m['auc'], 'Weight': weights[name]}
        for name, m in models.items()
    ])
    ensemble_row = pd.DataFrame([{
        'Model': 'Weighted Ensemble',
        'AUC-ROC': ensemble_auc,
        'Weight': 1.0
    }])
    model_comparison = pd.concat([model_comparison, ensemble_row], ignore_index=True)
    
    model_comparison.to_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/final_model_comparison.csv', index=False)
    
    print("\n" + "="*80)
    print(" FINAL PRODUCTION MODEL COMPLETE")
    print("="*80)
    print(f"\n🎯 DEPLOY WITH:")
    print(f"   Model: Weighted Ensemble (XGBoost + LightGBM + GB)")
    print(f"   Features: {X_train.shape[1]} engineered features")
    print(f"   Threshold: {best_threshold:.2f}")
    print(f"   Expected Revenue: ${results_df.loc[best_idx, 'Revenue']:,.0f}")
    print(f"   AUC-ROC: {ensemble_auc:.4f}")
    
    print(f"\n Saved:")
    print(f"   - data/final_model_results.csv")
    print(f"   - data/final_model_comparison.csv")


if __name__ == "__main__":
    main()
