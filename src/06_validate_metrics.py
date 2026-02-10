
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MetricValidator:
    """
    Validate custom metrics and establish benchmarks
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        
    def validate_metric_predictive_power(self, metric_name):
        """Test if metric predicts win rate"""
        print(f"\n{'='*60}")
        print(f"VALIDATING: {metric_name}")
        print(f"{'='*60}")
        
        # 1. Correlation test
        correlation = self.df[metric_name].corr(self.df['won'])
        p_value = stats.pearsonr(self.df[metric_name], self.df['won'])[1]
        
        print(f"\n1️⃣ Correlation Analysis:")
        print(f"   Correlation: {correlation:.4f}")
        print(f"   P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f" Statistically significant (p < 0.05)")
        else:
            print(f"  NOT statistically significant (p >= 0.05)")
        
        # 2. Quartile analysis
        quartiles = pd.qcut(self.df[metric_name], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        quartile_wr = self.df.groupby(quartiles)['won'].agg(['mean', 'count'])
        quartile_wr.columns = ['Win Rate', 'Count']
        quartile_wr['Win Rate'] *= 100
        
        print(f"\n2️⃣ Quartile Analysis:")
        print(quartile_wr.to_string())
        
        wr_range = quartile_wr['Win Rate'].max() - quartile_wr['Win Rate'].min()
        print(f"\n   Win Rate Range: {wr_range:.1f}pp")
        
        if wr_range > 5:
            print(f" Strong discriminatory power (>5pp range)")
        elif wr_range > 2:
            print(f"  Moderate discriminatory power (2-5pp range)")
        else:
            print(f"  Weak discriminatory power (<2pp range)")
        
        # 3. High vs Low comparison
        median = self.df[metric_name].median()
        high_metric = self.df[self.df[metric_name] > median]
        low_metric = self.df[self.df[metric_name] <= median]
        
        wr_high = high_metric['won'].mean() * 100
        wr_low = low_metric['won'].mean() * 100
        wr_diff = wr_high - wr_low
        
        print(f"\n3️⃣ High vs Low Analysis:")
        print(f"   High {metric_name}: {wr_high:.1f}% win rate ({len(high_metric)} deals)")
        print(f"   Low {metric_name}: {wr_low:.1f}% win rate ({len(low_metric)} deals)")
        print(f"   Difference: {wr_diff:+.1f}pp")
        
        # 4. Statistical test
        chi2, p_chi2 = stats.chi2_contingency(pd.crosstab(
            self.df[metric_name] > median, self.df['won']
        ))[:2]
        
        print(f"\n4️⃣ Chi-Square Test:")
        print(f"   Chi-square: {chi2:.2f}")
        print(f"   P-value: {p_chi2:.6f}")
        
        if p_chi2 < 0.05:
            print(f" Significant relationship (p < 0.05)")
        else:
            print(f" No significant relationship")
        
        # Overall verdict
        print(f"\n{'='*60}")
        if p_value < 0.05 and wr_range > 5:
            print(f"VERDICT: {metric_name} is a STRONG predictor")
        elif p_value < 0.05 and wr_range > 2:
            print(f" VERDICT: {metric_name} is a MODERATE predictor")
        else:
            print(f"VERDICT: {metric_name} is a WEAK predictor")
        print(f"{'='*60}")
        
        return {
            'metric': metric_name,
            'correlation': correlation,
            'p_value': p_value,
            'wr_range': wr_range,
            'wr_diff': wr_diff,
            'verdict': 'Strong' if (p_value < 0.05 and wr_range > 5) else
                      'Moderate' if (p_value < 0.05 and wr_range > 2) else 'Weak'
        }
    
    def establish_benchmarks(self):
        """Establish benchmarks for custom metrics"""
        print("\n" + "="*80)
        print("ESTABLISHING METRIC BENCHMARKS")
        print("="*80)
        
        benchmarks = {}
        
        # Calculate metrics (simplified versions for validation)
        
        # 1. Pipeline Quality Score
        rep_wr = self.df.groupby('sales_rep_id')['won'].transform('mean')
        source_wr = self.df.groupby('lead_source')['won'].transform('mean')
        self.df['pqs'] = (rep_wr * 0.4 + source_wr * 0.3 + self.df['won'] * 0.3)
        
        benchmarks['Pipeline Quality Score'] = {
            'mean': self.df['pqs'].mean(),
            'median': self.df['pqs'].median(),
            'p25': self.df['pqs'].quantile(0.25),
            'p75': self.df['pqs'].quantile(0.75),
            'good_threshold': self.df['pqs'].quantile(0.75),
            'poor_threshold': self.df['pqs'].quantile(0.25)
        }
        
        # 2. Deal Velocity Index
        median_cycle = self.df.groupby(pd.cut(self.df['deal_amount'], bins=4))['sales_cycle_days'].transform('median')
        self.df['dvi'] = median_cycle / (self.df['sales_cycle_days'] + 1)
        
        benchmarks['Deal Velocity Index'] = {
            'mean': self.df['dvi'].mean(),
            'median': self.df['dvi'].median(),
            'p25': self.df['dvi'].quantile(0.25),
            'p75': self.df['dvi'].quantile(0.75),
            'good_threshold': self.df['dvi'].quantile(0.75),  # Faster than expected
            'poor_threshold': self.df['dvi'].quantile(0.25)   # Slower than expected
        }
        
        # 3. Rep Efficiency Ratio
        rep_won_value = self.df[self.df['won'] == 1].groupby('sales_rep_id')['deal_amount'].sum()
        rep_total_value = self.df.groupby('sales_rep_id')['deal_amount'].sum()
        rep_efficiency = (rep_won_value / rep_total_value).fillna(0)
        self.df['rer'] = self.df['sales_rep_id'].map(rep_efficiency)
        
        benchmarks['Rep Efficiency Ratio'] = {
            'mean': self.df['rer'].mean(),
            'median': self.df['rer'].median(),
            'p25': self.df['rer'].quantile(0.25),
            'p75': self.df['rer'].quantile(0.75),
            'good_threshold': self.df['rer'].quantile(0.75),
            'poor_threshold': self.df['rer'].quantile(0.25)
        }
        
        # 4. Lead Source ROI
        source_revenue = self.df[self.df['won'] == 1].groupby('lead_source')['deal_amount'].sum()
        source_days = self.df.groupby('lead_source')['sales_cycle_days'].sum()
        source_roi = (source_revenue / source_days).fillna(0)
        self.df['ls_roi'] = self.df['lead_source'].map(source_roi)
        
        benchmarks['Lead Source ROI'] = {
            'mean': self.df['ls_roi'].mean(),
            'median': self.df['ls_roi'].median(),
            'p25': self.df['ls_roi'].quantile(0.25),
            'p75': self.df['ls_roi'].quantile(0.75),
            'good_threshold': self.df['ls_roi'].quantile(0.75),
            'poor_threshold': self.df['ls_roi'].quantile(0.25)
        }
        
        # Print benchmarks
        print("\n📊 BENCHMARK SUMMARY:")
        print("\n" + "-"*80)
        
        for metric, values in benchmarks.items():
            print(f"\n{metric}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Median: {values['median']:.4f}")
            print(f"  25th Percentile: {values['p25']:.4f}")
            print(f"  75th Percentile: {values['p75']:.4f}")
            print(f"  Good Threshold (>P75): {values['good_threshold']:.4f}")
            print(f"  Poor Threshold (<P25): {values['poor_threshold']:.4f}")
        
        # Save benchmarks
        benchmark_df = pd.DataFrame(benchmarks).T
        benchmark_df.to_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/metric_benchmarks.csv')
        print("\n Saved benchmarks to data/metric_benchmarks.csv")
        
        return benchmarks
    
    def validate_all_metrics(self):
        """Validate all custom metrics"""
        print("\n" + "="*80)
        print("VALIDATING ALL CUSTOM METRICS")
        print("="*80)
        
        # Calculate metrics first
        self.establish_benchmarks()
        
        # Validate each metric
        results = []
        
        metrics_to_validate = ['pqs', 'dvi', 'rer', 'ls_roi']
        metric_names = [
            'Pipeline Quality Score',
            'Deal Velocity Index',
            'Rep Efficiency Ratio',
            'Lead Source ROI'
        ]
        
        for metric, name in zip(metrics_to_validate, metric_names):
            result = self.validate_metric_predictive_power(metric)
            result['full_name'] = name
            results.append(result)
        
        # Summary
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(results_df[['full_name', 'correlation', 'p_value', 'wr_range', 'verdict']].to_string(index=False))
        
        # Save results
        results_df.to_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/metric_validation_results.csv', index=False)
        print("\n Saved validation results to data/metric_validation_results.csv")
        
        return results_df


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKYGENI METRIC VALIDATION & BENCHMARKING")
    print("="*80)
    
    # Load data
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f" Loaded {len(df):,} deals")
    
    # Initialize validator
    validator = MetricValidator(df)
    
    # Validate all metrics
    results = validator.validate_all_metrics()
    
    print("\n" + "="*80)
    print("METRIC VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
