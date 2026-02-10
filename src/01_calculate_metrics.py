"""
SkyGeni Sales Intelligence - Custom Metrics Calculator
Purpose: Calculate 7 custom metrics for advanced sales analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SalesMetricsCalculator:
    """
    Calculate custom metrics for sales performance analysis
    """
    
    def __init__(self, df):
        """
        Initialize with sales dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Sales data with required columns
        """
        self.df = df.copy()
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data with necessary transformations"""
        # Ensure dates are datetime
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        self.df['closed_date'] = pd.to_datetime(self.df['closed_date'])
        
        # Binary outcome
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        
        # Time features
        self.df['created_quarter'] = self.df['created_date'].dt.to_period('Q').astype(str)
        self.df['closed_quarter'] = self.df['closed_date'].dt.to_period('Q').astype(str)
        self.df['closed_month'] = self.df['closed_date'].dt.to_period('M').astype(str)
        
        print("Data prepared successfully")
    
    def calculate_pipeline_quality_score(self):
        """
        Metric 1: Pipeline Quality Score
        
        Weighted composite score based on historical win rates of similar deals
        
        Formula:
        PQS = (Historical_WR × 0.40) + (Velocity_Score × 0.30) + 
              (Rep_Performance × 0.20) + (Lead_Source_Quality × 0.10)
        
        Range: 0-100
        Interpretation: >70 = High quality, 50-70 = Medium, <50 = Low
        """
        print("\nCalculating Pipeline Quality Score...")
        
        # Component 1: Historical win rate by segment (40%)
        segment_wr = self.df.groupby(['industry', 'product_type', 'region'])['won'].mean()
        self.df['segment_key'] = self.df['industry'] + '_' + self.df['product_type'] + '_' + self.df['region']
        segment_wr_dict = segment_wr.to_dict()
        
        # Map to dataframe
        self.df['hist_wr_component'] = self.df.apply(
            lambda x: segment_wr_dict.get((x['industry'], x['product_type'], x['region']), 0.45) * 40,
            axis=1
        )
        
        # Component 2: Deal velocity (30%)
        # Expected cycle by deal size
        median_cycle = self.df.groupby(pd.cut(self.df['deal_amount'], 
                                                bins=[0, 10000, 30000, 60000, 100000]))['sales_cycle_days'].median()
        
        self.df['deal_size_bucket'] = pd.cut(self.df['deal_amount'], 
                                               bins=[0, 10000, 30000, 60000, 100000])
        self.df['expected_cycle'] = self.df['deal_size_bucket'].map(median_cycle)
        self.df['velocity_score'] = np.clip((self.df['expected_cycle'] / self.df['sales_cycle_days']) * 30, 0, 30)
        
        # Component 3: Rep historical performance (20%)
        rep_wr = self.df.groupby('sales_rep_id')['won'].mean()
        self.df['rep_component'] = self.df['sales_rep_id'].map(rep_wr) * 20
        
        # Component 4: Lead source quality (10%)
        source_wr = self.df.groupby('lead_source')['won'].mean()
        self.df['source_component'] = self.df['lead_source'].map(source_wr) * 10
        
        # Calculate final PQS
        self.df['pipeline_quality_score'] = (
            self.df['hist_wr_component'] + 
            self.df['velocity_score'] + 
            self.df['rep_component'] + 
            self.df['source_component']
        ).round(1)
        
        print(f"   Mean PQS: {self.df['pipeline_quality_score'].mean():.1f}")
        print(f"   High Quality Deals (>70): {(self.df['pipeline_quality_score'] > 70).sum():,}")
        print(f"   Low Quality Deals (<50): {(self.df['pipeline_quality_score'] < 50).sum():,}")
        
        return self.df['pipeline_quality_score']
    
    def calculate_deal_velocity_index(self):
        """
        Metric 2: Deal Velocity Index
        
        Measures how fast a deal is progressing compared to expected timeline
        
        Formula: DVI = Expected_Cycle / Actual_Cycle
        
        Interpretation:
        - DVI > 1.0: Faster than expected (good)
        - DVI = 1.0: On track
        - DVI < 1.0: Slower than expected (risk)
        """
        print("\nCalculating Deal Velocity Index...")
        
        # Calculate expected cycle by multiple factors
        expected_cycles = self.df.groupby(['industry', 'product_type'])['sales_cycle_days'].median()
        
        self.df['segment_key_velocity'] = self.df['industry'] + '_' + self.df['product_type']
        expected_dict = expected_cycles.to_dict()
        
        self.df['expected_cycle_days'] = self.df.apply(
            lambda x: expected_dict.get((x['industry'], x['product_type']), 65),
            axis=1
        )
        
        self.df['deal_velocity_index'] = (
            self.df['expected_cycle_days'] / self.df['sales_cycle_days']
        ).round(2)
        
        print(f"   Mean DVI: {self.df['deal_velocity_index'].mean():.2f}")
        print(f"   Fast Deals (DVI > 1.2): {(self.df['deal_velocity_index'] > 1.2).sum():,}")
        print(f"   Slow Deals (DVI < 0.8): {(self.df['deal_velocity_index'] < 0.8).sum():,}")
        
        return self.df['deal_velocity_index']
    
    def calculate_rep_efficiency_ratio(self):
        """
        Metric 3: Rep Efficiency Ratio
        
        Measures how much revenue a rep actually closes vs. what they work on
        
        Formula: RER = Total_Won_ACV / Total_Pipeline_ACV
        
        Interpretation:
        - Higher is better (more efficient at closing)
        - Benchmark: Top quartile reps
        """
        print("\nCalculating Rep Efficiency Ratio...")
        
        rep_stats = self.df.groupby('sales_rep_id').agg({
            'deal_amount': 'sum',
            'won': 'sum'
        })
        
        won_acv = self.df[self.df['won'] == 1].groupby('sales_rep_id')['deal_amount'].sum()
        total_acv = self.df.groupby('sales_rep_id')['deal_amount'].sum()
        
        rep_efficiency = (won_acv / total_acv).fillna(0)
        
        self.df['rep_efficiency_ratio'] = self.df['sales_rep_id'].map(rep_efficiency).round(3)
        
        print(f"   Mean RER: {rep_efficiency.mean():.3f}")
        print(f"   Top Quartile: {rep_efficiency.quantile(0.75):.3f}")
        print(f"   Bottom Quartile: {rep_efficiency.quantile(0.25):.3f}")
        
        return self.df['rep_efficiency_ratio']
    
    def calculate_lead_source_roi(self):
        """
        Metric 4: Lead Source ROI
        
        Measures revenue efficiency by channel
        
        Formula: LS_ROI = (Win_Rate × Avg_Deal_Size) / Avg_Sales_Cycle
        
        Interpretation: Revenue generated per day from each lead source
        Higher values indicate more efficient channels
        """
        print("\nCalculating Lead Source ROI...")
        
        source_stats = self.df.groupby('lead_source').agg({
            'won': 'mean',
            'deal_amount': 'mean',
            'sales_cycle_days': 'mean'
        })
        
        source_stats['lead_source_roi'] = (
            (source_stats['won'] * source_stats['deal_amount']) / 
            source_stats['sales_cycle_days']
        ).round(0)
        
        self.df['lead_source_roi'] = self.df['lead_source'].map(source_stats['lead_source_roi'])
        
        print("\n   Lead Source ROI ($/day):")
        for source, roi in source_stats['lead_source_roi'].sort_values(ascending=False).items():
            print(f"   {source}: ${roi:,.0f}/day")
        
        return self.df['lead_source_roi']
    
    def calculate_segment_attractiveness_index(self):
        """
        Metric 5: Segment Attractiveness Index
        
        Identifies most valuable segments to pursue
        
        Formula: SAI = (Win_Rate × Deal_Size × Volume^0.5) / Sales_Cycle
        
        Interpretation: Higher values = more attractive segments
        Volume is square-rooted to prevent over-weighting high-volume segments
        """
        print("\nCalculating Segment Attractiveness Index...")
        
        segment_stats = self.df.groupby(['industry', 'region']).agg({
            'won': 'mean',
            'deal_amount': 'mean',
            'sales_cycle_days': 'mean',
            'deal_id': 'count'
        })
        segment_stats.columns = ['win_rate', 'avg_deal', 'avg_cycle', 'volume']
        
        segment_stats['sai'] = (
            (segment_stats['win_rate'] * segment_stats['avg_deal'] * 
             np.sqrt(segment_stats['volume'])) / segment_stats['avg_cycle']
        ).round(0)
        
        # Map to dataframe
        sai_dict = segment_stats['sai'].to_dict()
        self.df['segment_attractiveness_index'] = self.df.apply(
            lambda x: sai_dict.get((x['industry'], x['region']), 0),
            axis=1
        )
        
        print("\n   Top 5 Most Attractive Segments:")
        top_segments = segment_stats.nlargest(5, 'sai')
        for idx, row in top_segments.iterrows():
            print(f"   {idx}: SAI = {row['sai']:,.0f}")
        
        return self.df['segment_attractiveness_index']
    
    def calculate_pipeline_decay_rate(self):
        """
        Metric 6: Pipeline Decay Rate
        
        Tracks pipeline health over time
        
        Formula: PDR = (Lost_Pipeline_Value / Total_Pipeline_Value) × 100
        
        Calculated monthly to track trends
        Interpretation: Lower is better, increasing trend is warning sign
        """
        print("\nCalculating Pipeline Decay Rate...")
        
        monthly_decay = self.df.groupby('closed_month').agg({
            'deal_amount': 'sum',
            'won': lambda x: (x == 0).sum()
        })
        monthly_decay.columns = ['total_value', 'lost_count']
        
        lost_value = self.df[self.df['won'] == 0].groupby('closed_month')['deal_amount'].sum()
        monthly_decay['lost_value'] = lost_value
        monthly_decay['pipeline_decay_rate'] = (
            (monthly_decay['lost_value'] / monthly_decay['total_value']) * 100
        ).round(1)
        
        # Map to dataframe
        self.df['pipeline_decay_rate'] = self.df['closed_month'].map(
            monthly_decay['pipeline_decay_rate']
        )
        
        print(f"   Mean Monthly Decay Rate: {monthly_decay['pipeline_decay_rate'].mean():.1f}%")
        print(f"   Recent Trend: {monthly_decay['pipeline_decay_rate'].tail(3).mean():.1f}%")
        
        return self.df['pipeline_decay_rate']
    
    def calculate_deal_complexity_score(self):
        """
        Metric 7: Deal Complexity Score
        
        Predicts likelihood of extended sales cycle
        
        Formula: DCS = (Deal_Size_Normalized × 0.4) + (Industry_Complexity × 0.3) + 
                       (Product_Complexity × 0.2) + (Region_Complexity × 0.1)
        
        Range: 0-100
        Interpretation: >70 = High complexity, likely longer cycle
        """
        print("\nCalculating Deal Complexity Score...")
        
        # Component 1: Deal size (normalized to 0-40)
        deal_size_normalized = (
            (self.df['deal_amount'] - self.df['deal_amount'].min()) / 
            (self.df['deal_amount'].max() - self.df['deal_amount'].min()) * 40
        )
        
        # Component 2: Industry complexity (based on avg cycle)
        industry_cycle = self.df.groupby('industry')['sales_cycle_days'].mean()
        industry_complexity = (
            (industry_cycle - industry_cycle.min()) / 
            (industry_cycle.max() - industry_cycle.min()) * 30
        )
        self.df['industry_complexity'] = self.df['industry'].map(industry_complexity)
        
        # Component 3: Product complexity
        product_complexity_map = {
            'Core': 10,
            'Pro': 15,
            'Enterprise': 20
        }
        self.df['product_complexity'] = self.df['product_type'].map(product_complexity_map)
        
        # Component 4: Region complexity
        region_cycle = self.df.groupby('region')['sales_cycle_days'].mean()
        region_complexity = (
            (region_cycle - region_cycle.min()) / 
            (region_cycle.max() - region_cycle.min()) * 10
        )
        self.df['region_complexity'] = self.df['region'].map(region_complexity)
        
        # Calculate final DCS
        self.df['deal_complexity_score'] = (
            deal_size_normalized + 
            self.df['industry_complexity'] + 
            self.df['product_complexity'] + 
            self.df['region_complexity']
        ).round(1)
        
        print(f"   Mean DCS: {self.df['deal_complexity_score'].mean():.1f}")
        print(f"   High Complexity Deals (>70): {(self.df['deal_complexity_score'] > 70).sum():,}")
        
        return self.df['deal_complexity_score']
    
    def calculate_all_metrics(self):
        """Calculate all custom metrics at once"""
        print("\n" + "="*80)
        print("CALCULATING ALL CUSTOM METRICS")
        print("="*80)
        
        self.calculate_pipeline_quality_score()
        self.calculate_deal_velocity_index()
        self.calculate_rep_efficiency_ratio()
        self.calculate_lead_source_roi()
        self.calculate_segment_attractiveness_index()
        self.calculate_pipeline_decay_rate()
        self.calculate_deal_complexity_score()
        
        print("\n" + "="*80)
        print("ALL METRICS CALCULATED SUCCESSFULLY")
        print("="*80)
        
        return self.df
    
    def get_metrics_summary(self):
        """Generate summary statistics for all metrics"""
        metrics = [
            'pipeline_quality_score',
            'deal_velocity_index',
            'rep_efficiency_ratio',
            'lead_source_roi',
            'segment_attractiveness_index',
            'pipeline_decay_rate',
            'deal_complexity_score'
        ]
        
        summary = pd.DataFrame()
        for metric in metrics:
            if metric in self.df.columns:
                summary[metric] = self.df[metric].describe()
        
        return summary.T
    
    def export_metrics(self, filepath):
        """Export dataframe with all metrics"""
        self.df.to_csv(filepath, index=False)
        print(f"\nMetrics exported to: {filepath}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKYGENI SALES INTELLIGENCE - CUSTOM METRICS CALCULATOR")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f"Loaded {len(df):,} deals")
    
    # Initialize calculator
    calculator = SalesMetricsCalculator(df)
    
    # Calculate all metrics
    df_with_metrics = calculator.calculate_all_metrics()
    
    # Get summary
    print("\n" + "="*80)
    print("METRICS SUMMARY STATISTICS")
    print("="*80)
    print(calculator.get_metrics_summary())
    
    # Export
    calculator.export_metrics('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data_with_metrics.csv')
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE DEALS WITH METRICS")
    print("="*80)
    sample_cols = [
        'deal_id', 'outcome', 'pipeline_quality_score', 'deal_velocity_index',
        'rep_efficiency_ratio', 'deal_complexity_score'
    ]
    print(df_with_metrics[sample_cols].head(10))
    
    print("\n" + "="*80)
    print("CUSTOM METRICS CALCULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
