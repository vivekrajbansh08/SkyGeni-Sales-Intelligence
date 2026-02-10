
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set professional style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


class AdvancedVisualization:
    """
    Create publication-quality visualizations
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for visualization"""
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        self.df['closed_date'] = pd.to_datetime(self.df['closed_date'])
        self.df['won'] = (self.df['outcome'] == 'Won').astype(int)
        self.df['closed_quarter'] = self.df['closed_date'].dt.to_period('Q').astype(str)
        self.df['closed_month'] = self.df['closed_date'].dt.to_period('M').astype(str)
        
        # Identify recent vs historical
        quarters = sorted(self.df['closed_quarter'].unique())
        self.last_two_quarters = quarters[-2:]
        self.historical_quarters = quarters[:-2]
        
        self.df['period'] = self.df['closed_quarter'].apply(
            lambda x: 'Recent' if x in self.last_two_quarters else 'Historical'
        )
    
    def create_correlation_heatmap(self):
        """Advanced correlation heatmap with hierarchical clustering"""
        print("\n📊 Creating correlation heatmap...")
        
        # Select numerical features
        numerical_cols = [
            'deal_amount', 'sales_cycle_days', 'won'
        ]
        
        # Add encoded categorical features
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ['region', 'industry', 'product_type', 'lead_source', 'deal_stage']:
            self.df[f'{col}_num'] = le.fit_transform(self.df[col])
            numerical_cols.append(f'{col}_num')
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/correlation_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("    Saved correlation_heatmap.png")
    
    def create_distribution_comparison(self):
        """Compare distributions between won and lost deals"""
        print("\n📊 Creating distribution comparisons...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        features = [
            ('deal_amount', 'Deal Amount ($)', True),
            ('sales_cycle_days', 'Sales Cycle (Days)', False),
        ]
        
        for idx, (feature, label, use_log) in enumerate(features):
            ax = axes[idx]
            
            won_data = self.df[self.df['won'] == 1][feature]
            lost_data = self.df[self.df['won'] == 0][feature]
            
            if use_log:
                won_data = np.log10(won_data + 1)
                lost_data = np.log10(lost_data + 1)
                label = f'Log10({label})'
            
            # KDE plots
            won_data.plot(kind='kde', ax=ax, label='Won', linewidth=2.5, color='#2ecc71')
            lost_data.plot(kind='kde', ax=ax, label='Lost', linewidth=2.5, color='#e74c3c')
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{label} Distribution by Outcome', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistical test
            statistic, pvalue = stats.ks_2samp(won_data, lost_data)
            ax.text(0.05, 0.95, f'KS test p-value: {pvalue:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        # Categorical features
        categorical_features = [
            ('region', 'Region'),
            ('industry', 'Industry'),
            ('product_type', 'Product Type'),
            ('lead_source', 'Lead Source')
        ]
        
        for idx, (feature, label) in enumerate(categorical_features, start=2):
            ax = axes[idx]
            
            # Calculate win rates by category
            win_rates = self.df.groupby(feature)['won'].agg(['mean', 'count'])
            win_rates = win_rates[win_rates['count'] >= 50].sort_values('mean', ascending=False)
            
            colors = plt.cm.RdYlGn(win_rates['mean'])
            ax.barh(range(len(win_rates)), win_rates['mean'] * 100, color=colors, alpha=0.8)
            ax.set_yticks(range(len(win_rates)))
            ax.set_yticklabels(win_rates.index, fontsize=10)
            ax.set_xlabel('Win Rate (%)', fontsize=11)
            ax.set_title(f'Win Rate by {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, v in enumerate(win_rates['mean'] * 100):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/distribution_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved distribution_comparison.png")
    
    def create_time_series_analysis(self):
        """Advanced time series analysis with trend decomposition"""
        print("\n📊 Creating time series analysis...")
        
        # Monthly aggregation
        monthly = self.df.groupby('closed_month').agg({
            'won': ['sum', 'count', 'mean'],
            'deal_amount': 'sum',
            'sales_cycle_days': 'mean'
        })
        monthly.columns = ['Wins', 'Total', 'Win_Rate', 'Revenue', 'Avg_Cycle']
        monthly['Win_Rate_Pct'] = monthly['Win_Rate'] * 100
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        
        x = range(len(monthly))
        
        # Win rate with confidence intervals
        ax = axes[0]
        ax.plot(x, monthly['Win_Rate_Pct'], marker='o', linewidth=2.5, 
               markersize=6, color='#3498db', label='Win Rate')
        
        # Add moving average
        ma_3 = monthly['Win_Rate_Pct'].rolling(window=3, center=True).mean()
        ax.plot(x, ma_3, linewidth=3, color='#e74c3c', 
               label='3-Month MA', alpha=0.8)
        
        # Add trend line
        z = np.polyfit(x, monthly['Win_Rate_Pct'], 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", linewidth=2, color='#2ecc71', 
               label=f'Trend (slope={z[0]:.3f})', alpha=0.7)
        
        ax.set_xticks(x[::2])
        ax.set_xticklabels(monthly.index[::2], rotation=45, ha='right')
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate Time Series with Trend Analysis', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Volume analysis
        ax = axes[1]
        ax.bar(x, monthly['Total'], color='#9b59b6', alpha=0.7, label='Deal Volume')
        ax.set_xticks(x[::2])
        ax.set_xticklabels(monthly.index[::2], rotation=45, ha='right')
        ax.set_ylabel('Number of Deals', fontsize=12)
        ax.set_title('Deal Volume Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Revenue analysis
        ax = axes[2]
        ax.bar(x, monthly['Revenue'] / 1000000, color='#16a085', alpha=0.7, label='Revenue')
        ax.set_xticks(x[::2])
        ax.set_xticklabels(monthly.index[::2], rotation=45, ha='right')
        ax.set_ylabel('Revenue ($M)', fontsize=12)
        ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/time_series_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved time_series_analysis.png")
    
    def create_segmentation_heatmap(self):
        """2D heatmap showing win rates across multiple dimensions"""
        print("\n📊 Creating segmentation heatmap...")
        
        # Create pivot table
        pivot = self.df.pivot_table(
            values='won',
            index='industry',
            columns='region',
            aggfunc='mean'
        ) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=45, linewidths=1, cbar_kws={"label": "Win Rate (%)"},
                   ax=ax, vmin=35, vmax=55)
        
        ax.set_title('Win Rate Heatmap: Industry × Region', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Industry', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/segmentation_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved segmentation_heatmap.png")
    
    def create_funnel_analysis(self):
        """Advanced funnel visualization with conversion rates"""
        print("\n📊 Creating funnel analysis...")
        
        # Define funnel stages in order
        stage_order = ['Qualified', 'Demo', 'Proposal', 'Negotiation', 'Closed']
        
        # Calculate metrics for each stage
        stage_stats = []
        for stage in stage_order:
            stage_df = self.df[self.df['deal_stage'] == stage]
            stage_stats.append({
                'Stage': stage,
                'Count': len(stage_df),
                'Win_Rate': stage_df['won'].mean() * 100,
                'Avg_Amount': stage_df['deal_amount'].mean(),
                'Avg_Cycle': stage_df['sales_cycle_days'].mean()
            })
        
        stage_df = pd.DataFrame(stage_stats)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Funnel visualization
        ax = axes[0]
        max_count = stage_df['Count'].max()
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(stage_df)))
        
        for i, row in stage_df.iterrows():
            width = row['Count'] / max_count
            ax.barh(i, width, height=0.8, color=colors[i], 
                   edgecolor='black', linewidth=2)
            
            # Add text
            label = f"{row['Stage']}\n{row['Count']} deals\n{row['Win_Rate']:.1f}% WR"
            ax.text(width/2, i, label, ha='center', va='center',
                   fontweight='bold', fontsize=10)
        
        ax.set_yticks([])
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Relative Volume', fontsize=12)
        ax.set_title('Sales Funnel by Stage', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Stage metrics
        ax = axes[1]
        x = np.arange(len(stage_df))
        width = 0.35
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, stage_df['Win_Rate'], width, 
                      label='Win Rate (%)', color='#2ecc71', alpha=0.8)
        bars2 = ax2.bar(x + width/2, stage_df['Avg_Cycle'], width,
                       label='Avg Cycle (days)', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12, color='#2ecc71')
        ax2.set_ylabel('Avg Sales Cycle (days)', fontsize=12, color='#e74c3c')
        ax.set_title('Win Rate & Sales Cycle by Stage', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stage_df['Stage'], rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Add legends
        ax.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/funnel_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" saved funnel_analysis.png")
    
    def create_cohort_analysis(self):
        """Cohort analysis showing performance trends"""
        print("\n📊 Creating cohort analysis...")
        
        # Create cohorts by creation quarter
        self.df['created_quarter'] = self.df['created_date'].dt.to_period('Q').astype(str)
        
        cohort_stats = self.df.groupby('created_quarter').agg({
            'won': ['mean', 'count'],
            'deal_amount': 'mean',
            'sales_cycle_days': 'mean'
        })
        cohort_stats.columns = ['Win_Rate', 'Count', 'Avg_Amount', 'Avg_Cycle']
        cohort_stats['Win_Rate_Pct'] = cohort_stats['Win_Rate'] * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = range(len(cohort_stats))
        
        # Win rate by cohort
        ax = axes[0, 0]
        ax.plot(x, cohort_stats['Win_Rate_Pct'], marker='o', linewidth=2.5,
               markersize=8, color='#3498db')
        ax.fill_between(x, cohort_stats['Win_Rate_Pct'], alpha=0.3, color='#3498db')
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate by Creation Cohort', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Volume by cohort
        ax = axes[0, 1]
        ax.bar(x, cohort_stats['Count'], color='#9b59b6', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Deals', fontsize=12)
        ax.set_title('Deal Volume by Creation Cohort', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Average deal size
        ax = axes[1, 0]
        ax.plot(x, cohort_stats['Avg_Amount'], marker='s', linewidth=2.5,
               markersize=8, color='#16a085')
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Avg Deal Amount ($)', fontsize=12)
        ax.set_title('Average Deal Size by Cohort', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Sales cycle
        ax = axes[1, 1]
        ax.plot(x, cohort_stats['Avg_Cycle'], marker='^', linewidth=2.5,
               markersize=8, color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Avg Sales Cycle (days)', fontsize=12)
        ax.set_title('Sales Cycle by Cohort', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/vivekrajbansh/Desktop/SkyGeni/visualizations/cohort_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved cohort_analysis.png")
    
    def create_all_visualizations(self):
        """Generate all advanced visualizations"""
        print("\n" + "="*80)
        print("CREATING ADVANCED VISUALIZATIONS")
        print("="*80)
        
        self.create_correlation_heatmap()
        self.create_distribution_comparison()
        self.create_time_series_analysis()
        self.create_segmentation_heatmap()
        self.create_funnel_analysis()
        self.create_cohort_analysis()
        
        print("\n" + "="*80)
        print(" ALL VISUALIZATIONS CREATED")
        print("="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKYGENI ADVANCED VISUALIZATION SUITE")
    print("="*80)
    
    # Load data
    df = pd.read_csv('/Users/vivekrajbansh/Desktop/SkyGeni/data/sales_data.csv')
    print(f"Loaded {len(df):,} deals")
    
    # Create visualizations
    viz = AdvancedVisualization(df)
    viz.create_all_visualizations()


if __name__ == "__main__":
    main()
