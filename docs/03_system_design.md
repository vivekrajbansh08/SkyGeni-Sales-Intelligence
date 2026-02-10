# Part 3: System Design - Sales Insight & Alert System

## The Big Picture

If SkyGeni wanted to productize this analysis, here's how I'd build a lightweight system that actually gets used (not just built and forgotten).

**Core idea**: Keep it simple. Don't over-engineer. Focus on the 20% of features that deliver 80% of the value.

---

## High-Level Architecture (Sketch)

Here's the flow I'm imagining:

```
1. DATA LAYER
   ↓
   Salesforce CRM (pull data daily at 2am)
   ↓
2. FEATURE ENGINEERING
   ↓
   Calculate metrics: rep win rate, deal velocity, pipeline quality, etc.
   ↓
3. MODEL LAYER
   ↓
   Run LightGBM to score all open deals (win probability 0-100%)
   ↓
4. ALERT LAYER
   ↓
   Identify top 20% riskiest deals
   Send alerts to reps (email + Slack)
   ↓
5. DASHBOARD
   ↓
   Simple view: Pipeline health, at-risk deals, rep performance
```

**Key design principle**: Start with batch processing (nightly), add real-time later if needed. Don't build real-time on day one—it's 10x more complex and sales teams don't need sub-second predictions.

---

## Data Flow (Daily Cycle)

### 2:00 AM - Pull Data from Salesforce
- Use Salesforce API to get all deals (new + updated)
- ~500-1000 deals per day
- Store in PostgreSQL

**What could go wrong**: API rate limits, authentication expires, schema changes  
**Fallback**: Manual CSV upload if API breaks

### 3:00 AM - Feature Engineering
- Calculate 30+ features for each deal:
  - Deal characteristics: log(amount), sales_cycle_days, deal_efficiency
  - Historical patterns: rep_win_rate, source_win_rate, industry_win_rate
  - Velocity metrics: cycle_vs_expected, deal_velocity_index
  - Interaction features: amount × cycle, rep_wr × source_wr

**What could go wrong**: Missing data (rep not in system, new industry category)  
**Fallback**: Use defaults (average values) for missing features

### 4:00 AM - Score All Open Deals
- Run LightGBM model on ~500-1000 open deals
- Output: Win probability (0-100%), risk score (HIGH/MEDIUM/LOW)
- Generate SHAP explanations for top 100 riskiest deals

**What could go wrong**: Model crashes, out of memory, predictions take too long  
**Fallback**: Use rule-based scoring if model fails

### 5:00 AM - Generate Alerts
- Identify deals that need attention:
  - Win probability < 30% AND value > $50K = HIGH PRIORITY
  - Sales cycle > expected + 20% = MEDIUM PRIORITY
  - Win probability dropped > 15pp in 7 days = MEDIUM PRIORITY

- **Alert fatigue prevention**: Max 3 alerts per rep per day, prioritized by deal_value × risk_score

**What could go wrong**: Too many alerts, reps ignore them  
**Fallback**: Adaptive thresholds (only alert on top 20% riskiest)

### 5:30 AM - Send Alerts
- Email digest to each rep (their top 3 at-risk deals)
- Slack message for HIGH priority only
- Dashboard updated

---

## Example Alerts

### High-Risk Deal Alert

```
🔴 HIGH PRIORITY

Deal: Acme Corp - Enterprise License ($125,000)
Win Probability: 24% (was 45% last week) ⬇️
Risk Level: HIGH

Why this deal is at risk:
Sales cycle 45% longer than expected
Stuck in Proposal stage for 21 days
No activity in last 14 days

Recommended actions:
1. Schedule executive alignment call this week
2. Send competitive comparison document
3. Propose pilot program to reduce risk

Potential revenue at risk: $125,000
```

### Rep Performance Alert (to Manager)

```
🟡 COACHING OPPORTUNITY

Rep: Sarah Johnson
Win Rate: 32% (Team Avg: 45%)
Deals at Risk: 5 out of 12 open deals

Performance insights:
📉 Win rate declined 8pp in last 30 days
📈 Strong in SaaS vertical (55% win rate)
📉 Struggling with Enterprise deals (25% win rate)

Suggested actions:
1. Pair with top performer for shadowing
2. Focus on SaaS deals (her strength)
3. Provide Enterprise sales training
```

---

## Execution Schedule

### Daily
- 2:00 AM: Data sync
- 3:00 AM: Feature engineering
- 4:00 AM: Batch scoring
- 5:00 AM: Alert generation
- 5:30 AM: Send alerts

### Weekly
- Monday: Pipeline health report (to VP Sales)
- Wednesday: Rep performance review (to managers)
- Friday: Model performance check (accuracy, drift)

### Monthly
- 1st: Retrain model on last 12 months of data
- 15th: Business review with sales leadership

---

## What Could Break (And How to Handle It)

### 1. Data Pipeline Fails (80% probability within 6 months)

**What breaks**: Salesforce API changes, authentication expires, network issues

**Symptoms**: No new predictions, alerts stop, dashboard shows stale data

**How to detect**: Monitor ETL job completion time (alert if >2x normal)

**How to fix**:
- Retry logic with exponential backoff
- Graceful degradation (use cached data up to 24 hours old)
- Manual CSV upload as last resort

---

### 2. Model Becomes Stale (100% probability within 6 months)

**What breaks**: Market conditions change, model predictions become systematically wrong

**Symptoms**: Accuracy drops from 54% → 48% (worse than random)

**How to detect**: Track predicted vs. actual win rates weekly

**How to fix**:
- Monthly automated retraining
- Drift detection (monitor feature distributions)
- A/B test new model on 10% traffic before full rollout

---

### 3. Alert Fatigue (60% probability within 3 months)

**What breaks**: Too many alerts, reps stop opening them

**Symptoms**: Alert open rate drops from 80% → 10%

**How to detect**: Track alert open rate and action rate

**How to fix**:
- Adaptive thresholds (only top 20% riskiest deals)
- Max 3 alerts per rep per day
- Prioritize by deal_value × risk_score
- "Was this helpful?" feedback button

---

### 4. Low Model Accuracy (100% probability - inherent limitation)

**Root cause**: Missing critical features (competitor presence, customer engagement, budget signals)

**Current accuracy**: 54% (barely better than random)  
**Potential with full data**: 65-75%

**How to handle**:
- Set expectations: "54% is normal for CRM-only data"
- Focus on high-confidence predictions (>70% or <30%)
- Emphasize explainability over accuracy
- Phase 2: Integrate marketing automation for engagement data
- Phase 3: Add competitor tracking

---

## Productization Roadmap (If I Had More Time)

### Phase 1: MVP (What I Built)
- Batch scoring for all open deals
- Basic email alerts
- Simple dashboard
- SHAP explanations

### Phase 2: Enhanced (Next 3 Months)
- Real-time API for on-demand predictions
- Salesforce plugin (show risk score in CRM)
- Slack integration for alerts
- Rep scorecards

### Phase 3: Advanced (6-12 Months)
- Marketing automation integration (email opens, demo attendance)
- Competitor tracking
- A/B testing framework
- Automated coaching recommendations

---

## Success Metrics

### Model Performance
-  AUC-ROC > 0.50 (better than random) - **Currently: 0.589**
-  Target: AUC > 0.55 by Month 6
-  Target: AUC > 0.65 by Year 2 (with full data)

### Business Impact
-  Improve win rate by 3pp within 6 months
-  Reduce sales cycle by 10% within 6 months
-  Increase forecast accuracy by 15%

### User Adoption
-  80% of reps use dashboard weekly
-  60% alert open rate
-  30% alert action rate

---

**What I'm worried about**:
-  Sales teams might not trust 54% accuracy
-  Data quality will degrade over time (reps stop updating CRM)
-  Model will drift as market changes
-  Might be too simple (sales leaders may want more features)

**What I'd do differently with more time**:
- Add competitor tracking from day one (biggest missing feature)
- Build A/B testing framework to validate alerts actually help
- Integrate with marketing automation for engagement data
- Create mobile app for reps on the go

---

## Next Steps

1. **Week 1-2**: Build MVP (data pipeline + batch scoring + basic alerts)
2. **Week 3**: Deploy to 10% of sales team (beta test)
3. **Week 4**: Collect feedback, iterate
4. **Month 2**: Full rollout if beta successful
5. **Month 3+**: Add Phase 2 features based on usage data

**My commitment**: This is a lightweight sketch, not a production spec. In a real engagement, I'd validate every design decision with sales teams, run pilots, and iterate based on feedback. But given the 6-8 hour time constraint, this is my best thinking on how to productize this analysis.

---
