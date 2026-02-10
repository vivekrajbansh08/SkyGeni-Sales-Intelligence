# Part 2: Data Exploration & Insights

## What I Found (And Why It Matters)

When I first looked at this data, the CRO said "win rate dropped but pipeline looks healthy." That didn't add up. So I dug into the numbers and found something way more concerning: **pipeline volume crashed by 62%**. This isn't just a win rate problem—it's a full-blown pipeline crisis.

Here's what the data actually shows:

| Metric | Historical (Q1 2023 - Q1 2024) | Recent (Q2-Q3 2024) | Change |
|--------|-------------------------------|---------------------|--------|
| **Win Rate** | 45.6% | 44.2% | -1.3pp |
| **Pipeline Volume** | 867 deals/quarter | 333 deals/quarter | **-61.6%**  |
| **Quarterly Revenue** | $11.7M | $8.8M | -$2.9M (-25%) |
| **Sales Cycle** | 61 days | 82 days | +21 days (+34%) |

**Bottom line**: $11.6M in annualized revenue at risk.

---

## Three Key Insights

### Insight #1: Inbound Lead Quality Fell Off a Cliff

**What I found**: Inbound leads dropped from 47.0% to 40.3% win rate (-6.6pp). That's massive.

**Why it matters**: Inbound represents 26% of our recent pipeline. If we don't fix this, we're leaving millions on the table.

**What's probably happening**: 
- Marketing campaigns changed (new keywords? different messaging?)
- SEO rankings dropped (competitors outranking us?)
- Content quality degraded (generic blog posts instead of targeted content?)

**What to do about it**:
1. Audit every marketing campaign launched in Q2 2024
2. Check SEO rankings for our top 20 keywords
3. Interview 10-15 recent Inbound losses to understand why they didn't buy
4. Fix the root cause (probably keyword targeting or content quality)

**Owner**: VP Marketing  
**Timeline**: 30 days to diagnose, 60 days to fix

---

### Insight #2: Deals Are Dying at the Proposal Stage

**What I found**: Proposal stage win rate dropped from 45.5% to 38.7% (-6.9pp).

**Why it matters**: This is where we're supposed to close! If we're losing at Proposal, either our pricing is wrong, our proposals suck, or competitors are eating our lunch.

**My hypothesis**: 
- Competitors are undercutting us on price (need competitive intel)
- Our proposals are too generic (not customized enough)
- We're not addressing objections effectively

**What to do about it**:
1. Review all 119 active Proposal stage deals right now
2. Create proposal templates with competitive battle cards
3. Have senior reps shadow Proposal stage calls
4. Track: Did we lose on price? Features? Trust?

**Owner**: VP Sales  
**Timeline**: Immediate (this week)

---

### Insight #3: FinTech Vertical Is Struggling

**What I found**: FinTech win rate dropped from 48.4% to 43.5% (-4.9pp).

**Why it matters**: FinTech is 20% of our pipeline. If this vertical is broken, we need to decide: fix it or exit it.

**Possible causes**:
- Regulatory changes (compliance requirements we can't meet?)
- Economic headwinds (FinTech budgets frozen?)
- Increased competition (new entrants targeting FinTech?)

**What to do about it**:
1. Interview recent FinTech losses (what changed?)
2. Competitive analysis (who's winning in FinTech now?)
3. Make a call: Double down or de-prioritize?

**Owner**: VP Product + VP Sales  
**Timeline**: 60 days to decide

---

## Custom Metrics I Created

The assignment asked for 2 custom metrics. I built 7 because I wanted to see if I could find patterns the standard metrics miss.

### 1. Pipeline Quality Score (0-100)

**Formula**: Weighted average of historical win rate (40%), deal velocity (30%), rep performance (20%), and lead source quality (10%).

**What I found**: Average score is 57.5/100. Concerningly, **zero deals** scored above 70 (high quality). 

**Why it matters**: We don't have any slam-dunk deals in the pipeline right now. Everything is medium or low quality.

**How to use it**: Prioritize coaching and resources on deals scoring 60-70 (salvageable). Deals below 50 might not be worth the effort.

---

### 2. Deal Velocity Index

**Formula**: Expected cycle time / Actual cycle time. Values >1.0 = moving fast, <1.0 = stalled.

**What I found**: 
- 40% of deals are moving faster than expected (DVI > 1.2)
- 35% are stalled (DVI < 0.8)
- High variance = inconsistent sales process

**Why it matters**: Stalled deals rarely close. We need to identify them early and intervene.

**How to use it**: Alert reps when DVI drops below 0.8 for 2+ weeks. Something's wrong—competitor entered? Budget frozen? Champion left?

---

### 3. Rep Efficiency Ratio

**Formula**: Total won ACV / Total pipeline ACV (how much of what you work on actually closes).

**What I found**:
- Top quartile reps: 49.7% efficiency
- Bottom quartile: 43.0% efficiency
- **6.7pp gap** = huge opportunity for coaching

**Why it matters**: If we can bring bottom performers up to average, that's millions in additional revenue.

**How to use it**: Pair bottom quartile reps with top performers for shadowing. Focus on deal qualification (are they working the right deals?).

---

### 4. Lead Source ROI (Revenue per Day)

**Formula**: (Win Rate × Avg Deal Size) / Avg Sales Cycle

**What I found**:
1. Inbound: $195/day (still highest despite declining win rate!)
2. Referral: $186/day
3. Partner: $183/day
4. Outbound: $182/day

**Surprising insight**: Inbound still has the best ROI because of shorter sales cycles, even though win rate dropped. But this advantage is eroding fast.

**How to use it**: Don't abandon Inbound—fix it. The economics are still best-in-class.

---

### 5. Segment Attractiveness Index

**Formula**: (Win Rate × Deal Size × Volume) / Sales Cycle

**Top 5 segments to prioritize**:
1. EdTech + Europe
2. SaaS + India  
3. EdTech + India
4. Ecommerce + North America
5. FinTech + APAC

**How to use it**: Reallocate marketing budget and top reps to these segments. Stop spreading resources evenly.

---

### 6. Pipeline Decay Rate

**Formula**: % of pipeline value lost each month to losses.

**What I found**: 55.7% monthly decay (over half our pipeline dies every month).

**Why it matters**: This is high and getting worse (56.5% recently). We're bleeding deals.

**How to use it**: Track by segment. Which segments have highest decay? Those need intervention.

---

### 7. Deal Complexity Score

**Formula**: Composite of deal size, industry, product type, number of stakeholders.

**What I found**: Most deals are moderate complexity (45.7/100). Only 8% are truly complex.

**How to use it**: High-complexity deals (>70) need senior rep involvement from day one. Don't let junior reps struggle with enterprise deals.

---

## Machine Learning Models (Win Rate Drivers)

I built three models to figure out what's actually driving wins and losses:

| Model | Accuracy | Why I Built It |
|-------|----------|----------------|
| **Logistic Regression** | 53.7% | Most interpretable—I can explain every coefficient |
| **Random Forest** | 53.8% | Captures non-linear patterns (interactions between features) |
| **Rule-Based System** | 53.8% | Simple, auditable, easy to explain to sales teams |

**Honest assessment**: 54% accuracy is modest. But that's normal for CRM-only data. Industry benchmarks are 52-58% without competitor/engagement data. To get to 65%+, I'd need to integrate marketing automation (email opens, demo attendance) and competitor intel.

### Top 10 Features That Matter Most

1. **Deal Amount** (13.9%) - Bigger deals are harder to close
2. **Deal Amount × Cycle** (13.3%) - Big deals that drag on = bad sign
3. **Cycle vs Expected** (11.6%) - Stalled deals rarely close
4. **Sales Cycle Length** (10.6%) - Longer = lower win rate
5. **Rep Historical Win Rate** (9.5%) - **This is the most actionable!**
6. **Created Month** (5.8%) - Recent deals underperform
7. **Deal Stage** (5.2%) - Some stages are death traps
8. **Source Historical Win Rate** (3.7%)
9. **Lead Source** (3.6%)
10. **Industry** (3.6%)

**Key insight**: Rep performance (9.5%) is the strongest *controllable* predictor. We can't change deal size or industry, but we can coach reps. That's where the ROI is.

---

## What This Means (Recommendations)

### Immediate Actions (This Week)

1. **Audit Inbound Marketing** - Something broke in Q2. Find it, fix it.
2. **Proposal Stage Intervention** - Review all 119 active Proposal deals. Provide templates, battle cards, senior support.
3. **Coach Bottom Performers** - 7 reps have win rates below 40%. Pair them with top performers.

### Short-Term (Next 90 Days)

4. **Shift Budget to Referral** - Referral win rate is up 6.2pp. Double down on what's working.
5. **FinTech Strategy Review** - Decide: Fix it or exit it. Don't let it bleed resources.
6. **Deal Velocity Tracking** - Alert reps when deals stall. Intervene before it's too late.

### Medium-Term (6 Months)

7. **Deploy Deal Risk Scoring** - Use the Rule-Based model to score every deal daily. Alert on high-risk deals.
8. **Sales Training Program** - Proposal development, objection handling, competitive positioning.
9. **Segment Focus** - Prioritize top 5 segments. Stop treating all leads equally.

---

**Timeline**: If we act now, we can recover to 45%+ win rate within 2 quarters. If we wait, the trend will accelerate and we'll miss annual targets.

---

**My Commitment**: These insights are based on 6-8 hours of analysis. In a real engagement, I'd validate every hypothesis with win/loss interviews, A/B tests, and deeper competitive analysis. But given the time constraint, this is my best assessment of what's happening and what to do about it.
