# ML-Based Dynamic Threshold Calibration for GFX Cash
**Business Requirements + Technical Design**

**Version:** 1.0  
**Date:** February 11, 2026  
**Owner:** [Your Name / Team]  
**Status:** Draft  

---

## 1. Background and Context

The GFX cash product currently uses manually defined thresholds per currency or per currency group, primarily based on historical volatility and business judgment. These thresholds are static between review cycles and require significant manual effort to maintain, especially in volatile markets. This creates operational, risk, and compliance challenges.

This initiative aims to design and implement a machine learning–based solution for **dynamic, data-driven threshold calibration** with strong explainability and governance. Similar approaches are increasingly adopted in investment banks to manage changing volatility and reduce false positives in monitoring processes.

---

## 2. Business Requirements (BRD)

### 2.1 Objectives

- Replace or augment manual thresholds with dynamic, ML-driven thresholds that adjust to market conditions.
- Ensure thresholds adapt automatically to changing volatility and volatility regimes for different currency groups.
- Reduce false positives while maintaining or improving detection of genuine risk events or abnormal activities.
- Provide transparent, explainable logic suitable for internal audit, model risk, and external regulators.
- Enable scalable coverage across all relevant GFX cash currencies and desks.

### 2.2 In Scope

- GFX cash trades and positions across all relevant currencies and books.
- Thresholds related to:
  - Price moves
  - Intraday and daily P&L
  - Position / exposure changes
  - Volumes or turnover (if required)
- Daily (or more frequent) recalibration of thresholds and anomaly scores.
- Dashboards and reporting for Operations, Risk, and Compliance teams.

### 2.3 Out of Scope (Phase 1)

- Non-FX asset classes (e.g., equities, rates, credit).
- Complex derivatives (options, structured products), unless explicitly onboarded later.
- Real-time sub-second or high-frequency surveillance (initial focus is end-of-day or near-real-time batch).

### 2.4 Stakeholders

| Role | Department | Responsibility |
|------|------------|----------------|
| Business Owner | Markets Operations / GFX Operations | Requirements, UAT, sign-off |
| Risk | Market Risk, Operational Risk | Risk framework alignment |
| Compliance | Trade Surveillance / Conduct Risk | Regulatory alignment |
| Technology | Markets IT, Data Engineering, Model Development | Build and delivery |
| Governance | Model Risk Management / Internal Audit | Validation and oversight |

---

## 3. Functional Requirements

### 3.1 Currency Grouping

**Requirement:** The system must automatically group currency pairs into homogeneous volatility/liquidity clusters using quantitative features (e.g., realized volatility, spreads, volumes, correlations).

**Details:**
- Groupings must be recomputed on a scheduled basis (e.g., monthly or quarterly) and stored with effective dates.
- Users must be able to view:
  - Currency-to-cluster mapping
  - Cluster-level summary statistics (average volatility, spread, volume, etc.)

### 3.2 Dynamic Threshold Calculation

**Requirement:** The system must calculate per-metric, per-currency (or per-cluster) thresholds that update at least daily and depend on forecasted volatility and/or recent empirical distributions.

**Details:**
- Support multiple risk levels (e.g., "standard" 95%, "heightened" 99%)
- Allow minimum/maximum caps defined by business rules
- Support manual overrides with full audit trails (user, timestamp, reason)

### 3.3 Regime Awareness

**Requirement:** The system must detect volatility regimes (e.g., normal, elevated, crisis) per currency group using time-series models such as regime-switching or change-point detection.

**Details:**
- Thresholds must be adjustable based on regime (e.g., different multipliers by regime)
- Users must be able to view current regime and regime history

### 3.4 Anomaly Detection

**Requirement:** The system must compute anomaly scores for trades or aggregated activity (e.g., per trader per day, per desk per day) using unsupervised models (Isolation Forest, Autoencoder, or similar).

**Details:**
- Anomaly detection should complement threshold breaches, not replace them
- The system must flag observations with anomaly scores above calibrated cutoffs

### 3.5 Alerts and Workflows

**Requirement:** The system must generate alerts when a dynamic threshold is breached or an anomaly score exceeds its predefined limit.

**Details:**

Each alert must contain:
- Context information (trader, desk, book, instrument, time, values vs. thresholds, regime)
- A concise explanation of why the alert was triggered, where technically possible

Users must be able to:
- View, filter, sort alerts by different dimensions
- Record dispositions (true positive, false positive, informational)
- Add comments and attach supporting documentation

### 3.6 Feedback and Continuous Improvement

**Requirement:** The system must store alert history with features, dispositions, and outcomes to enable supervised learning in later phases.

**Details:**
- Store feature values and model outputs at alert time
- Store user disposition, comments, and follow-up outcomes
- Support development of supervised prioritization models that suggest threshold refinements

### 3.7 Reporting and Dashboards

**Requirement:** Provide comprehensive dashboards and reports for monitoring threshold performance and alert quality.

**Dashboard Elements:**
- Time series of thresholds and realized metric values
- Counts of breaches by currency, desk, trader, and regime
- Distribution and evolution of anomaly scores
- Trends in false positives vs. true positives over time

**Export Formats:** CSV, Excel, PDF

---

## 4. Non-Functional Requirements

### 4.1 Performance

- Daily threshold and anomaly computations must complete within the agreed batch window
- User interface queries (e.g., alert search) should respond within 3-5 seconds under typical workload

### 4.2 Availability and Reliability

- Target availability: ≥ 99% during business hours
- **Fallback behavior:** If the latest ML run fails, the system must:
  - Fall back to last successfully computed thresholds, or
  - Revert to pre-defined conservative manual thresholds
  - Display clear flag that thresholds are stale

### 4.3 Security and Access Control

**Role-Based Access Control (RBAC):**

| Role | Permissions |
|------|-------------|
| View-only | Read dashboards and alerts |
| Analyst | View-only + record dispositions |
| Supervisor | Analyst + modify threshold caps |
| Admin/Model Owner | All permissions + modify models and parameters |

- Handle sensitive data (e.g., trader IDs, client flags) according to internal data protection and compliance policies

### 4.4 Auditability

**Audit Trail Requirements:**

Maintain full audit trail of:
- Threshold values, effective dates, and associated model versions
- Model parameter changes and deployments
- Manual overrides with user ID, timestamp, and reason
- Alert creation, modifications, and dispositions

---

## 5. Technical Design (TDD)

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application & UI Layer                    │
│  (Dashboards, Alert Management, Threshold Configuration)    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Threshold Engine                         │
│     (Combines models + business rules → final thresholds)   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Feature & Model Layer                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Clustering  │  │  Volatility  │  │ Anomaly Detection│  │
│  │   Models    │  │   Models     │  │     Models       │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐                        │
│  │   Regime    │  │ Supervised   │                        │
│  │  Detection  │  │Alert Scoring │                        │
│  └─────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ Market Data  │ │ Trade Data   │ │ Alert Data   │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Logical Layers:**

1. **Data Layer:** Market data, trade data, alert/feedback data
2. **Feature & Model Layer:** ML components for clustering, volatility, regime detection, anomaly detection, supervised scoring
3. **Threshold Engine:** Service combining model outputs and business rules
4. **Application & UI Layer:** User interfaces and APIs
5. **Model Governance & Monitoring:** Model registry, versioning, performance monitoring

---

### 5.2 Data Design

#### 5.2.1 Core Entities

**CurrencyPair**
- currency_pair_id (PK)
- base_currency
- quote_currency
- status (active/inactive)

**MarketDataDaily**
- id (PK)
- currency_pair_id (FK)
- date
- mid_price, high, low, close
- bid, ask
- volume
- realized_volatility

**ClusterAssignment**
- id (PK)
- currency_pair_id (FK)
- cluster_id
- effective_from
- effective_to
- method_version

**ThresholdDefinition**
- id (PK)
- metric_name
- currency_pair_id or cluster_id
- date
- threshold_value
- regime
- model_version
- business_caps
- override_flag
- override_user
- override_reason

**Trade**
- trade_id (PK)
- datetime
- trader_id
- desk_id
- book
- currency_pair_id (FK)
- side (buy/sell)
- notional
- price
- venue
- client_flag

**AnomalyScore**
- id (PK)
- key (trade_id or aggregation_key)
- model_version
- score
- score_date
- threshold_used
- features_snapshot (JSON)

**Alert**
- alert_id (PK)
- alert_type (threshold/anomaly/both)
- datetime
- entity (trade_id/trader_id/desk_id)
- currency_pair_id (FK)
- metric_values (JSON)
- threshold_values (JSON)
- anomaly_scores (JSON)
- regime
- model_versions

**AlertDisposition**
- id (PK)
- alert_id (FK)
- user_id
- status (true_positive/false_positive/informational)
- comments
- decision_datetime

#### 5.2.2 Daily Data Flow

```
1. Data Ingestion
   ↓
2. Feature Engineering
   ↓
3. [Scheduled] Run Clustering → Update ClusterAssignment
   ↓
4. Run Volatility & Regime Models → Update ThresholdDefinition
   ↓
5. Run Anomaly Detection → Update AnomalyScore
   ↓
6. Apply Thresholds & Scores → Generate Alert
   ↓
7. User Reviews → Update AlertDisposition
```

---

### 5.3 Model Design

#### 5.3.1 Currency Grouping (Clustering)

**Input Features (per currency pair, rolling windows 20/60/180 days):**
- Realized volatility
- Volatility of volatility
- Average bid–ask spread
- Average daily trading volume
- Return skewness and kurtosis
- Correlation with benchmark pairs (EURUSD, USDJPY, etc.)

**Algorithm:**
- **Primary:** K-Means clustering
  - Determine optimal k using elbow method and silhouette scores
  - Refit monthly/quarterly
- **Alternatives to evaluate:** Hierarchical clustering, DBSCAN

**Outputs:**
- Cluster ID per currency pair
- Cluster centroids and summary statistics
- Stability metrics (how often pairs change clusters)

**Python Libraries:** scikit-learn (KMeans, DBSCAN), pandas, numpy

---

#### 5.3.2 Volatility Modeling

**Input:**
- Historical log returns per currency pair or representative series per cluster

**Algorithm:**
- **Primary:** GJR-GARCH(1,1) model
  - Captures volatility clustering and leverage effects
  - Industry-standard for FX volatility forecasting
- **Alternatives to evaluate:** LSTM/GRU, Hybrid GARCH-GRU

**Outputs:**
- One-step-ahead volatility forecast: σ̂(t+1)
- Optionally multi-step forecasts
- Model diagnostics: AIC/BIC, residual tests (Ljung-Box)

**Python Libraries:** arch (for GARCH), statsmodels

**Model Formula Example:**

For GJR-GARCH(1,1):

r_t = μ + ε_t  
ε_t = σ_t × z_t  
σ_t² = ω + α×ε²_(t-1) + γ×ε²_(t-1)×I_(t-1) + β×σ²_(t-1)

where:
- r_t = log return at time t
- σ_t = conditional volatility
- I_(t-1) = 1 if ε_(t-1) < 0, else 0 (captures leverage effect)

---

#### 5.3.3 Threshold Calculation

**Core Principle:** Thresholds are functions of forecasted volatility and regime, not static numbers.

**Formula:**

Threshold_t(metric) = k(regime) × σ̂_(t+1) × scaling_factor + business_caps

where:
- σ̂_(t+1) = forecasted volatility for next period
- k(regime) = multiplier depending on regime and risk appetite
  - Normal regime: k = 2.0
  - Elevated regime: k = 2.5
  - Crisis regime: k = 3.0
- scaling_factor = metric-specific calibration constant
- business_caps = min/max limits per business rules

**Alternative Approach:** Quantile-based thresholds
- Use historical distribution of (metric / σ)
- Set threshold as 95th, 99th, or 99.9th percentile

**Configuration per Metric:**

| Metric | Scaling Factor | Business Cap (Min) | Business Cap (Max) |
|--------|----------------|-------------------|-------------------|
| Price Move (bps) | 1.0 | 5 bps | 500 bps |
| Intraday P&L | 10,000 | $10k | $10M |
| Position Change | 1.5 | 1M notional | 100M notional |

---

#### 5.3.4 Regime Detection

**Input:**
- Time series of volatility, returns, or stress indicators

**Algorithms:**

**Option 1: Hidden Markov Model (HMM)**
- Model with 2-3 latent states (normal, elevated, crisis)
- Transition matrix captures regime-switching dynamics
- Python library: hmmlearn

**Option 2: Bayesian Change-Point Detection**
- Online detection of structural breaks in volatility
- No parameter tuning required
- Python library: bayesian-changepoint

**Outputs:**
- Regime label per date (0=normal, 1=elevated, 2=crisis)
- Regime transition probabilities
- Regime summary statistics (mean vol per regime, duration)

**HMM Example States:**

| Regime | Avg Volatility | Characteristics | k Multiplier |
|--------|----------------|-----------------|--------------|
| Normal | σ < 10% | Stable markets | 2.0 |
| Elevated | 10% ≤ σ < 20% | Moderate stress | 2.5 |
| Crisis | σ ≥ 20% | High stress | 3.0 |

---

#### 5.3.5 Anomaly Detection

**Features (trade-level and aggregated):**

**Trade-level:**
- Trade size (raw and normalized by trader/desk/currency average)
- Price deviation from mid (bps)
- Time-of-day anomaly score
- Venue (unusual venue for trader/desk)
- Client vs. house flag

**Behavior-relative:**
- Z-score of size vs. trader's 60-day average
- Z-score of size vs. desk's 60-day average
- Deviation from trader's typical time-of-day pattern

**Threshold-relative:**
- Distance to current dynamic threshold (as % of threshold)
- Current regime indicator

**Algorithms:**

**Primary: Isolation Forest**
- Ensemble method that isolates anomalies via random partitioning
- Works well with high-dimensional tabular data
- No labeled data required
- Python library: scikit-learn (IsolationForest)

**Hyperparameters:**
- contamination = 0.01 (expected % of anomalies)
- n_estimators = 100
- max_samples = 256

**Alternative: Autoencoder**
- Neural network trained to reconstruct normal patterns
- Anomalies identified by high reconstruction error
- Python library: TensorFlow/Keras

**Outputs:**
- Anomaly score per trade or aggregated entity (range: -1 to 1 or 0 to 1 depending on method)
- Binary anomaly flag (score > calibrated threshold)
- Top contributing features for explainability

**Calibration:**
- Set anomaly score threshold to achieve target false positive rate (e.g., 1-2%)
- Validate on held-out data with known normal/anomalous periods

---

#### 5.3.6 Supervised Alert Prioritization (Phase 4)

**Purpose:** Use historical alert dispositions to build a model that prioritizes alerts by estimated risk.

**Input Features:**
- All features from threshold and anomaly models
- Threshold breach magnitude
- Anomaly score
- Currency pair cluster
- Regime at alert time
- Trader/desk historical disposition rate
- Time features (day of week, time of day)

**Algorithm:**
- **Primary:** XGBoost or LightGBM
- Target variable: alert_is_true_positive (0/1)

**Outputs:**
- Risk score per alert (probability 0-1)
- Feature importance (SHAP values)
- Recommendations for threshold adjustments

**Training Approach:**
- Require minimum 3-6 months of dispositioned alerts before training
- Retrain monthly
- Handle class imbalance (typically 5-20% true positives) via:
  - Class weights
  - SMOTE oversampling
  - Stratified cross-validation

---

### 5.4 Application Layer

#### 5.4.1 Services / APIs

**Threshold Service**
- GET /thresholds?date={date}&currency_pair={pair}&metric={metric}
- POST /thresholds/recalculate (trigger batch job)
- POST /thresholds/override (manual override with reason)

**Alert Service**
- GET /alerts?date_from={date}&date_to={date}&desk={desk}&status={status}
- PUT /alerts/{alert_id}/disposition (update disposition)
- POST /alerts/{alert_id}/comments (add comment)

**Model Management Service**
- GET /models (list all models with versions)
- GET /models/{model_id}/performance (metrics, diagnostics)
- POST /models/{model_id}/deploy (promote to production)

**Feature Service**
- GET /features?currency_pair={pair}&date={date} (retrieve feature values)

#### 5.4.2 UI / Dashboards

**Dashboard 1: Threshold Monitor**
- Time series chart: threshold vs. actual metric
- Regime overlay (colored background)
- Cluster selector dropdown
- Date range selector

**Dashboard 2: Alert Workbench**
- Alert table with filters (date, desk, trader, currency, status)
- Alert detail panel:
  - Context (who, what, when, where)
  - Threshold breach details
  - Anomaly score and contributing features
  - Disposition workflow
  - Comment thread
- Export to Excel

**Dashboard 3: Model Performance (Admin)**
- Clustering stability metrics
- Volatility model diagnostics (forecast errors)
- Anomaly detection confusion matrix
- Alert disposition trends (TP/FP rates over time)

**Technology Options:**
- Internal web framework + React/Vue frontend
- Streamlit (rapid prototyping)
- Tableau/PowerBI (for reporting layer)

---

### 5.5 Operations and Monitoring

#### 5.5.1 Scheduling

**Daily Batch (Core Process):**
```
Time: 18:00 IST (after market close)
Duration: ~2-4 hours

Steps:
1. Data ingestion (market data + trades)
2. Feature engineering
3. Volatility model run → update thresholds
4. Regime detection → update regime flags
5. Anomaly detection → update scores
6. Alert generation
7. Dashboard refresh
```

**Weekly Batch:**
- Clustering model refit (if configured for weekly)
- Model performance reporting

**Monthly Batch:**
- Clustering model refit (default schedule)
- Model validation reports
- Archive old alerts/dispositions

**Ad-hoc:**
- On-demand threshold recalculation (e.g., after major market event)

#### 5.5.2 Fallback Behavior

**If model run fails:**

1. Attempt automatic retry (3 attempts with exponential backoff)
2. If still failing:
   - Use last successfully computed thresholds (T-1)
   - Flag all thresholds as "STALE" in UI
   - Send alert to model owner and support team
   - Optionally: fall back to pre-defined conservative manual thresholds

**If data source unavailable:**
- Market data: fall back to last available day
- Trade data: pause alert generation for affected period, log gap

#### 5.5.3 Monitoring

**Infrastructure Monitoring:**
- Job success/failure rates
- Job run times (SLA: < 4 hours)
- API response times (SLA: < 3 seconds for 95th percentile)

**Model Performance Monitoring:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Threshold breach rate | ~1% for 99% thresholds | < 0.5% or > 2% |
| Anomaly detection rate | ~1-2% | > 5% |
| Alert disposition rate (TP%) | 10-30% | < 5% or > 50% |
| Model forecast error (RMSE) | < baseline | > 150% of baseline |

**Dashboards:**
- Grafana/Prometheus for infrastructure metrics
- Internal model monitoring dashboard for ML metrics

---

### 5.6 Technology Stack

**Languages:**
- Python 3.9+ (core modeling and services)
- SQL (data queries)

**Key Python Libraries:**

| Purpose | Library |
|---------|---------|
| Data manipulation | pandas, numpy |
| Clustering | scikit-learn |
| Volatility modeling | arch, statsmodels |
| Regime detection | hmmlearn, bayesian-changepoint |
| Anomaly detection | scikit-learn (IsolationForest), TensorFlow/Keras |
| Supervised learning | xgboost, lightgbm, scikit-learn |
| Explainability | shap |
| Visualization | matplotlib, plotly |

**Infrastructure:**
- **Data Storage:** SQL database (PostgreSQL/Oracle) + object storage (S3/Azure Blob) for models
- **Orchestration:** Apache Airflow or internal scheduler
- **APIs:** FastAPI or Flask
- **UI:** Streamlit (prototype), React/Vue (production)
- **Model Registry:** MLflow or internal system
- **Monitoring:** Prometheus + Grafana, or internal monitoring

---

## 6. Testing and Validation

### 6.1 Unit Testing
- Test each model component independently
- Mock data for reproducibility
- Target: > 80% code coverage

### 6.2 Backtesting

**Objective:** Validate dynamic thresholds against historical data

**Approach:**
1. Run models on historical data (2-3 years)
2. Compare breach frequencies vs. target percentiles
3. Evaluate performance during known stress periods:
   - Brexit (June 2016)
   - COVID-19 (March 2020)
   - Recent volatility episodes
4. Compare dynamic vs. current manual thresholds:
   - False positive reduction
   - True positive retention

**Success Criteria:**
- Breach rates within ±20% of target percentiles
- ≥30% reduction in false positives vs. manual thresholds
- No degradation in detection of known historical issues

### 6.3 User Acceptance Testing (UAT)

**Duration:** 4-6 weeks

**Approach:**
- Run new system in parallel with existing process
- Users review side-by-side comparison
- Collect structured feedback via surveys
- Iteratively refine based on feedback

**UAT Exit Criteria:**
- ≥80% user satisfaction
- All critical and high-priority issues resolved
- Performance SLAs met

### 6.4 Model Validation

**Performed by:** Model Risk Management

**Validation Elements:**
- Conceptual soundness (model choice, assumptions)
- Data quality and representativeness
- Implementation verification (code review, replication)
- Outcome analysis (backtesting results)
- Limitations and use-test documentation

**Deliverables:**
- Model validation report
- Approval for production deployment

---

## 7. Implementation Phases

### Phase 1: Foundation and Baseline (Weeks 1-6)

**Objectives:**
- Build data pipelines
- Implement clustering
- Generate baseline static ML thresholds

**Deliverables:**
- Data ingestion pipeline (market + trade data)
- Feature engineering module
- Clustering model (K-Means)
- Cluster-based static thresholds
- Initial analysis report comparing manual vs. cluster thresholds

**Resources:**
- 1 Data Engineer
- 1 ML Engineer
- 0.5 Business Analyst

---

### Phase 2: Dynamic Thresholds (Weeks 7-14)

**Objectives:**
- Implement volatility models
- Build threshold engine
- Conduct comprehensive backtesting

**Deliverables:**
- GJR-GARCH volatility models per cluster
- Threshold calculation engine
- Backtesting framework and results
- UAT-ready prototype

**Resources:**
- 1 ML Engineer (lead)
- 1 Quantitative Analyst
- 1 Data Engineer
- 0.5 Business Analyst

---

### Phase 3: Anomaly Detection & UI (Weeks 15-22)

**Objectives:**
- Implement anomaly detection
- Build user interface
- Integrate alert workflow

**Deliverables:**
- Isolation Forest anomaly detection
- Alert generation service
- Dashboards (Threshold Monitor, Alert Workbench)
- Integrated system in UAT environment

**Resources:**
- 1 ML Engineer
- 2 Full-Stack Developers
- 1 UI/UX Designer
- 1 QA Engineer

---

### Phase 4: Feedback Loop & Production (Weeks 23+)

**Objectives:**
- Capture analyst feedback
- Build supervised alert scoring
- Production hardening
- Ongoing optimization

**Deliverables:**
- Alert disposition capture system
- Supervised XGBoost prioritization model
- Production deployment
- Model monitoring dashboards
- Documentation and runbooks

**Resources:**
- 1 ML Engineer
- 1 DevOps Engineer
- 1 Data Engineer
- 0.5 Model Risk Analyst

---

## 8. Success Metrics

### 8.1 Quantitative Metrics

| Metric | Baseline (Manual) | Target (ML) |
|--------|------------------|-------------|
| False positive rate | 60-80% of alerts | < 50% of alerts |
| True positive rate | 15-25% of alerts | ≥ 20% of alerts |
| Threshold recalibration frequency | Quarterly (manual) | Daily (automated) |
| Threshold breach rate | Variable, often misaligned | Within ±20% of target percentile |
| Time to detect regime change | Days to weeks | Same day |

### 8.2 Qualitative Metrics

- Positive feedback from Ops/Risk/Compliance (≥80% satisfaction in surveys)
- Reduced escalations and ad-hoc threshold requests
- Successful Model Risk review and approval
- Positive Internal Audit assessment

### 8.3 Business Impact

- Reduced operational burden on manual threshold maintenance
- Improved risk detection during volatile markets
- Enhanced regulatory readiness and explainability
- Foundation for extending ML approach to other products

---

## 9. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data quality issues | High | Medium | Data validation checks, multiple data sources, fallback logic |
| Model underperformance | High | Medium | Extensive backtesting, parallel run, conservative fallback thresholds |
| User resistance to ML | Medium | Medium | Early stakeholder engagement, transparent explainability, gradual rollout |
| Regulatory concerns | High | Low | Engage compliance early, document methodology, model validation process |
| Infrastructure delays | Medium | Medium | Early infrastructure scoping, buffer time in plan |
| Key person dependency | Medium | Medium | Knowledge sharing, documentation, cross-training |

---

## 10. Governance and Controls

### 10.1 Model Ownership

| Role | Owner | Responsibilities |
|------|-------|------------------|
| Model Owner | [Name], Head of Market Risk Analytics | Accountability for model performance and compliance |
| Model Developer | [Name], Senior ML Engineer | Development, maintenance, enhancement |
| Model Validator | [Name], Model Risk Manager | Independent validation and ongoing review |
| Business Owner | [Name], Head of GFX Operations | Business requirements and UAT sign-off |

### 10.2 Change Management

**Minor Changes** (e.g., parameter tuning within validated ranges):
- Model Owner approval
- Documentation in change log
- Monthly summary to Model Risk

**Major Changes** (e.g., new algorithm, new data source):
- Full re-validation by Model Risk
- Approval by Model Oversight Committee
- Updated documentation

### 10.3 Review Cycle

- **Monthly:** Model performance review (Owner + Developer)
- **Quarterly:** Business review with stakeholders
- **Annual:** Full model validation by Model Risk
- **Ad-hoc:** After major market events or material performance degradation

---

## 11. Documentation and Training

### 11.1 Documentation Deliverables

1. **Model Design Document** (this document)
2. **Model Validation Report** (by Model Risk)
3. **User Guide** (for Ops/Risk/Compliance)
4. **API Documentation** (for developers/integrators)
5. **Runbooks** (for support and operations)
6. **Code Documentation** (docstrings, README)

### 11.2 Training Plan

**Target Audience: End Users (Ops/Risk/Compliance)**
- 2-hour workshop: ML concepts overview, how the system works
- Hands-on training: navigating dashboards, handling alerts
- Q&A session

**Target Audience: Model Owners / Risk**
- 4-hour deep dive: model methodology, assumptions, limitations
- Interpretation of model outputs
- How to interpret explainability (SHAP values, cluster descriptions)

**Target Audience: Support / IT**
- Technical training: architecture, troubleshooting
- Runbook walkthrough
- Escalation procedures

---

## 12. Appendices

### Appendix A: Glossary

- **GARCH:** Generalized Autoregressive Conditional Heteroskedasticity (volatility model)
- **GJR-GARCH:** Glosten-Jagannathan-Runkle GARCH (asymmetric volatility model)
- **HMM:** Hidden Markov Model (regime detection)
- **Isolation Forest:** Ensemble anomaly detection algorithm
- **SHAP:** SHapley Additive exPlanations (model explainability method)
- **Regime:** Market state characterized by distinct volatility/correlation behavior

### Appendix B: References

- LPA Consulting, "Machine Learning in Trade Surveillance" (2023)
- Bank for International Settlements, "FX execution algorithms and market functioning"
- Industry best practices from tier-1 investment banks (JPMorgan, Goldman Sachs, LSEG)
- Academic research on GARCH models, regime-switching, and anomaly detection in financial markets

### Appendix C: Approval Sign-offs

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Business Owner | | | |
| Model Owner | | | |
| Technology Lead | | | |
| Model Risk | | | |
| Compliance | | | |

---

**Document Control**

- **Version:** 1.0
- **Last Updated:** February 11, 2026
- **Next Review:** [Date]
- **Distribution:** Internal - Restricted
- **Classification:** Confidential

---

**End of Document**