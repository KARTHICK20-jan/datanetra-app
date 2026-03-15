# DataNetra.ai
## AI-Powered Retail Data Intelligence Platform for MSMEs

> **India AI Innovation Challenge — Problem Statement 2**
> Digital Onboarding & ONDC Marketplace Readiness for Indian MSMEs

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Landing Page](#2-landing-page)
3. [Platform Access](#3-platform-access)
4. [MSME Registration Workflow](#4-msme-registration-workflow)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [ONDC Marketplace Analysis](#6-ondc-marketplace-analysis)
7. [Business Intelligence Dashboard](#7-business-intelligence-dashboard)
8. [Business Intelligence Report](#8-business-intelligence-report)
9. [Technology Stack](#9-technology-stack)
10. [Deployment](#10-deployment)
11. [Project Vision](#11-project-vision)

---

## 1. Platform Overview

DataNetra.ai is an AI-powered MSME data intelligence platform that converts raw Excel or POS sales datasets into structured business insights, forecasting intelligence, and ONDC marketplace readiness analytics.

**Input:** Excel (.xlsx), CSV, POS exports, or billing software reports — any column naming convention accepted.

**Output:**
- Cleaned, standardized dataset
- Business health and performance scores
- 6-month and 12-month revenue forecasts
- ONDC marketplace readiness score
- Category, store, and product-level drill-down analytics
- Downloadable Business Intelligence PDF report

### DataNetra vs. Traditional Dashboards

| Traditional Dashboard | DataNetra |
|---|---|
| Requires clean, structured data input | Accepts messy, real-world Excel and CSV files |
| User must know which charts to read | AI generates plain-language business insights |
| No forecasting | Prophet + Holt-Winters + Linear Regression + Baseline forecasting |
| No ONDC readiness guidance | Explicit ONDC readiness score with actionable gap analysis |
| Static reports | Dynamic drill-down by store, category, and product |
| No data quality assessment | Multi-dimension data readiness check before analysis |

---

## 2. Landing Page

### Hero Section

The landing page supports full language switching between English and Hindi. All interface labels, headings, field names, buttons, and card content switch to natural, easy-to-read Hindi when the language toggle is activated. This applies across all steps of the platform.

### Platform Capabilities

**What DataNetra Delivers:**
- Business Performance Clarity — sales, margins, returns, and category performance
- Demand Forecasting Intelligence — AI-driven demand prediction
- Business Health Score — automated scoring from sales, returns, and margin data
- Marketplace Readiness Insights — identify ONDC-ready products and categories

**Platform Capabilities:**
- Intelligent Column Detection — auto-detects retail fields from any file layout
- Automatic Cleaning and Normalization — fixes dates, duplicates, and currency formats
- AI Readiness Scoring — instant Data Readiness Score calculation
- Retail Intelligence Engine — converts clean data into forecasting insights and recommendations

### How DataNetra Processes Your Data

A visual pipeline on the landing page shows the five automated stages every dataset passes through:

1. **Upload Dataset** — Excel (.xlsx) or CSV sales data
2. **Validate Data** — structure and format checks
3. **Clean and Standardize** — removes duplicates, standardizes formats
4. **Readiness Score** — dataset quality evaluation
5. **Forecasting and Insights** — forecasting, scoring, and actionable insights

### Data Readiness Engine

Before committing to full analysis, users can run a **Data Readiness Check** directly on the landing page. This engine evaluates dataset quality across multiple dimensions and produces a structured report.

The Data Readiness Check is organized into six tabs:

| Tab | Description |
|---|---|
| 📊 Overview | High-level summary of dataset health, record count, column coverage, and readiness score |
| 🔎 Data Quality | Detailed quality assessment — missing values, outliers, duplicates, date consistency |
| 📐 Structure | Column mapping results, detected field types, and structural warnings |
| 🛡️ Authenticity | Data authenticity checks — flags suspicious patterns or synthetic-looking records |
| 🧹 Cleaning | Automatic cleaning summary showing what was fixed and how |
| 🗂️ Field Mapping | Maps detected columns to standard retail fields (date, product, sales, category, etc.) |

Once the dataset passes the readiness checks, the cleaned dataset becomes the input for the forecasting models and the business intelligence dashboards used in later steps.

After the readiness check, users can download their **clean, standardized dataset** for use in the full analysis pipeline.

### Dataset Template

Users who do not have structured sales data can download a **guided dataset template** from the landing page. The template provides the recommended column structure with example data.

### Supported Data Formats

- Excel (.xlsx)
- CSV
- POS system exports
- Billing software reports

---

## 3. Platform Access

DataNetra provides two distinct access modes from the landing page.

### Government Login

Government Login provides state-level MSME intelligence dashboards for ministry officials and district-level government officers.

| Field | Value |
|---|---|
| Username | `Admin` |
| Password | `Admin` |

Officials can upload aggregated MSME datasets to monitor the performance of multiple businesses across a district or state. The government dashboard provides visibility into:

- Category sales trends across the MSME portfolio
- Margin patterns and financial risk indicators
- Fulfillment performance and supply-chain reliability
- Return behaviour by category and product type
- Marketplace readiness indicators for ONDC onboarding

This dashboard supports district or state-level monitoring of MSME ecosystem performance, scheme evaluation, and policy decision-making.

### MSME Login — Individual Business Intelligence

MSME Login enables individual business intelligence analysis for registered business owners.

MSME users can analyze their business across multiple dimensions:
- Store-level performance — revenue, margins, returns per location
- Category-level performance — best and worst performing product categories
- Product-level performance — SKU-level margin, return rate, and fulfillment
- Demand forecasting — 6-month and 12-month revenue projections
- ONDC readiness insights — marketplace fit score and readiness assessment

**Two entry points:**

**Returning users** can enter their registered mobile number directly on the landing page to proceed directly to Step 5 (data upload and analysis).

**New users** click **Login as MSME** to begin the Step 1 registration workflow, which captures their identity, verifies their Udyam number, and builds their business profile before enabling data analysis.

---

## 4. MSME Registration Workflow

The registration workflow is a guided, step-by-step process. Each step validates before proceeding to the next. Voice input is available in Steps 1 and 2 for hands-free data entry (Chrome and Edge browsers only).

---

### Step 1 — User Information

The user provides their personal and professional details.

**Fields collected:**
- Full Name
- Mobile Number (must be 10 digits, starting with 6, 7, 8, or 9)
- Email Address
- Role (Business Owner, Co-Founder, Category Manager, Analyst, Store Manager)

Both manual entry and voice entry tabs are available. In voice mode, the user speaks a sentence such as *"My name is [Your Name], mobile [Your Number], role [Your Role]"* and the system fills the fields automatically. Multilingual voice UI is set up and will be fully functional in Stage 2.

---

### Step 2 — MSME Verification

The user's Udyam registration is verified through OTP authentication.

**Process:**
1. User enters their Udyam Number (format: UDYAM-XX-XX-XXXXXXX)
2. An OTP is sent to the registered mobile number
3. After OTP entry, the system fetches the MSME's registered information automatically
4. Fetched data includes: Enterprise Name, Organisation Type, Major Activity, Enterprise Type, State, City, and Industry Domain

Voice entry is available for this step. The user can speak their Udyam number and OTP for hands-free verification.

---

### Step 3 — MSME Certificate Review

The fetched MSME details are displayed for the user's review and confirmation.

**Process:**
- All enterprise details fetched in Step 2 are shown as read-only fields
- The user confirms accuracy with a checkbox: *"I confirm the above MSME details are correct"*
- The user provides consent for certificate verification: *"I consent to verify the MSME certificate"*
- If the certificate requires manual upload (missing or unverifiable), the user can upload the MSME certificate PDF for system validation

The step will not proceed unless both consent checkboxes are confirmed.

---

### Step 4 — Business Profile

The user provides additional business context to enable accurate scoring and benchmarking.

**Fields collected:**
- Business Type (Hypermarket, Supermarket, Retail Store, etc.)
- Years in Operation
- Monthly Revenue Range

This data is used in scoring formulas and to contextualize the AI-generated insights against appropriate benchmarks.

---

### Step 5 — Upload Business Dataset and Analysis

After completing registration, the user uploads their sales dataset and triggers the full AI analysis pipeline.

**Upload process:**
- Accepted formats: Excel (.xlsx) or CSV
- The user confirms data analysis consent
- Clicking **Analyze Data** initiates the full processing pipeline
- A loading indicator is displayed during processing
- Results are rendered automatically when complete

**Analysis models used:**

| Model | Purpose |
|---|---|
| **Prophet** (Facebook) | Primary time-series forecasting — 6-month and 12-month revenue projections with confidence intervals |
| **Holt-Winters ExponentialSmoothing** | Seasonal decomposition forecasting — captures trend and seasonality patterns |
| **Linear Regression** | Baseline growth trend modeling and fallback forecasting when data volume is limited |
| **Baseline (NumPy)** | Simple historical average — always computed as a reference benchmark alongside ML models |
| **K-Means Clustering** | Used internally to group products and categories based on sales patterns. This segmentation helps generate category performance insights and supports marketplace readiness evaluation. |

Prophet is used as the primary forecasting engine when sufficient data is available (minimum 4 monthly data points). The system falls back to Holt-Winters and then Linear Regression automatically based on data availability.

---

## 5. Data Processing Pipeline

Every dataset uploaded to DataNetra passes through a fully automated processing pipeline before analysis is performed.

### Stage 1 — Upload

The raw Excel or CSV file is ingested. The system accepts files with any column naming convention — it does not require a fixed schema.

### Stage 2 — Validation

The validation stage checks:
- Minimum required columns are present or mappable
- Date columns are parseable
- Numeric sales columns contain valid values
- Record count is sufficient for statistical analysis

### Stage 3 — Column Normalization and Mapping

`normalize_headers()` standardizes column names by stripping whitespace, lowercasing, and removing special characters. `map_columns()` then maps detected column names to the standard internal schema using an alias dictionary covering over 60 common naming variations across Indian retail and POS systems (e.g., `invoice_date`, `txn_date`, `orderdate`, `saledate` are all mapped to `date`).

### Stage 4 — Cleaning and Standardization

The cleaning stage performs:
- Duplicate row removal
- Date format standardization
- Currency string parsing (removes ₹, commas, lakhs notation)
- Negative sales value correction
- Null value imputation using category-level medians
- Outlier detection and flagging

### Stage 5 — Feature Extraction

From the cleaned data, the pipeline extracts:
- Monthly aggregates by store, category, and product
- Rolling averages for trend detection
- Fulfillment rate calculation (units sold vs. units ordered)
- Return rate per category and SKU
- Gross and net revenue separation

### Stage 6 — Scoring

`calculate_scores()` computes five composite scores:

| Score | Description |
|---|---|
| **Health Score** | Overall business health — blends margin, return rate, fulfillment, and forecast growth |
| **Performance Score** | Operational performance relative to category benchmarks |
| **Financial Risk Score** | Risk index based on margin compression, return rates, and revenue volatility |
| **Vendor / Supply-Chain Score** | Fulfillment consistency and inventory turnover efficiency |
| **ONDC Readiness Score** | Composite score (0–100%) weighted: 35% growth trajectory, 25% vendor reliability, 20% profit margin, 20% return rate management |

### Stage 7 — Forecasting

`forecast_sales()` runs the multi-model forecasting pipeline and returns:
- 6-month revenue forecast with lower and upper confidence bounds
- 12-month revenue forecast
- Peak demand month identification
- Month-over-month growth percentage

### Stage 8 — Insight Generation

`generate_insights()` converts all computed scores, forecasts, and segment data into structured AI business insights including demand outlook, inventory health, supplier cost opportunity, and return risk assessment.

---

## 6. ONDC Marketplace Analysis

Step 6 presents the **ONDC Impact Dashboard** — a full analysis of the MSME's readiness for onboarding to the Open Network for Digital Commerce.

### ONDC Readiness Score

The ONDC Readiness Score is a composite metric (0–100%) computed from four weighted components:

- **35%** — Revenue growth trajectory (forecast-driven)
- **25%** — Vendor and supply-chain reliability
- **20%** — Profit margin relative to 20% target
- **20%** — Return rate management relative to 7% threshold

### KPI Cards

The Step 6 dashboard displays six live KPI cards:

| KPI | Description |
|---|---|
| Units Sold | Total units across selected filter period |
| Net Sales | Revenue after returns |
| Margin % | Average profit margin |
| Return Rate | Percentage of sold units returned |
| Replacements | Units replaced due to returns |
| Fulfillment | Order fulfillment rate |

### Visualization Charts

Step 6 generates seven analytical charts:

1. **Monthly Sales Trend** — revenue over time with period-on-period comparison
2. **Profit Margin Trend** — margin percentage over time with 20% target reference line
3. **6-Month Revenue Forecast** — Prophet, Holt-Winters, Linear Regression, and baseline projections overlaid
4. **12-Month Revenue Forecast** — extended forecast with confidence interval bands
5. **Fulfillment Rate Analysis** — fulfillment consistency with 85% target reference
6. **Sales vs. Returns** — monthly units sold against returns volume
7. **Inventory Turnover** — turnover rate with good (≥12x) and warning (<6x) threshold bands

### Drill-Down Filters

All Step 6 charts respond to three filter dimensions:
- **Store** — individual store or all stores combined
- **Category** — product category or all categories
- **Product** — specific SKU or all products

### Government Dashboard (Step 6a)

Government officials access a separate portfolio dashboard through Step 6a. After uploading an aggregated MSME dataset, the system generates a state-level intelligence view covering multiple businesses, enabling policy-level monitoring and scheme impact assessment.

---

## 7. Business Intelligence Dashboard

Step 7 is the **Forecast Intelligence Dashboard** — a granular drill-down interface for deep business analysis beyond the ONDC summary view.

### Dashboard Filters

The Step 7 dashboard provides three independent filter controls:
- 🏪 **Store** filter
- 📂 **Category** filter
- 📦 **Product** filter

All seven charts and the KPI snapshot update dynamically when filters are changed.

### KPI Snapshot

A high-level KPI row displays:
- Total Revenue (₹)
- Average Margin %
- Average Return Rate %
- Fulfillment Rate %
- 6-Month Forecast (₹)

### Category Performance Table

A sortable table shows each category's revenue contribution, margin, return rate, and fulfillment rate — enabling the business owner to identify which categories to prioritize or review.

### Seven Dashboard Charts

| Chart | What It Shows |
|---|---|
| Category Sales Comparison | Revenue by category in the selected period |
| Category Margin Comparison | Margin percentage by category with 20% target line |
| 6-Month Forecast | Near-term revenue projection with confidence range |
| 12-Month Forecast | Full-year revenue projection |
| Fulfillment Rate Trend | Order fulfillment over time with 85% target band |
| Sales vs. Returns | Volume of sales against returns by period |
| Inventory Turnover | Turnover rate against good/warning thresholds |

### AI Business Insights Panel

The Step 7 dashboard includes an **AI Business Insights** panel that generates five contextual insight statements based on live data:

1. **Demand Outlook** — stable, rising, or declining demand forecast for the top category
2. **Inventory Levels** — healthy stock or restocking action required with specific percentage recommendation
3. **Supplier Cost Improvement Opportunity** — margin gap analysis with estimated profit recovery from a 5% cost reduction
4. **High Product Returns / Return Rate Healthy** — return risk assessment with SKU-level guidance
5. **Recommended Restocking Action** — specific inventory increase recommendation when fulfillment falls below 85%

Each insight includes the actual metric value and a specific recommended action — not generic advice.

---

## 8. Business Intelligence Report

Clicking **Download Business Intelligence Report (PDF)** generates a comprehensive, multi-page PDF report using ReportLab.

### Report Structure

The PDF report is organized into six sections:

| Section | Content |
|---|---|
| **Cover Page** | Company name, owner, Udyam ID, report date, total revenue, record count |
| **Section 1 — Business Performance** | Five KPI cards: Total Revenue, Data Records, Avg Profit Margin, MSME Health Score, Performance Score |
| **Section 2 — Score Breakdown** | Financial Risk Score, Health Score, Performance Score, Vendor Score — each with progress bar, badge, formula explanation, and interpretation |
| **Section 3 — Business Opportunity Insights** | Profit margin analysis, inventory assessment, ONDC marketplace readiness with comparative before/after revenue projections |
| **Section 4 — Sales Forecast** | 6-month and 12-month ML-powered revenue projections with model name, growth percentage, peak month identification, and confidence ranges |
| **Section 5 — SNP Mapping Intelligence** | ONDC Seller Network Participant category mapping and readiness analysis |
| **Section 6 — Action Plan** | Four structured AI insight panels (Demand Forecasting, Inventory Optimisation, Supplier Cost Opportunity, Return Risk Management) followed by a prioritized Action Plan table and Business Intelligence Summary |

The PDF is generated server-side and served as a downloadable file. The filename follows the format:
```
Business_Intelligence_Report_UDYAM-XX-XX-XXXXXXX_YYYYMMDD.pdf
```

---

## 9. Technology Stack

### Core Framework

| Layer | Technology | Version |
|---|---|---|
| UI Framework | Gradio | 4.44.1 |
| Language | Python | 3.10 / 3.11 |

### Data and Machine Learning

| Library | Purpose |
|---|---|
| Pandas | Data ingestion, cleaning, aggregation, and transformation |
| NumPy | Numerical operations, array processing, fallback forecasting |
| scikit-learn | K-Means clustering, Linear Regression, StandardScaler |
| Prophet (Facebook) | Primary time-series forecasting |
| statsmodels | Holt-Winters ExponentialSmoothing for seasonal forecasting |

### Visualisation

| Library | Purpose |
|---|---|
| Matplotlib | All chart generation (headless Agg backend for server deployment) |

### Report Generation

| Library | Purpose |
|---|---|
| ReportLab | PDF report generation — layout, tables, charts, and text |
| pdfplumber | PDF certificate parsing for MSME certificate validation |

### File Handling

| Library | Purpose |
|---|---|
| openpyxl | Excel (.xlsx) file reading and template export |

### Database

| Library | Purpose |
|---|---|
| SQLAlchemy | User profile and session persistence (SQLite backend, falls back to in-memory if unavailable) |

### Deployment Dependencies

| Package | Purpose |
|---|---|
| cmdstanpy | Required by Prophet for CmdStan compilation on Linux |
| lark 1.1.9 | Pinned to avoid jsonschema rfc3987 conflict on Python 3.12 |
| jsonschema 4.17.3 | Pinned for Gradio compatibility |

---

## 10. Deployment

### Local Setup

```bash
git clone https://github.com/<your-org>/datanetra.git
cd datanetra
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

### Render Deployment

DataNetra is configured for one-click deployment on Render.

**Files required:**
```
app.py
requirements.txt
Procfile
render.yaml
```

**Render settings:**

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python app.py` |
| Python Version | 3.11 |
| Instance Type | Standard (2 GB RAM minimum — Prophet requires it) |

The application binds to `0.0.0.0` and reads the `$PORT` environment variable injected by Render automatically.

**Note:** First build takes 5–8 minutes due to Prophet's C++ CmdStan compilation. Subsequent deploys are approximately 2 minutes.

---

## 11. Project Vision

### Current State

DataNetra is a functional platform covering the full journey from raw data upload to ONDC readiness assessment and PDF report generation. An MSME with no data team can derive actionable business intelligence from their existing sales records in under five minutes.

### Roadmap

**Phase 1 — Individual MSME Intelligence (Current)**
Any MSME uploads their sales data and receives a Business Intelligence Report, ONDC readiness score, and 12-month revenue forecast.

**Phase 2 — District and State Analytics**
Aggregated, anonymized MSME data enables district-level dashboards for government officials — tracking scheme uptake and sectoral trends across geographies.

**Phase 3 — ONDC Integration**
Direct integration with the ONDC network — enabling DataNetra-verified MSMEs to onboard to digital marketplaces with their readiness score as a trust signal.

**Phase 4 — Predictive Policy Intelligence**
DataNetra evolves into a real-time economic intelligence layer — surfacing demand signals, supply disruptions, and market opportunities to inform policy decisions.

---

## License

DataNetra.ai — AI Innovation Challenge Submission
Problem Statement 2: Digital Onboarding and ONDC Marketplace Readiness for Indian MSMEs
