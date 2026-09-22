# Credit Risk & Loan Default Analytics

## Project Overview

This project analyzes historical LendingClub loan data to understand loan default risk and borrower characteristics.

The project combines Python, SQL, Machine Learning, and Power BI to perform:

- Data cleaning and exploratory data analysis
- Business-focused SQL analysis
- Borrower risk segmentation
- Loan default prediction
- Interactive portfolio risk visualization

The goal is to demonstrate how a data analyst can transform raw lending data into meaningful business insights and support credit risk decision-making.

## Business Problem

Lending institutions need to understand which borrower and loan characteristics are associated with higher default risk.

This project addresses the following business questions:

- What is the overall loan default rate?
- How has portfolio performance changed over time?
- Which loan purposes have higher observed default rates?
- How do credit score and debt-to-income ratio relate to default risk?
- How does default risk vary geographically?
- Can borrower and loan characteristics be used to estimate the probability of default?
- How can these insights be presented in an interactive dashboard for business users?

## Dataset

The project uses the **Lending Club Loan Dataset for Granting Models**, based on historical LendingClub loan data.

- **Source:** Zenodo
- **Dataset:** Lending Club loan dataset for granting models
- **Publication date:** May 25, 2024
- **Loan period:** 2007–2018
- **Records:** 1.34 million+
- **Original variables:** 15

The dataset contains borrower, loan, employment, credit score, debt-to-income, loan purpose, geographic, and loan outcome information.

### Target Variable

The target variable is `default`:

- `0` — Fully paid
- `1` — Default / Charged off

The project focuses on application-time variables to avoid using information that would only become available after the loan decision.

The original dataset is not included in this GitHub repository because of its large file size. The dataset can be downloaded from the original source and processed using the project's Python scripts.

**Dataset Source:**  
https://zenodo.org/records/11295916

## Tools & Technologies

| Tool/Technology | Purpose |

| Python          | Data cleaning, exploratory data analysis and feature engineering |
| Pandas          | Data manipulation and analysis |
| Matplotlib      | Data visualization during exploratory analysis |
| SQL             | Database-level business analysis using SQLite |
| SQLite          | Storage and querying of the cleaned lending dataset |
| Scikit-learn    | Machine learning and model evaluation |
| Power BI        | Interactive dashboard and portfolio risk visualization |
| GitHub          | Project documentation and version control |

## Project Workflow

The project follows an end-to-end analytics workflow:

1. **Data Collection**
   - Obtained historical LendingClub loan data from the original dataset source.

2. **Data Cleaning & Preparation**
   - Inspected data types, missing values and duplicates.
   - Converted the loan issue date into a proper date format.
   - Removed an almost constant feature with very limited analytical value.
   - Created additional analytical features such as loan-to-income ratio and issue year.

3. **Exploratory Data Analysis**
   - Analyzed default distribution and borrower characteristics.
   - Examined relationships between default risk and factors such as FICO score, DTI, income, loan amount and loan purpose.

4. **SQL Business Analysis**
   - Loaded the cleaned dataset into SQLite.
   - Analyzed portfolio performance by year.
   - Compared default rates across loan purposes.
   - Analyzed geographic differences in observed default rates.

5. **Machine Learning**
   - Created a time-based train/test split.
   - Built a Logistic Regression baseline model for default prediction.
   - Evaluated the model using ROC-AUC, PR-AUC, precision, recall and confusion matrix.
   - Compared the baseline model with a Random Forest benchmark.
   - Examined different probability thresholds to understand the precision-recall trade-off.

6. **Power BI Dashboard**
   - Built an interactive dashboard to monitor portfolio volume, default rates and borrower risk patterns.
   - Added filters for year, loan purpose and state.

7. **Business Insights**
   - Converted analytical findings into business-oriented insights related to portfolio growth, borrower risk and observed default patterns.

   ## Python Analysis

Python was used for data preparation, exploratory analysis and feature engineering.

### Data Preparation

The dataset was inspected for:

- Data types
- Missing values
- Duplicate records
- Target variable distribution
- Numerical variable distributions
- Potential outliers

Key preparation steps included:

- Renaming columns for clearer analysis
- Converting `issue_d` into a proper date field
- Removing the nearly constant `experience_c` feature
- Retaining categorical variables such as employment length, loan purpose, home ownership and state
- Creating `loan_to_income` as an additional risk-related feature
- Creating `issue_year` for time-based analysis

### Exploratory Analysis

The analysis examined default patterns across:

- Loan purpose
- Employment length
- Home ownership
- FICO score bands
- DTI bands
- Income levels
- Loan amount

The analysis also examined borrower risk segmentation by combining key financial characteristics, including:

- FICO score and DTI
- FICO score and income

These segmentations helped identify borrower groups with different observed default rates and provided additional context for understanding credit risk beyond individual variables.


## SQL Business Analysis

The cleaned dataset was loaded into a SQLite database to perform business-focused portfolio analysis.

The SQL analysis focused on three key business questions:

| Business Question | Analysis |

| How did portfolio performance change over time? | Portfolio performance by year |
| Which loan purposes had higher observed default rates? | Risk by loan purpose |
| How did observed default rates vary across states? | Geographic risk |

### 1. Portfolio Performance by Year

The analysis calculated:

- Number of loans
- Total loan amount
- Observed default rate

This helps understand how the lending portfolio changed over time and how observed default rates varied across different loan years.

**Output:** `outputs/sql/portfolio_by_year.csv`

### 2. Risk by Loan Purpose

Loan purposes were compared using:

- Number of loans
- Total loan amount
- Observed default rate

A minimum volume threshold was applied so that very small loan-purpose groups did not dominate the comparison.

**Output:** `outputs/sql/purpose_risk.csv`

### 3. Geographic Risk

State-level analysis was performed using:

- Number of loans
- Total loan amount
- Observed default rate

A minimum loan-count threshold was applied to make comparisons more meaningful.

**Output:** `outputs/sql/state_risk.csv`

These SQL analyses complement the Python analysis by focusing specifically on portfolio-level business questions from a database perspective.

## Machine Learning

A baseline classification model was developed to estimate the probability of loan default using information available at the time of the loan application.

### Objective

The objective was to determine whether borrower and loan characteristics could be used to estimate default risk and demonstrate the trade-off between identifying more potential defaults and maintaining prediction precision.

### Features Used

The model used:

- Revenue
- Debt-to-income ratio (DTI)
- Loan amount
- FICO score
- Loan-to-income ratio
- Issue year
- Employment length
- Loan purpose
- Home ownership
- State

### Modeling Approach

A time-based train/test split was used to simulate a historical-to-future prediction scenario:

- Training data: 2007–2016
- Test data: 2017–2018

A Logistic Regression model was developed as the primary baseline model.

Categorical variables were one-hot encoded, while numerical variables were standardized using a preprocessing pipeline.

A Random Forest model was also evaluated as a benchmark.

### Model Evaluation

Because the dataset contains a substantial difference between default and non-default cases, accuracy alone was not considered sufficient.

The models were evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- Confusion Matrix

Probability threshold analysis was also performed to understand how changing the classification threshold affects precision and recall.

### Model Result

The baseline Logistic Regression model achieved:

- Accuracy: **78.57%**
- Precision: **45.99%**
- Recall: **3.86%**
- ROC-AUC: **0.6767**
- PR-AUC: **0.3436**

Time-series cross-validation produced a mean ROC-AUC of approximately **0.659**.

GridSearchCV selected **C = 0.1** for Logistic Regression, with a best cross-validation ROC-AUC of approximately **0.659**.

At the provisional probability threshold of **0.30**, Logistic Regression achieved:

- Precision: **37.42%**
- Recall: **36.65%**

A Random Forest model was also evaluated as a benchmark. At the same 0.30 threshold, Random Forest achieved:

- Precision: **44.51%**
- Recall: **14.30%**

For this dataset and the specific credit-risk objective, Logistic Regression was selected as the primary model because it provided substantially higher recall at the 0.30 threshold while remaining easier to interpret and computationally efficient.

The model should be considered a **baseline analytical model**, not a production-ready credit decision system.

## Power BI Dashboard

An interactive Power BI dashboard was developed to provide a business-friendly view of portfolio performance and observed credit risk.

### Dashboard Components

The dashboard includes:

- Total Loans
- Total Loan Amount
- Defaulted Loans
- Overall Default Rate
- Portfolio growth and observed default rate by year
- Observed default rate by loan purpose
- Observed default rate by FICO score band
- Observed default rate by DTI band
- Geographic default risk by state

### Interactive Filters

Users can filter the dashboard by:

- Issue Year
- Loan Purpose
- State

These filters allow users to explore how portfolio volume and observed default rates change across different borrower and loan segments.

### Dashboard Objective

The dashboard is designed to help business users quickly monitor:

- Portfolio size and growth
- Default trends over time
- Borrower risk characteristics
- Purpose-level risk patterns
- Geographic differences in observed default rates

The dashboard complements the Python and SQL analysis by presenting the key findings in an interactive and business-friendly format.

## Key Business Insights

The analysis produced several important observations from the historical lending portfolio:

- The overall observed default rate in the dataset was approximately **19.98%**.
- The lending portfolio expanded substantially over the analyzed period, with loan volume reaching its highest level around **2015**.
- Observed default rates varied considerably across loan purposes, with `small_business` showing a relatively high observed default rate among sufficiently large loan-purpose groups.
- Larger loan-purpose segments, such as `debt_consolidation`, represented substantial portfolio volume and loan exposure.
- Lower FICO score bands generally showed higher observed default rates than higher FICO score bands.
- Higher DTI bands generally showed higher observed default rates across the main borrower segments.
- Income and credit characteristics showed different observed risk patterns across borrower segments.
- State-level observed default rates varied across the portfolio, although smaller states should be interpreted carefully because of their lower loan volumes.
- Combining borrower characteristics such as FICO score and DTI provided additional segmentation of observed default risk.
- The Logistic Regression model demonstrated that application-time borrower and loan characteristics contain predictive information about future loan default, while also showing the precision-recall trade-off involved in selecting a probability threshold.

## Project Structure

```text
Credit Risk & Loan Default Analytics/
│
├── data/
│   ├── cleaned_loans.csv
│   └── credit_risk.db
│
├── python/
│   ├── 01_data_inspection.py
│   ├── 02_load_to_sqlite.py
│   └── 03_credit_risk_model.py    
│
├── outputs/
│   └── sql/
│       ├── portfolio_by_year.csv
│       ├── purpose_risk.csv
│       └── state_risk.csv
│
├── README.md
└── Power BI Dashboard.pbix