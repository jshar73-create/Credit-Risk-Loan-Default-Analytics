import pandas as pd
import sqlite3
from pathlib import Path

# 1. File paths

data_path = Path("../data/cleaned_loans.csv")
database_path = Path("../data/credit_risk.db")
output_path = Path("../outputs/sql")

# Create SQL output folder if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# 2. Load cleaned data

df = pd.read_csv(data_path)

print("Cleaned data loaded successfully!")
print("Rows:", len(df))

# 3. Create SQLite database
connection = sqlite3.connect(database_path)

df.to_sql("loans",connection,if_exists="replace",index=False)

print("Loans table created successfully!")

# 4. SQL Business Analysis

# Portfolio performance by year

query_1 = """
SELECT
    issue_year,
    COUNT(*) AS loan_count,
    ROUND(SUM(loan_amount), 0) AS total_loan_amount,
    ROUND(AVG("default") * 100.0, 2) AS default_rate
FROM loans
GROUP BY issue_year
ORDER BY issue_year;
"""
result_1 = pd.read_sql_query(query_1, connection)

result_1.to_csv(output_path / "portfolio_by_year.csv",index=False)

print("\nSQL Analysis 1 - Portfolio Performance by Year")
print(result_1)

# Risk by loan purpose

query_2 = """
SELECT
    purpose,
    COUNT(*) AS loan_count,
    ROUND(SUM(loan_amount), 0) AS total_loan_amount,
    ROUND(AVG("default") * 100.0, 2) AS default_rate
FROM loans
GROUP BY purpose
HAVING COUNT(*) >= 10000
ORDER BY default_rate DESC;
"""

result_2 = pd.read_sql_query(query_2, connection)

result_2.to_csv(output_path / "purpose_risk.csv",index=False)

print("\nSQL Analysis 2 - Risk by Loan Purpose")
print(result_2)

# Geographic risk by state

query_3 = """
SELECT
    state,
    COUNT(*) AS loan_count,
    ROUND(SUM(loan_amount), 0) AS total_loan_amount,
    ROUND(AVG("default") * 100.0, 2) AS default_rate
FROM loans
GROUP BY state
HAVING COUNT(*) >= 1000
ORDER BY default_rate DESC;
"""

result_3 = pd.read_sql_query(query_3, connection)

result_3.to_csv(output_path / "state_risk.csv",index=False)

print("\nSQL Analysis 3 - Geographic Risk")
print(result_3)

# 7. Close database connection

connection.close()

print("\nSQL analysis completed successfully!")
print("Results saved in:", output_path)
