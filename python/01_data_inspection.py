import pandas as pd
#load data
file_path = "../data/LC_loans_granting_model_dataset.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

#Data Cleaning

print(df.isnull().sum())

print("Unique loan IDs:", df["id"].nunique())

print("Duplicate rows:", df.duplicated().sum())

print(df.describe())

# there are some numerical coloumn which i need to investigate first before moving ahead

import matplotlib.pyplot as plt
plt.boxplot(df['dti_n'])
plt.title('Distribution of Debt-to-Income ratio(DTI)')
plt.ylabel("DTI")
plt.show()


#revenue also have maximum value=11 million which also looks to investigate
plt.boxplot(df['revenue'])
plt.title('Distribution of Borrower Revenue')
plt.ylabel('Revenue')
plt.show()

top10_revenue=df.sort_values('revenue', ascending=False).head(10)
print(top10_revenue)

df.rename(columns={
    "issue_d": "issue_date",
    "dti_n": "dti",
    "loan_amnt": "loan_amount",
    "fico_n": "fico_score",
    "home_ownership_n": "home_ownership",
    "addr_state": "state",
    "Default": "default"}, inplace=True)

print(df.columns)

df["issue_date"] = pd.to_datetime(df["issue_date"],format="%b-%Y")
print(df["issue_date"].dtype) 

print(df["emp_length"].value_counts()) 
print(df["purpose"].value_counts()) 
print(df["home_ownership"].value_counts())
print(df["state"].value_counts())

# %% [markdown]
# # EDA — Exploratory Data Analysis

# %% [markdown]
# ## 1. Overall Default Rate

default_rate=df['default'].value_counts(normalize=True)*100
print(default_rate)        

# ## 2. Default rate on Purpose

default_bt_purpose=df.groupby('purpose')['default'].mean()*100
print(default_bt_purpose.sort_values(ascending=False))    

default_bt_purpose.plot(kind='barh', figsize=(12,6))
plt.title('Default rate by Loan Purpose')
plt.xlabel('Default rate(%)')
plt.ylabel('Loan Purpose')
plt.show()

## 3. Default rate on emp_length

default_by_emp_length=df.groupby('emp_length')['default'].mean()*100
print(default_by_emp_length.sort_values(ascending=False))
# I retained the missing employment-length category rather than imputing it because missingness itself may contain information about credit risk.
default_by_emp_length.plot(kind='barh',figsize=(10,6))
plt.title('Default rate bt Employment Length')
plt.ylabel('Employment Length')
plt.xlabel('Default rate(%)')
plt.show() 

## 4. Default rate by Home ownership
default_by_home=df.groupby('home_ownership')['default'].mean()*100
print(default_by_home.sort_values(ascending=False))
default_by_home.plot(kind='bar',figsize=(8,6))
plt.title('Default rate by Home ownership')
plt.xlabel('Home ownership')
plt.ylabel('Default rate (%)')
plt.xticks(rotation=0)
plt.show()

## 5. EDA on DTI values
bins=[0,10,20,30,40,50,100,200,500,999]
labels=['0-10','10-20','20-30','30-40','40-50','50-100','100-200','200-500','500-999']
df['dti_band']=pd.cut(df['dti'], bins,labels=labels,include_lowest=True)
print(df["dti_band"].value_counts().sort_index())

default_by_dti_band=df.groupby("dti_band")['default'].mean()*100
print(default_by_dti_band)

default_by_dti_band.plot(kind='bar',figsize=(12,6))
plt.title('Default rate by DTI band')
plt.xlabel('DTI band')
plt.ylabel('Default rate (%)')
plt.xticks(rotation=0)
plt.show()

## 6. Default risk on FICO score
#FICO stands for Fair Isaac Corporation. A FICO score is a numerical measure used by lenders to estimate how likely a person is to repay borrowed money.
#The commonly used FICO scoring range is: 300-850

fico_bins=[600,650,700,750,800,850]
fico_labels=['600-650','650-700','700-750','750-800','800-850']
df['fico_band']=pd.cut(df['fico_score'],fico_bins,labels=fico_labels, include_lowest=True)
print(df['fico_band'].value_counts().sort_index()) #600-650 have only 231 values but after that from 650-700 number is increasing

default_by_fico_band=df.groupby('fico_band')['default'].mean()*100
print(default_by_fico_band)

default_by_fico_band.plot(kind='bar',figsize=(10,6))
plt.title('Default rate by FICO score')
plt.xlabel('Fico score band')
plt.ylabel('Default rate (%)')
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# ## 7. Loan Amount and Default Risk
loan_bins = [500, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
loan_labels = ["500-5K","5K-10K","10K-15K","15K-20K", "20K-25K","25K-30K","30K-35K","35K-40K"]

df["loan_amount_band"] = pd.cut(df["loan_amount"],loan_bins,labels=loan_labels,include_lowest=True)

print(df["loan_amount_band"].value_counts().sort_index())
default_by_loan_amount = df.groupby("loan_amount_band")["default"].mean() * 100
print(default_by_loan_amount)

default_by_loan_amount.plot(kind="bar", figsize=(12, 6))
plt.title("Default Rate by Loan Amount")
plt.xlabel("Loan Amount Band")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# ## 8. Income and Default Risk

income_bins = [0, 30000, 50000, 75000, 100000, 150000, 200000, 11000000]
income_labels = ["0-30K","30K-50K","50K-75K","75K-100K","100K-150K","150K-200K","200K+"]

df["income_band"] = pd.cut(df["revenue"],income_bins,labels=income_labels,include_lowest=True)
print(df["income_band"].value_counts().sort_index())

default_by_income = df.groupby("income_band")["default"].mean() * 100
print(default_by_income)

default_by_income.plot(kind="bar",figsize=(10, 6))
plt.title("Default Rate by Borrower Income")
plt.xlabel("Income Band")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# ## Risk Segmentation 1 — FICO + DTI
#"I first analyzed the individual risk factors and then selected interactions based on their observed relationship with default risk and their business relevance to lending decisions, rather than creating every possible combination."

fico_dti_risk = pd.pivot_table(df,values="default",index="fico_band",columns="dti_band",aggfunc="mean") * 100
print(fico_dti_risk)

#lets count for each sample as well
# Borrower Count by FICO Score and DTI

fico_dti_count = pd.pivot_table(df,values="default",index="fico_band",columns="dti_band",aggfunc="count")
print(fico_dti_count)

# %% [markdown]
# ## Risk Segmentation 2 — FICO Score + Income

fico_income_risk = pd.pivot_table(df,values="default",index="fico_band",columns="income_band",aggfunc="mean") * 100
print(fico_income_risk)

fico_income_count = pd.pivot_table(df,values="default",index="fico_band",columns="income_band",aggfunc="count")
print(fico_income_count)

# %%[markdown]
# ## Data Cleaning and Feature Engineering

print(df["experience_c"].value_counts())
df.drop(columns=["experience_c","title", "desc","zip_code"], inplace=True)   #remove these because these are not useful for my credit risk analysis
print(df.columns)

# Check unique values in categorical column
print(df["emp_length"].unique())
print(df["purpose"].unique())
print(df["home_ownership"].unique())
print(df["state"].nunique())

## Feature Engineering

df['loan_to_income']=df['loan_amount']/df['revenue']
print(df[['loan_amount','revenue','loan_to_income']].head())

df['issue_year']=df['issue_date'].dt.year
print(df[["issue_date", "issue_year"]].head()) 

#save cleaned data
df.to_csv("../data/cleaned_loans.csv", index=False)
print("Cleaned dataset saved successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nFinal cleaned dataset columns:")
print(df.columns.tolist())

print("\nFinal dataset shape:")
print(df.shape)
