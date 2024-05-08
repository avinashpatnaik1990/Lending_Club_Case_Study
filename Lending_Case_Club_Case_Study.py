#!/usr/bin/env python
# coding: utf-8

# # Import Python Libraries:

# In[104]:


##Importing Necessary Libraries
# Numerical and Data analysis
import numpy as np
import pandas as pd

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Extra
import warnings
warnings.filterwarnings('ignore')


# # Setting Display Options For Pandas DataFrame in Python:

# In[105]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


# # Reading Data Set:

# In[80]:


file_path="C:/Users/avina/OneDrive/Desktop/Upgrad/Lending_Club_Case_Study/loan/loan.csv"
df =pd.read_csv(file_path, low_memory=False)
df.head(2)


# # Plotting functions

# In[106]:


def visualize_loan_status_by_variable(df, x_column):
    y_column='loan_status'
    title="Loan Status by " + str(y_column)
    cross_tab = pd.crosstab(df[x_column], df[y_column], normalize='index') * 100
    
    # Plotting
    plt.figure(figsize=(12, 8))
    ax = cross_tab.plot(kind='bar', stacked=True)
    plt.xlabel(x_column)
    plt.ylabel('Percentage')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Loan Status ')

    # Annotate percentages on the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(x_column)
    plt.show()


def visualize_loan_status_variable(df, x_column):
    """
    Visualize loan status distribution by employee length.

    Parameters:
    - df: DataFrame containing the required columns.
    - x_column: Name of the column representing employee length.
    - y_column: Name of the column representing loan status.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.

    Returns:
    - None
    """
    title='Loan Status by ' + x_column
    y_column='loan_status'
    cross_tab = pd.crosstab(df[x_column], df[y_column])
    plt.figure(figsize=(12, 8))
    cross_tab.plot(kind='bar', stacked=True)

    plt.xlabel(x_column)
    plt.ylabel('count')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Loan Status ')
    plt.tight_layout()
    plt.savefig(x_column)
    plt.show()


# # Inspect Data Frames

# In[82]:


# Checking the variables of the dataframes
df.describe()


# In[83]:


# To Determine the Shape of the Dataset
df.shape


# In[84]:


# To obtain the Summary of the date Frame including Data Type
df.info('all')


# # Data Cleaning & Manipulation

# In[107]:


null_DF = pd.DataFrame(100*df.isnull().mean()).reset_index()
null_DF.columns = ['Column Name', 'Null Values %']
fig = plt.figure(figsize=(20,10))
ax = sns.pointplot(x="Column Name",y="Null Values %",data=null_DF,color='green')
plt.xticks(rotation =90,fontsize =10)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in given data")
plt.ylabel("Null Values %")
plt.xlabel("COLUMNS")
plt.show()


# In[108]:


# Cleaning the Missing data

# Listing the Null value Columns having More than 30%

emptycol=100*df.isnull().mean()
emptycol=emptycol[emptycol.values>30.00]
len(emptycol)


# In[109]:


# Removing the above 58 Null Value Columns having More than 30%

emptycol = list(emptycol[emptycol.values>=30.00].index)
df.drop(labels=emptycol,axis=1,inplace=True)


# In[88]:


# To Determine the Shape of the Present Dataset after removing Null Value Columns

df.shape


# In[33]:


# Listing the Null value Rows having More than 30%

emptyrow=df.isnull().mean()
emptyrow=list(emptyrow[emptyrow.values>=30.00].index)
print(len(emptyrow))


# In[34]:


# Removing the  Null Value Rows having More than 30%

emptyrow=df.isnull().mean()
emptyrow=list(emptyrow[emptyrow.values>=30.00].index)
df.drop(labels=emptyrow,axis=0,inplace=True)


# In[89]:


# To Determine the Shape of the Present Dataset after removing Null Value Columns and Rows

df.shape


# # Null check again

# In[90]:


null_DF = pd.DataFrame(100*df.isnull().mean()).reset_index()
null_DF.columns = ['Column Name', 'Null Values %']
fig = plt.figure(figsize=(20,10))
ax = sns.pointplot(x="Column Name",y="Null Values %",data=null_DF,color='green')
plt.xticks(rotation =90,fontsize =10)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in given data")
plt.ylabel("Null Values %")
plt.xlabel("COLUMNS")
plt.show()


# ## Dropping columns that has only single values and columns that wont be usefull for analyzing

# In[91]:


df.drop(['grade', 'sub_grade','id','pymnt_plan', 'member_id','collections_12_mths_ex_med','emp_title'
         ,'policy_code','application_type','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt',
        'tax_liens','url','funded_amnt_inv'],axis=1,inplace=True)


# ## Tried and tested date columns and other columns, Since Evident inference could not be obtained from these columns, dropping these columns.

# # Null check again after dropping columns

# In[92]:


null_DF = pd.DataFrame(100*df.isnull().mean()).reset_index()
null_DF.columns = ['Column Name', 'Null Values %']
fig = plt.figure(figsize=(20,10))
ax = sns.pointplot(x="Column Name",y="Null Values %",data=null_DF,color='green')
plt.xticks(rotation =90,fontsize =10)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in given data")
plt.ylabel("Null Values %")
plt.xlabel("COLUMNS")
plt.show()


# In[93]:


df['interest_rate_float'] = df['int_rate'].str.rstrip('%').astype('float')
subset_df = df[['total_pymnt', 'funded_amnt', 'pub_rec_bankruptcies', 'pub_rec','interest_rate_float','loan_amnt','total_pymnt_inv']]

# Calculate the correlation matrix
correlation_matrix = subset_df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # Inference:
# ### We can conclude that the 'total_pymnt' and 'total_pymnt_inv', then 'loan_amnt' and 'funded_amnt'  are highly correlated

# In[94]:


# categorized_counts
categorized_counts = df.groupby(['addr_state', 'loan_status']).size().reset_index(name='count')
plt.figure(figsize=(12, 6))
sns.barplot(data=categorized_counts, x='addr_state', y='count', hue='loan_status', palette='muted')
plt.title('Distribution of Loan Status by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Loan Status ')
plt.tight_layout()
plt.show()


# In[96]:


#Checking the number of unique values each column possess to identify categorical columns
cat_columns = df.nunique().sort_values().loc[(df.nunique() > 1) & (df.nunique() < 15)].index.tolist()


# # Inference:
# ## Ordered - Term,emp_length,pub_rec,pub_rec_bankruptcies,
# ## Unordered - Purpose,home_ownership,loan_status,verification_status

# # Converting interest rate column to quartiles for analyzing

# In[97]:


df['int_rate_float'] = df['int_rate'].str.rstrip('%').astype('float')

#finding quartiles
q1 = df['int_rate_float'].quantile(0.25)  # 1st quartile (25th percentile)
q2 = df['int_rate_float'].quantile(0.50)  # 2nd quartile (median)
q3 = df['int_rate_float'].quantile(0.75)  # 3rd quartile (75th percentile)


def assign_quartile(row):
  if row['int_rate_float'] <= q1:
    return 1
  elif (row['int_rate_float'] <= q2) & (row['int_rate_float'] >= q1):
    return 2
  elif (row['int_rate_float'] <= q3) & (row['int_rate_float'] >= q2):
    return 3
  elif row['int_rate_float'] >= q3:
    return 4

# Apply the function to create the 'Loan_quartile' column
df['Loan_quartile'] = df.apply(assign_quartile, axis=1)
df.drop('int_rate',axis=1,inplace=True)


# # Segmented dataframe

# In[98]:


charged_off=df[df["loan_status"]=="Charged Off"]#5627
fully_paid=df[df["loan_status"]=="Fully Paid"]#32950
home_ownership_own=df[df["home_ownership"]=="OWN"]#3058


# # Analyzing data from here.......

# ## Imbalance Data

# In[99]:


loan_status=df.groupby('loan_status').size()
loan_status
plt.pie(loan_status.values, labels=loan_status.index, autopct="%1.0f%%")
plt.title("Percentage of Each Category of loan status")
plt.savefig("loan_status")
plt.show()


# # Remove duplicates

# In[102]:


print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)


# # No duplicates removed

# # Remove outlier

# In[103]:


def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]
    return df_no_outliers

print(df.shape)

# Remove outliers from numeric columns
numeric_cols = df.select_dtypes(include='number').columns
df_no_outliers_iqr = remove_outliers_iqr(df[numeric_cols])

outlier_indices = df.index.isin(df_no_outliers_iqr.index)
df = df[outlier_indices]

print(df.shape)


# # setting high quartile values removes most values data ,so setting outliers very small value

# # Univariate Analysis

# In[100]:


visualize_loan_status_variable(charged_off, 'home_ownership')
#people who take loans most reside in rent house`
#We can conclude that Own house borrowers mostly pay their loans,we can strongly provide loan to own house borrowers


# # Inference:
# ## People who take loans mostly reside in rented homes.
# ## We can conclude that Own house borrowers mostly pay their loans, We can Strongly provide loans to Own house borrowers.

# In[101]:


visualize_loan_status_variable(charged_off, 'emp_length')


# # Inference:
# ## People who are employed less than 1 year are more prone to be defaulters.
# ## As the Emp length increases the charged off loan status rate decreases.

# In[63]:


visualize_loan_status_variable(charged_off, 'Loan_quartile')


# # Inference:
# ## Borrowers charge off the loan when Interest rate of the loan  is higher.

# In[64]:


visualize_loan_status_variable(fully_paid, 'Loan_quartile')


# In[65]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.xlabel('Loan Status')
plt.ylabel('Loan Amount')
plt.title('Distribution of Loan Amount by Loan Status')
plt.xticks(rotation=45)
plt.show()


# # Inference:
# ## Irrespective of the loan amount, the loan defaulters are present.

# # Bivariate Analysis

# In[66]:


visualize_loan_status_by_variable(df, 'verification_status')


# In[67]:


visualize_loan_status_by_variable(df, 'pub_rec')


# # Inference:
# ## The pub_rec of 0,1 and 2 have more defaulters and zero defaulters in 3 and 4.

# In[68]:


visualize_loan_status_by_variable(df, 'inq_last_6mths')


# # Inference
# ## The observed variation in trends may be because of the below reasons
# ## Seasonal Variations,
# ## Interest Rate Changes,
# ## Marketing or Promotional Activities
# ## Financial Planning or Budgeting. 

# In[69]:


visualize_loan_status_by_variable(df, 'delinq_2yrs')


# In[70]:


visualize_loan_status_by_variable(df, 'term')


# # Inference:
# ## The longer the term of the loan, more the defaulters.

# In[71]:


visualize_loan_status_by_variable(df, 'purpose')


# ## Inferences:
# ### The percentage of small business charged off is higher than other.
# ### So its clear that its better to avoid borrowers who take loan for small_business.

# In[72]:


visualize_loan_status_by_variable(df, 'pub_rec_bankruptcies')


# ## Inferences:
# ### The higher the Public bankruptcies , the higher the charge offs.

# #  Conclusions

# - Approximately 15% of loans end in charge offs, indicating a notable proportion compared to those fully and currently paid.
# - Notably high correlations exist between 'total_pymnt' and 'total_pymnt_inv', as well as between 'loan_amnt' and 'funded_amnt'.
# - Individuals with less than one year of employment exhibit a higher likelihood of defaulting.
# - There's a discernible trend of decreasing charge off rates with increasing employment length.
# - Borrowers tend to default more frequently when faced with higher interest rates.
# - Loan defaults are evident across various loan amounts, suggesting no dependency on loan size.
# - Longer loan terms correlate with higher default rates.
# - Small business loans exhibit a higher charge off percentage compared to other loan purposes, suggesting caution in lending to small business borrowers.
# - The incidence of charge offs increases with higher public bankruptcies, highlighting a potential risk factor in lending decisions.
# - The pub_rec of 0,1 and 2 have more defaulters and zero defaulters in 3 and 4.
# - The borrowers who have their home ownership as mortgage and rent are the prime defaulters of the loan.

# In[ ]:




