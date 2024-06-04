# Election Data Analysis and Modeling
![screenshot](2880px-Lok_Sabha_Zusammensetzung_2019.svg.png)

This project involves analyzing election data to determine various factors that influence election outcomes. It includes data preprocessing, visualization, statistical tests, and machine learning modeling to predict election winners.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Statistical Analysis](#statistical-analysis)
- [Machine Learning Modeling](#machine-learning-modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost lightgbm
```
## Usage
- Clone the repository and navigate to the project directory.
- Ensure your data is in a suitable format, and adjust the code to load your specific dataset.
- Run the script to perform data preprocessing, visualization, statistical analysis, and machine learning modeling.
- Data Preprocessing
- Removing Estimation.The remove_estimation function cleans the asset values by removing any estimations indicated by the '~' symbol and the 'Rs' prefix.
```python
def remove_estimation(value):
    if value.startswith('Rs'):
        value = value.replace('Rs', '').strip()
    return value.split('~')[0].strip()

df['Total Assets'] = df['Total Assets'].astype(str).apply(remove_estimation)
df['Liabilities'] = df['Liabilities'].astype(str).apply(remove_estimation)
```
## Handling Missing Values and Data Transformation
Replace 'Nil' values with NaN and convert columns to appropriate types.

```python
df_inter['Total Assets'] = df_inter['Total Assets'].replace('Nil', np.nan).astype(float).fillna(0)
```

## Data Visualization
- Several visualizations are created to understand the distribution of different features and their relationship with election outcomes.

## State Frequency Plot
```python
state_counts = df_inter["State"].value_counts()
state_counts.plot(kind='bar')
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Frequency of States')
plt.show()
```
## Education Level Distribution
```python
sns.countplot(x='Education', data=df_inter)
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.title('Distribution of Education Levels')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```
## Education Level vs Win/Lose
```python
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Is Winner', data=df_inter)
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.title('Education Level vs Is Winner')
plt.xticks(rotation=45)
plt.legend(title='Is Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Criminal Cases vs Win/Lose
```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Criminal Cases', hue='Is Winner', data=df_inter)
plt.xlabel('Criminal Cases')
plt.ylabel('Count')
plt.title('Criminal Cases vs Win/Lose')
plt.legend(title='Is Winner')
plt.tight_layout()
plt.show()
```
## Statistical Analysis
### T-test for Total Assets
```python
t_stat, p_value = ttest_ind(winners_assets, losers_assets, equal_var=False)
print("T-statistic:", t_stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```
## Logistic Regression Analysis for Criminal Cases
```python
X = sm.add_constant(df_inter['Criminal Cases'])
y = df_inter['Is Winner']
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

p_value_criminal_cases = result.pvalues['Criminal Cases']
alpha = 0.05
if p_value_criminal_cases < alpha:
    print("Reject the null hypothesis: 'Criminal Cases' is a significant predictor of election outcomes.")
else:
    print("Fail to reject the null hypothesis: 'Criminal Cases' is not a significant predictor of election outcomes.")
```
## Logistic Regression Analysis for Age
```python
X = sm.add_constant(df_inter['Age'])
y = df_inter['Is Winner']
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

p_value_age = result.pvalues['Age']
alpha = 0.05
if p_value_age < alpha:
    print("Reject the null hypothesis: Age has an effect on winning elections.")
else:
    print("Fail to reject the null hypothesis: Age has no effect on winning elections.")
```
## Logistic Regression Analysis for Age Groups
```python
bins = [0, 30, 40, 50, 60, 70, np.inf]
labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
df_inter['Age Group'] = pd.cut(df_inter['Age'], bins=bins, labels=labels, right=False)

age_dummies = pd.get_dummies(df_inter['Age Group'], drop_first=True)

X = sm.add_constant(age_dummies)
y = df_inter['Is Winner']
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

p_values = result.pvalues
alpha = 0.05

for group in age_dummies.columns:
    p_value = p_values[group]
    if p_value < alpha:
        print(f"Reject the null hypothesis for age group {group}: This age group has an effect on winning elections.")
    else:
        print(f"Fail to reject the null hypothesis for age group {group}: This age group has no effect on winning elections.")
```
## Machine Learning Modeling
```python
model_param = {
    'LogisticRegression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'param': {
            'penalty': ('l1', 'l2'),
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'param': {
            'max_depth': [3, 5]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'param': {
            'n_neighbors': [5, 25]
        }
    },
    'SVC': {
        'model': SVC(),
        'param': {
            'C': [10, 100]
        }
    },
    'RandomForest
```






