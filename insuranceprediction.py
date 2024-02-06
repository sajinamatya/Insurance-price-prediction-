# %% [markdown]
# # Importing necessary libraries 

# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import statsmodels.api as sm 

# %% [markdown]
# # Data collection 

# %%
data = pd.read_csv("insurance.csv")
print(data)

# %% [markdown]
# # Explorative Data Analysis

# %% [markdown]
# ## Data inspection 

# %%
data.head()

# %%
data.tail()


# %% [markdown]
# ## Data types of the columns 

# %%
data.dtypes

# %% [markdown]
# ## Information about the column of a dataframe 

# %%
data.info()

# %% [markdown]
# ## Number of rows and columns in a dataframe

# %%
data.shape


# %% [markdown]
# ## Descriptive Statistics 

# %%
data.describe()


# %% [markdown]
# ## Inspection of missing value
# 

# %%
data.isnull().sum()

# %% [markdown]
# ## Univariate data analysis

# %%
plt.figure(figsize=(8,6))
sns.histplot(data['age'], bins=30, kde=True,color='red')
plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('Distribution of age')
plt.show()

# %%
plt.figure(figsize=(8,6))
sns.histplot(data['bmi'], bins=30, kde=True,color='green')
plt.xlabel('bmi')
plt.ylabel('Frequency')
plt.title('Distribution of bmi');
plt.show()

# %%
plt.figure(figsize=(8,6))
sns.histplot(data['charges'], bins=50, kde=True,color='red')
plt.xlabel('charge')
plt.ylabel('Frequency')
plt.title('Distribution of charge');
plt.show()

# %%
data['charges'].median()

# %%
data['region'].value_counts().plot(kind='bar',width =0.6,color = 'green')
plt.ylim(0,500)
plt.xlabel('Region')
plt.ylabel('count')
plt.title('countplot for region column')

# %%
data1 = data['sex'].value_counts()
plt.pie(data1,labels=data1.keys(), autopct='%.0f%%')

# %% [markdown]
# ## Bivariate analysis

# %%
sns.pairplot(data,diag_kind='kde',hue='sex')
plt.show()

# %%
sns.heatmap(data[['charges','children','bmi','age']].corr(),annot=True,cmap='coolwarm')

# %%
sns.jointplot(x='charges',y='age',data=data,kind='hex')

# %%
sns.regplot(x='charges',y='age',data=data,scatter_kws={'alpha': 0.5},line_kws={'alpha': 0.7, 'color': 'red'})

# %%
sns.jointplot(x='charges',y='bmi',data=data,kind='hex')

# %%
sns.regplot(x='charges',y='bmi',data=data,scatter_kws={'alpha': 0.5},line_kws={'alpha': 0.7, 'color': 'green'})

# %%
dada = pd.crosstab(data['region'],data['sex'])
print(dada)

# %%
dada.plot(kind='bar',stacked=False)
plt.ylabel('count')

# %% [markdown]
# # Label Encoding 

# %%
encoder = LabelEncoder()
encoded_gender = encoder.fit_transform(data['sex'])
data_sex = pd.DataFrame(encoded_gender)
data['sex_encoded'] = data_sex
encoded_smoker = encoder.fit_transform(data['smoker'])
data['encoded_smoker'] = encoded_smoker
print(data.head(20))


# %% [markdown]
# # Removing columns 

# %%

data.drop(['sex','smoker'], axis=1, inplace=False)
# 0 is female and no 

# %% [markdown]
# # Changing the order of the dataframe columns 

# %%
data = data.loc[:,['age','bmi','children','region','sex_encoded','encoded_smoker','charges']]
print(data)

# %% [markdown]
# # Feature  scaling 
# 

# %%
scaler =  MinMaxScaler()
scale_data =scaler.fit_transform(data[['age','bmi']])
data12 = pd.DataFrame(scale_data)
print(data12.head())

# %% [markdown]
# 
# # Feature Selection 

# %%
# feature 
X = data.drop(['charges','region'],axis=1)
y= data['charges'] # target data 
print(X)

# %% [markdown]
# # Train Test split 

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# %% [markdown]
# # Linear Regression model

# %%
model = LinearRegression()
model.fit(X_train, y_train)


# %% [markdown]
# # Prediction on training data 

# %%
y_pred = model.predict(X_train)

# %% [markdown]
# # Prediction on test data 

# %%
y_pred2 = model.predict(X_test)

# %%
print(y_pred.shape)
print(y_pred2.shape)


# %%
slope = model.coef_
print(slope)

# %%
intercept = model.intercept_
print(intercept)

# %% [markdown]
# # Model Evaluation on training data 

# %%
r2 = r2_score(y_train, y_pred)
print(r2)
y_predict = pd.DataFrame(y_pred)


# %%
y_predict.shape


# %% [markdown]
# # Model Evaluation on testing data

# %%
r2_test = r2_score(y_test,y_pred2)
print(r2_test)

# %% [markdown]
# ###  R2 score is higher on the testing data (0.78) compared to the training data (0.74) is typically considered favorable. It indicates that the model is likely not suffering from overfitting to the training data and has some ability to generalize to unseen data. 

# %%
X_test.shape

# %%
y_test.shape


# %%
import pickle 
with open('insurance1.pkl','wb') as file:
    pickle.dump(model,file)
    file.close()


# %% [markdown]
# # Model Building using statsmodel 

# %%

model1 = sm.OLS(y,X).fit()
print(model1.summary())

# %%
ols_predict = model1.predict(X)
print(ols_predict)

# %%
print(r2_score(y,ols_predict))

# %% [markdown]
# # Actual vs predicted value plot

# %%
sns.regplot(x=y_test, y= y_pred2, label='Actual', color='blue',line_kws={'alpha': 0.7, 'color': 'red'})

plt.plot()
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.title('Actual vs. Predicted Values')

# %% [markdown]
# # Residual 

# %%
residual = y_train - y_pred

# %% [markdown]
# # Ploting the residual 

# %%
sns.kdeplot(residual,shade = True)
plt.title("kernel density plot of residual value")

# %%
sns.residplot(x= y_train,y= y_pred, color = 'red')


