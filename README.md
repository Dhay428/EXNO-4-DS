# NAME: DHAYALAPRABU.S
# REG NO: 212224230065

# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

<img width="434" height="264" alt="image" src="https://github.com/user-attachments/assets/fc8635e6-b8e9-42bc-8dbc-4388c3648b94" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="192" height="258" alt="image" src="https://github.com/user-attachments/assets/d84faa4f-d5a2-4753-979a-6bb5953a96b4" />

```
df.dropna()

```

<img width="351" height="519" alt="image" src="https://github.com/user-attachments/assets/9a6bafa9-d8df-4b04-a9d7-ef15d11b1f80" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

<img width="150" height="193" alt="image" src="https://github.com/user-attachments/assets/eb52e513-fa63-4d18-a28d-bd22d67012d2" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

```
<img width="403" height="250" alt="image" src="https://github.com/user-attachments/assets/eb6043f1-8729-44d7-a4d2-395668db1a08" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="394" height="456" alt="image" src="https://github.com/user-attachments/assets/7e133bf0-cab4-4aaa-9f4e-69358cefb32e" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="361" height="438" alt="image" src="https://github.com/user-attachments/assets/f93cb446-b7f3-4b9f-87c5-17bff3149c92" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="390" height="486" alt="image" src="https://github.com/user-attachments/assets/66208814-bc21-4af3-ae69-e50a4eae73c6" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="409" height="275" alt="image" src="https://github.com/user-attachments/assets/909749e0-5341-47a3-ab9e-75d259f682bb" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

```
<img width="621" height="463" alt="image" src="https://github.com/user-attachments/assets/423eb3dc-0cf3-4643-bf02-6e34de591756" />

```
df_null_sum=df.isnull().sum()
df_null_sum

```
<img width="188" height="615" alt="image" src="https://github.com/user-attachments/assets/eef88bff-eea9-4560-a3e2-fce8dc21b2a8" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```
<img width="1041" height="503" alt="image" src="https://github.com/user-attachments/assets/a51a3a80-aaf7-43a3-a820-cbb90507cc31" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```
<img width="919" height="501" alt="image" src="https://github.com/user-attachments/assets/ea80a6d5-ca4c-4d36-86f6-c5eaf63ab68f" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat'] 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```

<img width="436" height="75" alt="image" src="https://github.com/user-attachments/assets/78c0edc6-eb58-4430-8aba-c2e11d2e12d1" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="448" height="431" alt="image" src="https://github.com/user-attachments/assets/e210131d-8f40-4e56-a4d3-324030d4e06d" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```
<img width="1056" height="509" alt="image" src="https://github.com/user-attachments/assets/89d6e060-2b2b-447c-8f96-46c76bb913bc" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="937" height="483" alt="image" src="https://github.com/user-attachments/assets/0ee7c112-c393-4ea0-a1cc-83b56ad509a6" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="740" height="94" alt="image" src="https://github.com/user-attachments/assets/8eab334e-32f4-450f-8c4b-bb2943638e6d" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```
<img width="444" height="115" alt="image" src="https://github.com/user-attachments/assets/ff05472a-2f15-4fce-87ec-01f11fac66e5" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")

```

<img width="607" height="49" alt="image" src="https://github.com/user-attachments/assets/911cb8ef-fac1-449b-956e-c84f9bd74f85" />

```
!pip install skfeature-chappers
```
<img width="1502" height="357" alt="image" src="https://github.com/user-attachments/assets/ff7f88f0-1464-46e7-bb4b-cebbda309c86" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]

```
<img width="948" height="516" alt="image" src="https://github.com/user-attachments/assets/e1f0a454-9bab-4e37-87ac-60c4b1a2bd86" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)

```
<img width="860" height="85" alt="image" src="https://github.com/user-attachments/assets/36fa8783-731c-4d44-b454-b2361f2be73e" />

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```
<img width="944" height="488" alt="image" src="https://github.com/user-attachments/assets/bde66224-ef12-4698-ad76-176a0e5105a5" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

```
<img width="1326" height="643" alt="image" src="https://github.com/user-attachments/assets/1c8433cb-d5b8-4991-8207-31c6cb652d85" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
