### NAME: MUHAMMAD ASJAD E
### REG.NO : 212225240091
### DTE : 03.11.2026


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
df=pd.read_csv('/content/bmi.csv') 
df.head()
```

<img width="605" height="286" alt="image" src="https://github.com/user-attachments/assets/0a2972f3-988a-44f6-b8d4-cd6a65631dd5" />


```
df.isnull().sum()
```


<img width="228" height="276" alt="image" src="https://github.com/user-attachments/assets/0d7504f7-8840-40fa-8793-89247a902a95" />


```
df.dropna()
```

<img width="332" height="297" alt="image" src="https://github.com/user-attachments/assets/8c578fbe-8994-4a48-959e-2a5781eca7bc" />


```
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("/content/bmi.csv") 
df1.head()
```

<img width="395" height="152" alt="image" src="https://github.com/user-attachments/assets/942f02d1-f2d1-4009-b9ba-918bd1a1276d" />



```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
```

<img width="285" height="255" alt="image" src="https://github.com/user-attachments/assets/70fab7e1-7287-4f51-b389-d7a8366b7b17" />


```
#MIN-MAX SCALING: 
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df.head(10)
```

<img width="337" height="248" alt="image" src="https://github.com/user-attachments/assets/f2478a11-50c7-4b75-93ba-ab5fa9131ff6" />



```
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler() 
df3=pd.read_csv("bmi.csv") 
df3.head()
```

<img width="301" height="140" alt="image" src="https://github.com/user-attachments/assets/8671413b-d511-461d-bc0d-eb4f747bf71b" />



```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df
```


<img width="355" height="296" alt="image" src="https://github.com/user-attachments/assets/9c155a28-7f4f-4c0a-9303-06aed17e7cac" />



```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```

<img width="352" height="148" alt="image" src="https://github.com/user-attachments/assets/00dc97d4-b5c1-4310-981c-cfabcdb053a3" />



```
df=pd.read_csv("/content/income(1) (1).csv") 
df.info()
```

<img width="340" height="270" alt="image" src="https://github.com/user-attachments/assets/a696c750-8735-42a7-bde8-c3a8f6131fd2" />


```
df.isnull().sum()
```

<img width="122" height="358" alt="image" src="https://github.com/user-attachments/assets/95c11fc0-91e8-4f57-88e8-376704dc4c72" />



```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="690" height="316" alt="image" src="https://github.com/user-attachments/assets/b42b9ff7-20bf-485f-86f0-7f7e6aedd148" />



```
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories. 
df[categorical_columns]
```

<img width="592" height="292" alt="image" src="https://github.com/user-attachments/assets/ef285d20-3757-4fd1-86d9-57945734b68d" />


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

<img width="295" height="67" alt="image" src="https://github.com/user-attachments/assets/43cf8ddb-c916-45b2-9d0b-50611eb420d3" />


```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv") 
df.info()
```

<img width="353" height="255" alt="image" src="https://github.com/user-attachments/assets/9d9747d6-f212-48b8-ad02-bf24ac435366" />


```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns]
```

<img width="688" height="290" alt="image" src="https://github.com/user-attachments/assets/cd2d1c94-c4ff-4f4e-a1da-5dee07db7324" />


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
```

<img width="652" height="292" alt="image" src="https://github.com/user-attachments/assets/25b7bbab-5c06-411d-9194-a6d77f145030" />


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


<img width="507" height="68" alt="image" src="https://github.com/user-attachments/assets/ee6a03f4-22c9-4643-8723-138cecfcc98d" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split 
# Importing the missing function 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek'] 
X = df[selected_features]
y = df['SalStat'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```

<img width="246" height="57" alt="image" src="https://github.com/user-attachments/assets/8db158be-7b85-4c72-a00f-80450d27c739" />



```
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="417" height="33" alt="image" src="https://github.com/user-attachments/assets/a3b5ce25-8311-4d44-8e1c-0a4cb5efd645" />


```
!pip install skfeature-chappers

import numpy as np 
import pandas as pd
from skfeature.function.similarity_based import fisher_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
# @title 
df[categorical_columns]
```

<img width="992" height="487" alt="image" src="https://github.com/user-attachments/assets/2b8b3228-8f71-4d64-ad06-258d574ac4a7" />


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

<img width="568" height="52" alt="image" src="https://github.com/user-attachments/assets/7a42a004-7a4a-4e20-9bd6-09d5ba5f7065" />


```
import pandas as pd 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 
df=pd.read_csv("income(1) (1).csv") # List of categorical columns 
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ] # Convert the categorical columns to category dtype 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="562" height="281" alt="image" src="https://github.com/user-attachments/assets/72dfba01-fda5-4c05-bb3c-b7077429493c" />



```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select) 
rfe.fit(X, y)
```



<img width="915" height="622" alt="image" src="https://github.com/user-attachments/assets/c998903f-8654-47d5-b776-c345266cd6a9" />



# RESULT:
       
       To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file was done and executed sucessesfully
