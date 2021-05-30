import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# load data
df = pd.read_csv('bank-full.csv', delimiter=';', skipinitialspace=True)

# split inputs and outputs
X = df.drop(columns=['y'])
y = df['y']

# split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#print("X_train: ", X_train.shape)
#print("y_train ", y_train.shape)
#print("X_test: ", X_test.shape)
#print("y_test: ", y_test.shape)

# columns with numeric value
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.values
numeric_features = numeric_features[numeric_features != 'y']

# columns with text value --> will be transformed
category_features = X_train.select_dtypes(include=['object', 'bool']).columns.values

#print(numeric_features)
#print(category_features)

# function for splitting columns with text value to each value having own column
def dummify(ohe, x, columns):
    transformed_array = ohe.transform(x)

    enc = ohe.named_transformers_['cat'].named_steps['onehot']
    feature_lst = enc.get_feature_names(category_features.tolist())   
    
    cat_colnames = np.concatenate([feature_lst]).tolist()
    all_colnames = numeric_features.tolist() + cat_colnames 
    
    df = pd.DataFrame(transformed_array, index = x.index, columns = all_colnames)
    
    return transformed_array, df

# replace missing values with medians and scale values
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# replace missing values and onehot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# transfrom
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, category_features)])

ohe = preprocessor.fit(X_train)

X_train_t = ohe.transform(X_train)
X_test_t = ohe.transform(X_test)

# transform training and test set and then convert it to dataframe
X_train_t_array, X_train_t = dummify(ohe, X_train, category_features)
X_test_t_array, X_test_t = dummify(ohe, X_test, category_features)

X_train_t.head()

X_train_columns = X_train_t.columns
#print(X_train_columns)

X_train_columns = X_train_t.columns
#print(X_train_columns)

# summarize class distribution
counter = Counter(y_train)
#print(counter)

# transform the dataset
oversample = SMOTE()
X_train_smote, y_train = oversample.fit_resample(X_train_t, y_train)

# summarize the new class distribution
counter = Counter(y_train)
#print(counter)

final_X_train = pd.DataFrame(data=X_train_smote,columns=X_train_columns )
final_y_train = pd.DataFrame(data=y_train,columns=['y'])

rfe_model = RFE(LogisticRegression(solver='lbfgs', max_iter=1000), 25)
rfe_model = rfe_model.fit(final_X_train, np.ravel(final_y_train))

selected_columns = X_train_columns[rfe_model.support_]
print(selected_columns.tolist())

X_train_final = final_X_train[selected_columns.tolist()]
y_train_final = final_y_train['y']
X_test_final = X_test_t[selected_columns.tolist()]
y_test_final = y_test

X_test_final.head()

logreg = LogisticRegression()
logreg.fit(X_train_final, y_train_final)

y_pred = logreg.predict(X_test_final)

print('\n\n -------------------------- RESULTS -------------------------- \n\n')

print('ACCURACY: \n {:.2f}'.format(logreg.score(X_test_final, y_test_final)))

cm = confusion_matrix(y_test_final, y_pred)

index = 0
for vals in cm:
    for val in vals:
        if index == 0:
            pnan = val
        elif index == 1:
            pyan = val
        elif index == 2:
            pnay = val
        elif index == 3:
            pyay = val
        else:
            print('something went wrong')
        index = index+1 

table = [['\t', 'predicted NO', 'predicted YES'],
         ['actual NO', (str(pnan) + '\t'), str(pyan)],
         ['actual YES', (str(pnay) + '\t'), str(pyay)]]

print('\nCONSUSION MATRIX:')
row_nbr = 0
for row in table:
    if row_nbr != 0:
        print('\n')
    for col in row:
        sys.stdout.write(col + '\t')
        row_nbr = row_nbr+1

# Recall
print('\n\nRECALL')
rs = recall_score(y_test_final, y_pred, average=None)
for val in rs:
    sys.stdout.write('{:.2f} \t'. format(val))

# Precision
print('\n\nPRECISION')
ps = precision_score(y_test_final, y_pred, average=None)
for val in ps:
    sys.stdout.write('{:.2f} \t'. format(val))

print('\n\n ------------------------------------------------------------- \n\n')

print('\nSELECTED VARIABLES\n')
row=0
for var in selected_columns.tolist():
    sys.stdout.write(var + '\t')
    row = row+1
    if row%2 == 0:
        print('\n')