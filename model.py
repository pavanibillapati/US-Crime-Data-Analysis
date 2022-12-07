# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Trilo\Downloads\crimedata_updated.csv",sep= ',', encoding= "ISO-8859-1")
df=df.rename(columns = {'Êcommunityname':'Community Name'})
df = df.replace('?', '0')
df.loc[df['countyCode'] == '?']
df.loc[df['ViolentCrimesPerPop'] == '?']
violent_crimes = list(map(float, df.ViolentCrimesPerPop))
violent_crimes_mean = sum(violent_crimes)/len(violent_crimes)
df['mean_violent_crimes'] = violent_crimes_mean
df['violent_crime_occurence'] = np.where(violent_crimes>=df['mean_violent_crimes'], '1', '0')
df.groupby('violent_crime_occurence').mean()
df1 = df.iloc[:200]
age12t21 = df1['age'].astype(int)
age12t21.replace('?','0', inplace = True)
X_LogReg= ['countyCode','communityCode','age', 'PctUnemployed']
y_LogReg = df1[['violent_crime_occurence']]
X_train_LogReg, X_test_LogReg, y_train_LogReg, y_test_LogReg = train_test_split(df1[X_LogReg], y_LogReg, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train_LogReg, y_train_LogReg)
y_pred_LogReg = logreg.predict(X_test_LogReg)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_LogReg, y_test_LogReg)))
cnf_matrix_LogitRegression = metrics.confusion_matrix(y_test_LogReg, y_pred_LogReg)
cnf_matrix_LogitRegression

class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_LogitRegression), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Model Accuracy for Logistic Regression:",metrics.accuracy_score(y_test_LogReg, y_pred_LogReg))
model = LogisticRegression()
model.fit(X_train_LogReg, y_train_LogReg)
filename = 'US_crime_analysis_model.pkl'
pickle.dump(model, open(filename, 'wb'))
#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test_LogReg, y_test_LogReg)
print(result)