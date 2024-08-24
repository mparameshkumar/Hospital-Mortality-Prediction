import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("C:\HMP\HMP\data01.csv")
df.head(10)
df.info()
del(df['ID'])
df.memory_usage(deep=True)
df.drop_duplicates(inplace=True)
df['outcome'].unique()
df.groupby('age')['outcome'].mean()
df.groupby('age')['outcome'].mean().nlargest(20).plot.bar()
df.corr()['outcome']
sns.set_style(style='whitegrid')
sns.distplot(df['age'])
plt.figure(figsize=(8,4))
chains=df['age'].value_counts()[0:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most age")
plt.xlabel("Number of ouccurrences")
df.columns
df.column=['group', 'age', 'gendera', 'BMI', 'hypertensive',
       'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
       'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'heart rate',
       'Systolic blood pressure', 'Diastolic blood pressure',
       'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
       'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte',
       'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
       'NT-proBNP', 'Creatine kinase', 'Creatinine', 'Urea nitrogen',
       'glucose', 'Blood potassium', 'Blood sodium', 'Blood calcium',
       'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbonate',
       'Lactic acid', 'PCO2', 'EF']

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']
print(cat_cols)
print(num_cols)
plt.figure(figsize = (20, 15))
plotnumber = 1
for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(df[column],color='blue',)
        plt.xlabel(column)


    plotnumber += 1

plt.tight_layout()
plt.show()
px.scatter(df, x="age", y="heart rate", color="outcome")
df.isnull().sum()
feature_na=[feature for feature in df.columns if df[feature].isnull().sum()>0]
feature_na
for feature in feature_na:
    print('{} has {} % missing values'.format(feature,np.round(df[feature].isnull().sum()/len(df)*100,4)))
def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
for col in num_cols:
    random_value_imputation(col)
df[num_cols].isnull().sum()
df['outcome'].value_counts()
df['age'].value_counts()
df['heart rate'].value_counts()
x = df.drop('outcome', axis=1)
y = df['outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("Shape of X_train:" ,X_train.shape)
print("Shape of y_train:" ,y_train.shape)
print("Shape of X_test:"  ,X_test.shape)
print("Shape of y_test:"  ,y_test.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 20, criterion = 'gini', random_state =0)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)
y_pred[:10]
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix
from sklearn.metrics import roc_curve, auc #for model evaluation
from matplotlib import pyplot
fig, (ax2) = plt.subplots(figsize = (5,4))
fpr, tpr, thresholds_roc = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)
ax2.plot(fpr,tpr, label = " AUROC = {:0.2f}".format(roc_auc))
ax2.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
ax2.set_xlabel("False Positive Rate", fontsize = 14)
ax2.set_ylabel("True Positive Rate", fontsize = 14)
ax2.set_title("ROC Curve", fontsize = 18)
ax2.legend(loc = 'best')
plt.title('ROC curve for RandomForestClassifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
        #find default threshold
close_default = np.argmin(np.abs(thresholds_roc - 0.5))
ax2.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)
plt.tight_layout()
from xgboost import XGBClassifier
from sklearn import metrics
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
print('model_xgb Train Score is : ' , model_xgb.score(X_train, y_train))
print('model_xgb Test Score is : ' , model_xgb.score(X_test, y_test))
y_pred1 = model_xgb.predict(X_test)
y_pred1[:10]
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix2=confusion_matrix(y_test,y_pred1)
confusion_matrix2
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred1)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=model_xgb.classes_)
disp.plot()
plt.show()
from sklearn.metrics import roc_curve, auc
#for model evaluation
from matplotlib import pyplot
fig, (ax2) = plt.subplots(figsize = (5,4))

fpr, tpr, thresholds_roc = roc_curve(y_test,y_pred1)
roc_auc = auc(fpr,tpr)
ax2.plot(fpr,tpr, label = " AUROC = {:0.2f}".format(roc_auc))
ax2.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
ax2.set_xlabel("False Positive Rate", fontsize = 14)
ax2.set_ylabel("True Positive Rate", fontsize = 14)
ax2.set_title("ROC Curve", fontsize = 18)
ax2.legend(loc = 'best')
plt.title('ROC curve for Decision Tree Classifier ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
        #find default threshold
close_default = np.argmin(np.abs(thresholds_roc - 0.5))
ax2.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)
plt.tight_layout()
