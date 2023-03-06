#import liberaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score
import pickle

# this code to display two data frame horizontal
from IPython.display import display_html
from itertools import chain, cycle


def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2>{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)

#load data
data=pd.read_csv(r"diabetes.csv")
data

# from here you can see min , max,std and  count of each columns
data.describe()

#to show number of null ,type of data and len of data
data.info()

# Show the sum of duplicated in Data
DuplicatedDataSum = data.duplicated().sum()
print("Sum of the Dublicate in Data",DuplicatedDataSum)


# to  show number of patient and normal in data
print("Sum of the normal in Data ",data['Outcome'].value_counts()[0])
print("Sum of the un normal in Data ",data['Outcome'].value_counts()[1])
data['Outcome'].value_counts().plot(kind='bar', title='count (target)')

# to show unique_value for and columns
columns = data.columns.values
unique = []
data_unique = {'columns': columns, 'unique_value': unique}
for column in columns:
    unique.append(len(data[column].unique()))

data_unique = pd.DataFrame(data_unique)
# to sort data
data_unique.sort_values(by=['unique_value'], inplace=True)
data_unique

# to show number of patient for value of Pregnancies
age = []
diabetes_patient = []
age_and_diabetes = {'age': age, 'diabetes_patient': diabetes_patient}

for i in data['Age'].unique():
    diabetes_patient.append(data[data['Age'] == i]['Outcome'].sum())
    age.append(i)

age_and_diabetes = pd.DataFrame(age_and_diabetes)
# to sort data
age_and_diabetes.sort_values(by=['diabetes_patient'], inplace=True)

# to show number of patient for every value of Pregnancies
Pregnancies = []
diabetes_patient = []
Pregnancies_and_diabetes = {'Pregnancies': Pregnancies, 'diabetes_patient': diabetes_patient}

for i in data['Pregnancies'].unique():
    # print(data[data['continent']==i]['gdp_cap'].sum(),i)
    diabetes_patient.append(data[data['Pregnancies'] == i]['Outcome'].sum())
    Pregnancies.append(i)

Pregnancies_and_diabetes = pd.DataFrame(Pregnancies_and_diabetes)
# to sort data
Pregnancies_and_diabetes.sort_values(by=['diabetes_patient'], inplace=True)


x = data.drop('Outcome', axis=1)
y = data['Outcome']
print('len of data ',len(data))

# len data is very small and therefore we use algrithm not complicated like logistic regreesion
# we split data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=100,random_state=0)
print("x_train shape is:",x_train.shape)
print("x_test shape is:",x_test.shape)
print("y_train shape is:",y_train.shape)
print("y_test shape is:",y_test.shape)


#the expected is true and algrithm suffer from over fitting

#we use LogisticRegression to overcome over fitting
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



# fit the model with data
model.fit(x_train,y_train)

#
y_pred=model.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print('LogisticRegression Train Score is : ' , model.score(x_train, y_train))

print('LogisticRegression Test Score is : ' , model.score(x_test, y_test))

# we plot auc
y_pred_proba = model.predict_proba(x_test)[::,1]
fpr, tpr, thre = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

confusion_matrix =confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_matrix, vmax=.8, square=True,annot=True)

#to know index of False_positive in data
Covid=y_test[y_test==1]
Covid=list(Covid.index)

Normal=y_test[y_pred==0]
Normal=list(Normal.index)

False_positive=[ e for e in Covid if e in Normal ]

len(False_positive)

#to know index of False_negative in data

Covid=y_test[y_test==0]
Covid=list(Covid.index)

Normal=y_test[y_pred==1]
Normal=list(Normal.index)
False_negative=[ e for e in Covid if e in Normal ]

len(False_negative)

# we display False_positive and False_negative
display_side_by_side(data.loc[False_positive],data.loc[False_negative] ,titles=['False_positive','False_negative'])

# we show data , show False_positive and False_negative
fig,axs=plt.subplots(4,2)
feature=list(data.columns)
index=-1
for row in range(4):
    for columns in range(2):
        index+=1
        axs[row,columns].scatter(data[feature[index]],data['Outcome'], label=f" {feature[index]}",c='b')
        axs[row,columns].scatter(data.loc[False_positive][feature[index]],data.loc[False_positive]['Outcome'], label="False_positive",c='r')
        axs[row,columns].scatter(data.loc[False_negative][feature[index]],data.loc[False_negative]['Outcome'], label="False_negative",c='y')
        axs[row,columns].legend(loc=2)

plt.subplots_adjust(left=0,right=1.9,bottom=0,top=1.9)

#we will detect outlier

outlier_sample=[]
drop = data[(data['DiabetesPedigreeFunction']>=2)&(data['Outcome']>0)].index.values
outlier_sample.extend([drop][0])
drop = data[(data['Insulin']>=800)&(data['Outcome']==1)].index.values
outlier_sample.extend([drop][0])

print('the outlier is ',outlier_sample)

data.drop(outlier_sample, axis=0, inplace=True)

x = data.drop('Outcome', axis=1)
y = data['Outcome']
print('len of data ',len(data))

# we split data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=100,random_state=0)
print("x_train shape is:",x_train.shape)
print("x_test shape is:",x_test.shape)
print("y_train shape is:",y_train.shape)
print("y_test shape is:",y_test.shape)

#we use LogisticRegression to overcome over fitting

model = LogisticRegression()



# fit the model with data
model.fit(x_train,y_train)

#
y_pred=model.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print('LogisticRegression Train Score is : ' , model.score(x_train, y_train))

print('LogisticRegression Test Score is : ' , model.score(x_test, y_test))
print(x_test.head(3))
print(y_pred[2])
print(model.predict(np.array([(8,196,76,29,280,37.5,0.605,57)])))


#inputt=[int(x) for x in "45 32 60".split(' ')]
#final=[np.array(inputt)]

#b = model.predict_proba(final)


pickle.dump(model,open('diabetes.pkl','wb'))
model=pickle.load(open('diabetes.pkl','rb'))
print("done")

#we see the accuracy increase when we delete outllier