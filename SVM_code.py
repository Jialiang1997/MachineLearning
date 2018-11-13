#Jialiang Liang Pro3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
style.use('ggplot')


whole = pd.read_csv('../data/adult.csv')
#print (whole.head(3))
#print (whole.info())

#### Abandon samples with missing term
whole = whole.replace('?',np.nan)
whole = whole.dropna(axis=0)
print ('After preprocessing, there are {0} rows in the whoel dataset'.format(len(whole)))
#print (whole.head(3))

#### Explore the number of class for each object column
def num_of_class(whole,col_name):
    unique_class = whole[col_name].unique()
    print ('The number of class for column {0} is {1}.\n'.format(col_name,unique_class.shape[0]))
    #print ('The unique item of class {0} is {1} \n'.format(col_name,unique_class))
    
for col in whole.columns:
    if whole[col].dtype == "object":
        num_of_class(whole,col)


#### Feature reconstruction
whole["marital.status"] = whole["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
whole['native.country'][whole['native.country']!="United-States"] = "None-US"


#### 
def label_encode(unique,to_transform):
    le = LabelEncoder()
    le.fit(unique)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return le.transform(to_transform),le_name_mapping

def encode(data,names,encode_fun):
    map_rule = []
    for name in names:
        data[name] = data[name].astype(str)
        unique = pd.Series(data[name]).unique()
        to_transform = data[name]
        transform,mapping = encode_fun(unique,to_transform)
        map_rule.append(mapping)
        data[name] = transform
        
    return data,map_rule

names = ["workclass","education","marital.status","occupation","relationship","race","sex","income","native.country"]
encode_whole,map_rule = encode(whole,names,label_encode)

#for rule in map_rule:
#    print (rule)

#X = encode_whole[["workclass","education","marital.status","occupation","relationship","race","sex","native.country"]]
#"workclass","occupation","native.country","relationship","race"
col_name = encode_whole.columns 
col_to_drop = ["income","education"]
X = encode_whole[col_name.drop(col_to_drop)]
y = encode_whole["income"]

#### Get the information gain of each feature

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)

#clf = DecisionTreeClassifier(criterion="gini",random_state=0)
clf = RandomForestClassifier(criterion='gini',n_estimators=100)
clf.fit(X,y)

important_features = X.columns 
feature_importance = clf.feature_importances_                                            
sorted_idx = np.argsort(feature_importance)[::-1]
sort_features = important_features[sorted_idx[::-1]]
sort_value = feature_importance[sorted_idx[::-1]]

for i in range(len(sort_features)):
    print ("The importance of feature {0} is {1}".format(sort_features[i],sort_value[i]))

x1_var = "age"
x2_var = "fnlwgt"

#### Normalize the training data
from sklearn import preprocessing
X_value = X.values
scaler = preprocessing.MinMaxScaler().fit(X_value)
X_value = scaler.transform(X_value)
X = pd.DataFrame(X_value,columns=X.columns)

#### train the model
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.svm import SVC
from sklearn.metrics import precision_score,f1_score,recall_score,explained_variance_score,accuracy_score


skf = StratifiedKFold(n_splits=10,random_state=40)#,shuffle=True) 
num = 5000
label = y[:num]
fea = X[[x1_var,x2_var]][:num]

for train_index, test_index in skf.split(fea, label):
    X_train, X_test = fea.iloc[train_index], fea.iloc[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]
    
    clf = SVC(kernel="linear",C=10)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    variance = np.var(y_pred,ddof=1)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test,y_pred)
    print ("The precision is {0}".format(precision))
    print ("The variance is {0}".format(variance))
    print ("The recall is {0}".format(recall))
    print ("The f1 is {0}".format(f1))
    print ("The accuracy is {0}".format(acc))
    
plt.scatter(X_train[x1_var], X_train[x2_var], c=y_train, s=30, cmap=plt.cm.Paired)
plt.xlabel(x1_var)
plt.ylabel(x2_var)
# plot the decision function    
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
    
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors=['g','r','g'], levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
plt.show()

### explore with the different kernel and C-value
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.svm import SVC
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=40)
num = -1
label = y[:num]
fea = X[:num]

def train_SVM(fea,label,kernel,c_value):
    
    pre_mean = []
    var_mean = []
    recall_mean = []
    f1_mean = []
    acc_mean = []
    
    for train_index, test_index in skf.split(fea, label):
        X_train, X_test = fea.iloc[train_index], fea.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        
        clf = SVC(kernel=kernel,C=c_value) 
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        precision = precision_score(y_test, y_pred)
        variance = np.var(y_pred,ddof=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test,y_pred)
        
        pre_mean.append(precision)
        var_mean.append(variance)
        recall_mean.append(recall)
        f1_mean.append(f1)
        acc_mean.append(acc)
    
    print ("The precision of kernel {0} with C {1} is {2}".format(kernel,c_value,sum(pre_mean)/(len(pre_mean)*1.0)))
    print ("The variance of kernel {0} with C {1} is {2}".format(kernel,c_value,sum(var_mean)/(len(var_mean)*1.0)))
    print ("The recall of kernel {0} with C {1} is {2}".format(kernel,c_value,sum(recall_mean)/(len(recall_mean)*1.0)))
    print ("The f1 of kernel {0} with C {1} is {2}".format(kernel,c_value,sum(f1_mean)/(len(f1_mean)*1.0)))
    print ("The accuracy of kernel {0} with C {1} is {2}".format(kernel,c_value,sum(acc_mean)/(len(acc_mean)*1.0)))

#'linear',
for kernel in ['sigmoid','linear','rbf','poly']:
    for c_value in [1.1,1.2,1.5]:
        train_SVM(fea,label,kernel,c_value)



#### Draw the ROC curve
from sklearn.metrics import roc_curve, auc

def draw_roc_curve(y_test,y_predict_1,y_predict_2):
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_test,y_predict_1[:,1])
    fpr_2, tpr_2, thresholds_2 = roc_curve(y_test,y_predict_2[:,1])
    
    roc_auc_1 = auc(fpr_1,tpr_1)
    roc_auc_2 = auc(fpr_2,tpr_2)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_1, tpr_1, 'b',label='AUC = %0.2f'% roc_auc_1)
    plt.plot(fpr_2, tpr_2, 'r',label='AUC = %0.2f'% roc_auc_2)
    
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

num = 10000
label = y[:num]
fea = X[:num]
for train_index, test_index in skf.split(fea, label):
    X_train, X_test = fea.iloc[train_index], fea.iloc[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        
    rf = RandomForestClassifier(n_estimators=100,random_state=23)#, max_features=4,max_depth=8) 
    svm = SVC(probability=True,kernel='rbf')
    rf.fit(X_train,y_train)
    svm.fit(X_train,y_train)
    rf_y_pred_prob = rf.predict_proba(X_test)
    svm_y_pred_prob = svm.predict_proba(X_test)
    
    draw_roc_curve(y_test,rf_y_pred_prob,svm_y_pred_prob)







