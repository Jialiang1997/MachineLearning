import time
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(train_file,test_file,split):


    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    columns = train.columns
    train = train[:1000]
    
    #dataset[columns.drop('Outcome')] = StandardScaler().fit_transform(dataset[columns.drop('Outcome')])
    #train[columns.drop('label')] = MinMaxScaler().fit_transform(train[columns.drop('label')])
    y_label = train['label']
    train_data, test_data = train_test_split(train,train_size = split)#stratify=y_label)

    y_train = train_data[['label']]
    y_test = test_data[['label']]

    X_train = train_data[columns.drop('label')]
    X_test = test_data[columns.drop('label')]



    return X_train,y_train,X_test,y_test

def getCon_matrix(y_test,y_pre):
    y_test = y_test.as_matrix()
    con_ma = np.zeros(shape=[10,10])
    for x in range(len(y_test)):
        con_ma[y_test[x][0]][y_pre[x]] += 1
    print (con_ma)
    return con_ma

# This function is for calculating each number accuracy from confusion matrix
def get_each_accuracy(con_ma):
    accuracy = {}
    for i in range(10):
        temp = con_ma[i][i] / float(con_ma[i].sum())
        accuracy[i] = temp
    return accuracy

def knn(X_train,y_train,X_test,y_test,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    y_pre = neigh.predict(X_test)
    return y_pre

# number is the digit you want to see its accuracy vary with the K value
def get_graph(X_train,y_train,X_test,y_test,number):
    X_list = []
    y_list = []
    for i in range(2,11):
        X_list.append(i)
        y_pre = knn(X_train,y_train,X_test,y_test,i)
        con_ma = getCon_matrix(y_test,y_pre)
        accuracy = get_each_accuracy(con_ma)
        y_list.append(accuracy[number])

    plt.figure(figsize=(8,4))
    plt.plot(X_list,y_list,"b--",linewidth=1)  
    plt.xlabel("K value") 
    plt.ylabel("Accuracy") 
    plt.title("The accuracy vary with the K value") 
    plt.show()  
    #plt.savefig("pic.jpg") 




def main():

    X_train,y_train,X_test,y_test = load_data('./data/MNIST/train.csv','./data/MNIST/test.csv',0.8)
    ts = time.time()
    y_pre = knn(X_train,y_train,X_test,y_test,3)
    con_ma = getCon_matrix(y_test,y_pre)
    accuracy = get_each_accuracy(con_ma)
    print (accuracy)
    print ('The processing time is %s',time.time()-ts)
    get_graph(X_train,y_train,X_test,y_test,2)

    
if __name__ == "__main__":
    main()
