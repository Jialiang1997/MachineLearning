import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


pima_data = pd.read_csv('./data/diabetes.csv')
columns = pima_data.columns

print ("the data shape is ",pima_data.shape)
print (pima_data.head())
print (pima_data.describe())

print ('==========Histgram of labels==========\n')
pima_data[["Outcome"]].hist(figsize=(4,3))
plt.show()

print ('==========Histgram of features==========\n')
pima_data[columns.drop("Outcome")].hist(figsize=(12,8))
plt.show()

mnist_train = pd.read_csv('./data/MNIST/train.csv')

print (mnist_train.shape)
print (mnist_train.head())

X_train = mnist_train.drop(['label'],axis=1)
y_train = mnist_train[['label']]

y_train.hist(figsize=(5,4))
plt.show()

'''
draw the picture of the digit number
'''
def num2show(num_show,nums):
    data = mnist_train[mnist_train['label']==num_show].drop(['label'],axis=1)
    return data[:nums]

row = 5
for i in range(10):
    show_data = num2show(i,row).as_matrix()
    for j in range(row):
        plt_idx = j * 10 + i + 1
        plt.subplot(row,10, plt_idx)
        plt.imshow(show_data[j].reshape((28, 28)))
        plt.axis("off")
plt.show()

# check if there is any missing data
print (mnist_train[mnist_train.isnull().values==True])
print (pima_data[pima_data.isnull().values==True])

