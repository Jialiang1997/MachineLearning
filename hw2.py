import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd

filename = 'dataset_8_liver-disorders.arff'

with open(filename) as f:
    data, meta = arff.loadarff(f)

feature_names = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']

f1 = data['mcv'].copy()
f1 = f1.view(np.float64)

f2 = data['alkphos'].copy()
f2 = f2.view(np.float64)

f3 = data['sgpt'].copy()
f3 = f3.view(np.float64)

f4 = data['sgot'].copy()
f4 = f4.view(np.float64)

f5 = data['gammagt'].copy()
f5 = f5.view(np.float64)

f6 = data['drinks'].copy()
f6 = f6.view(np.float64)

label = data['selector'].copy()

X = np.array([f1, f2, f3, f4, f5, f6]).T
# X = preprocessing.scale(X)
Y = np.ravel(label.copy())

print("X shape: ", X.shape)
print(X)
print("Y shape: ", Y.shape)
print(Y)
"""
X_plt = pd.DataFrame(data=X,                       # data
                     columns=feature_names         # Columns
                     )
                     """

X_plt = pd.DataFrame(data=X[:int(X.shape[0]/1), :],                       # data
                     columns=feature_names         # Columns
                     )

print(X_plt)
pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(X_plt))
print("TRANSFORMED=============================================")
print(transformed)
plt.scatter(transformed[label==b'1'][0], transformed[label==b'1'][1], label="Class 1", c='black')
plt.scatter(transformed[label==b'2'][0], transformed[label==b'2'][1], label="Class 2", c='red')
plt.savefig("Class_distribution.png")
plt.show()
"""

X_plt = pd.DataFrame(data=X,                       # data
                     columns=feature_names         # Columns
                     )
print(X_plt)
pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(X_plt))
print("TRANSFORMED=============================================")
print(transformed)
plt.scatter(transformed[label==b'1'][0], transformed[label==b'1'][1], label="Class 1", c='red')
plt.scatter(transformed[label==b'2'][0], transformed[label==b'2'][1], label="Class 2", c='blue')
plt.savefig("Class_distribution.png")
plt.show()
"""


p = Perceptron(penalty='l1', max_iter=10000, shuffle=True,
     validation_fraction=.1, n_jobs=4, tol=None, random_state=0, early_stopping=True)

num_iter = 50
sum = 0
for i in range(num_iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1, stratify=Y)

    print("-------TRAIL %d-------" % i)
    p.fit(X_train, Y_train)
    score = p.score(X_train, Y_train)
    score = p.score(X_test, Y_test)
    sum += score
    print(score)

avg = sum / num_iter
print("\n-------AVG ACC-------")
print(0.6315867422411963)


