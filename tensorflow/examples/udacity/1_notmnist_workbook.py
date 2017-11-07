#!/usr/bin/python3
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
import time

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

import sys

target_names = list('abcdefghij'.upper())

dataset = pickle.load(open('notMNIST.pickle', 'rb'))

train_size = 400000
y_train = dataset['train_labels'] #[:train_size]
y_valid = dataset['valid_labels']
y_test = dataset['test_labels']

def reshape(dataset):
    a, b, c = dataset.shape
    return dataset.reshape((a, b*c))

X_train = dataset['train_dataset']
X_train = reshape(X_train) #[:train_size]
X_valid = dataset['valid_dataset']
X_valid = reshape(X_valid)
X_test = dataset['test_dataset']
X_test = reshape(X_test)

#clf = LogisticRegression(solver='saga', n_jobs=-1)

pca = PCA(n_components=50)
pca.fit(X_train)
print('pca.components', pca.components_.shape)

X_mod_train = pca.transform(X_train)
print('X_mod_train.shape', X_mod_train.shape)

#sys.exit(0)

clf = SVC()
start = time.time()
clf.fit(X_mod_train[:train_size], y_train[:train_size])
end = time.time()

print('fit duration %.2f seconds' % (end-start,))

f = open('SVC_pca_50components_400ktrain.pickle', 'wb')
pickle.dump({'pca': pca, 'clf': clf}, f)
f.close()

p_train = clf.predict(pca.transform(X_train[:train_size]))
p_valid = clf.predict(pca.transform(X_valid))
p_test = clf.predict(pca.transform(X_test))

p_train = np.array([int(a) for a in p_train])
p_valid = np.array([int(a) for a in p_valid])
p_test = np.array([int(a) for a in p_test])

print('SVC train score', clf.score(pca.transform(X_train[:train_size]), y_train[:train_size]))
print(classification_report(p_train, y_train[:train_size], target_names=target_names))
print(confusion_matrix(p_train, y_train[:train_size], labels=range(len(target_names))))

print('SVC valid score', clf.score(pca.transform(X_valid), y_valid))
print(classification_report(p_valid, y_valid, target_names=target_names))
print(confusion_matrix(p_valid, y_valid, labels=range(len(target_names))))

print('SVC test score', clf.score(pca.transform(X_test), y_test))
print(classification_report(p_test, y_test, target_names=target_names))
print(confusion_matrix(p_test, y_test, labels=range(len(target_names))))




