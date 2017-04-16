#學號：0524817，學生姓名：陳以浩
#105年第二學期，雲端運算期中作業：自scikit-learn網站上的CLASSIFICATION，自選一個分類法實作，並列出confusion matrix，計算f1-score
#自選分類法為 Passive Aggressive Classifier

#以下自web抓取iris.csv檔案，已修正部分程式碼以符合python3
from urllib.request import urlopen
from contextlib import closing
url = 'http://aima.cs.berkeley.edu/data/iris.csv'
with closing(urlopen(url)) as u, open('iris.csv', 'w') as f:
    iris_file = u.read()
    f.write(iris_file.decode('utf-8'))

#-----------------------------------------------------
#以下僅保留期中作業CLASSIFICATION所需程式碼
from numpy import genfromtxt ,zeros
# read the first 4 columns
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

#以下使用CLASSIFICATION,Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier()
classifier.fit(data,t) # training on the iris dataset

#print(classifier.predict(data[0]))
#print(t[0])

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t, test_size=0.4, random_state=0)
classifier.fit(train,t_train) # train
print(classifier.score(test,t_test)) # test
print()

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(classifier.predict(test),t_test))
print()

#計算f1-score
from sklearn.metrics import classification_report
print(classification_report(classifier.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica']))
print()

#-----------------------------------------------------

from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier, data, t, cv=6)
print(scores)

from numpy import mean
print(mean(scores))
print()