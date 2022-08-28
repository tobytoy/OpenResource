今天會介紹兩個方法去做存檔讀檔， ```pickel``` 跟 ```joblib```。


```python
# pickle save
from sklearn import svm
from sklearn import datasets
import pickle

model_path = 'model.pickle'

classifier = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)

classifier.fit(X, y)

with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)
```


```python
# pickle load
import pickle

with open(model_path, 'rb') as f:
    classifier_load = pickle.load(f)
```


```python
# check pickel
print('classifier pred:', classifier.predict(X[0:10]))
print('classifier load pred:', classifier_load.predict(X[0:10]))

```

下面示範 joblib


```python
# joblib save
import joblib
from sklearn import svm
from sklearn import datasets


model_path = 'model.joblib'

classifier = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)

classifier.fit(X, y)

with open(model_path, 'wb') as f:
    joblib.dump(classifier, f)

```


```python
# joblib load
import joblib

with open(model_path, 'rb') as f:
    classifier_load = joblib.load(f)

```


```python
# joblib check

print('classifier pred:', classifier.predict(X[0:10]))
print('classifier load pred:', classifier_load.predict(X[0:10]))

```

scikit learn 官方是推薦使用 joblib，
官方提到 joblib 比 pickel 更高效，對於大的 numpy 陣列更好。



