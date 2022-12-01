

# Pipeline

這邊要介紹管道 （Pipeline） 的使用方法，第一種是使用 key, value 的方法去建立 Pipeline 。




```python 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

estimators = [
    ('reduce_dim', PCA()), 
    ("scaler", StandardScaler()),
    ('classifier', SVC())
]

pipeline = Pipeline(estimators)
pipeline

```


也可以利用 make_pipeline 指令快速建立 pipeline，他會幫忙自動填入名稱，把你填入的名稱都自動改為小寫。



```python 
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer

pipeline = make_pipeline(Binarizer(), MultinomialNB())
pipeline

```


# 訪問


```python 
pipeline.steps[1]
```


# 得到所有參數


```python 
pipeline.get_params().keys()
```
