<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# Pipeline

這邊要介紹管道的使用，第一種用法用 Pipeline，
使用 key, value 的方法建立 Pipeline 。



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


也可以利用 make_pipeline 快速建立 pipeline，他會自動填入名稱，他會把你的名稱都改為小寫自動填入。



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
