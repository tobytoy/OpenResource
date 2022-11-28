<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# feature selection

下面介紹一些常見的特徵選擇的方法，第一種就是根據 Variance 去做篩選，如果有非常相似的就可以直接刪除。






```python 
from sklearn.feature_selection import VarianceThreshold

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

```


我們也可以基於統計的變異數分析 （ANOVA） 去選，
使用 SelectKBest 直接選出最好的，
或是用 SelectPercentile 根據百分比去挑選出前面較好的特徵。




```python 
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)

print('Shape: ', X.shape)

# 最好的幾個
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print('Best 2: ', X_new.shape)

# 前百分比好的
X_new = SelectPercentile(chi2, percentile=30).fit_transform(X, y)
X_new.shape

```


下面看一下 pipeline 的使用


```python 
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)


model_pipeline = Pipeline(
    [
        ("anova", SelectPercentile(chi2)),
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1, random_state=87, tol=1e-5)),
    ]
)

percentiles = (1, 10, 20, 50, 100)
kernels = ('linear', 'poly', 'rbf', 'sigmoid')

for percentile in percentiles:
    for ker in kernels:
        model_pipeline.set_params(anova__percentile=percentile, svc__kernel=ker)
        model_pipeline.fit(X_train, y_train)
        print('Percent %s, kernel %s Testing accuracy: %s' % (percentile, ker, round(model_pipeline.score(X_test, y_test),3)))


```


```python 
# 可以知道 pipeline 的可調參數
model_pipeline.get_params().keys()
```


還有其他的特徵選擇器，目前就介紹到這邊，更多內容可以參考 [feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html)。



