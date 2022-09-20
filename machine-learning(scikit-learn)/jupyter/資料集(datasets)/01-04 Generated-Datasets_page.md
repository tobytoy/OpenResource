<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



當我們需要更大量的資料來做測試的時候，我們可以使用 scikit learn 提供的自動生成數據。

# Generated datasets 生成的數據集

- make_blobs
- make_classification
- make_gaussian_quantiles
- make_biclusters
- make_blobs
- make_checkerboard
- make_circles
- make_classification
- make_friedman1
- make_friedman2
- make_friedman3
- make_gaussian_quantiles
- make_hastie_10_2
- make_low_rank_matrix
- make_moons
- make_multilabel_classification
- make_regression
- make_s_curve
- make_sparse_coded_signal
- make_sparse_spd_matrix
- make_sparse_uncorrelated
- make_spd_matrix
- make_swiss_roll


因為使用方法很類似我們來看幾個實際使用的範例。




```python 
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, 
                   n_features=7,
                   n_informative=4,
                   n_classes=3)

X.shape, y.shape

```


```python 
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, 
                   n_features=7,
                   n_targets=2,
                   random_state=87)

X.shape, y.shape
```


```python 
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, 
                    n_features=4, 
                    n_targets=2,
                    random_state=87)

X.shape, y.shape

```


```python 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, 
                centers=3, 
                n_features=2,
                random_state=87)

X.shape, y.shape

```
