<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



在 scikit-learn 裡面的任務類型可以分為 $3$ 類，Classification, Regression, Clustering，所以指標也會分為三個部分來說明。


下面列出在 scikit learn 提供的方法。


| Regression 回歸 | Classification 分類 | Clustering 分群 |
| :----: | :----: | :----: |
| explained_variance_score | accuracy_score | adjusted_mutual_info_score |
| max_error | balanced_accuracy_score | adjusted_rand_score |
| mean_absolute_error | top_k_accuracy_score | completeness_score |
| mean_squared_error | average_precision_score | fowlkes_mallows_score |
| mean_squared_log_error | brier_score_loss | homogeneity_score |
| median_absolute_error | f1_score | mutual_info_score |
| r2_score | log_loss | normalized_mutual_info_score |
| mean_poisson_deviance | precision_score | rand_score |
| mean_gamma_deviance | recall_score | v_measure_score |
| mean_absolute_percentage_error | jaccard_score | |
| d2_absolute_error_score | roc_auc_score | | 
| d2_pinball_score | | |
| d2_tweedie_score | | |



最後還會提及一種特別的 Dummy 評估器，可以用來做 baseline 的值，我們廢話不多說直接進入 Dummy 的實戰環節。




```python 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

# 建立不平衡資料
y[y != 1] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=87)

classifier = SVC(kernel='linear', C=1).fit(X_train, y_train)
display(classifier.score(X_test, y_test))

# classifier_dummy = DummyClassifier(strategy='most_frequent', random_state=87)
classifier_dummy = DummyClassifier(strategy='constant', constant=1, random_state=87)
classifier_dummy.fit(X_train, y_train)

classifier_dummy.score(X_test, y_test)

```


# Dummy Classifier


可以選用的參數有:
- stratified : 根據訓練集類別的分佈，產生隨機預測值。
- most_frequent : 預測訓練集中頻率最高的標籤。
- prior : 預測可以使類別最大化的類。
- uniform : 產生均勻隨機的預測值。
- constant : 預測某一類。


# Dummy Regressor

可以選用的參數有:
- mean : 預測的是平均值。
- median : 預測的是中間值。
- quantile : 預測的是分位數。
- constant : 預測某一固定值。


