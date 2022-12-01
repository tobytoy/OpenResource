<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# GridSearchCV

下面來舉一個簡單的使用範例



```python 
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


X, y = datasets.load_digits(return_X_y=True)
n_samples = len(X)
X = X.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=87)

# 設置搜尋參數
tuned_parameters = [
    {'kernel': ['rbf'], 
     'gamma': [1e-3, 1e-4, 'scale'],
     'C': [1, 10, 100, 1000],
     },
    {'kernel': ['linear'], 
     'C': [1, 10, 100, 1000],
     },
    {'kernel': ['poly'],
     'gamma': ['scale', 'auto'],
     'degree': [2, 3, 4, 5, 6],
     'C': [1, 10, 100, 1000],
     },
]

scores = ['precision_macro', 'recall_micro', 'f1_weighted', 
          'accuracy', 'balanced_accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s \n" % score)

    classifier = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s' % score
    ).fit(X_train, y_train)

    print("Best parameters set found on development set: \n")
    print(classifier.best_params_)
    print("\nGrid scores on development set:\n")
    
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    print("\nDetailed classification report:\n")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    
```


## 可以查詢有哪些是可以使用的 scorer 



```python 
from sklearn import metrics

metrics.get_scorer_names()

```


# XGboost with GridSearchCV

* n_estimators：樹的數量
* eta [defalt = 0.3, 別名 : leanring_rate]： 學習步長
* max_depth [defalt = 6]：樹的最大深度，越大越容易 overfitting
* min_child_weight [defalt = 1]：最小葉子節點的權重合。當他的值大可以避免學到局部樣本。如果太大會發生 overfitting
* lambda [defalt = 1, 別名 : reg_alpha]： L2 正則化權重，越大必免 overfitting
* alpha [defalt = 0, 別名 : reg_alpha]： L1 正則化權重，越大必免 overfitting 
* seed：隨機樹種子

更多參數請[參考官網](https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)

下面給一個 XGboost + Pipeline + GridSearchCV 的使用範例，
有其他需求可以拿他來改。



```python 
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 設置搜尋參數
tuned_parameters = [
    {'classifier__n_estimators': [100, 200, 300], 
     'classifier__max_depth': [1, 2, 3],
     'classifier__min_child_weight': [1, 2, 3],
     },
]

# 設置 Pipeline
estimators = [
    ('reduce_dim', PCA()), 
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier())
]
pipeline = Pipeline(estimators)


scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s \n" % score)

    classifier = GridSearchCV(
        pipeline, 
        tuned_parameters, 
        scoring='%s' % score,
        cv=3
    ).fit(X_train, y_train)

    print("Best parameters set found on development set: \n")
    print(classifier.best_params_)
    print("\nGrid scores on development set:\n")
    
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    print("\nDetailed classification report:\n")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))


```


# LightGBM with GridSearchCV

更多參數請[參考官網](https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html)

下面也是給個使用範例




```python 
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 設置搜尋參數
tuned_parameters = [
    {'classifier__n_estimators': [100, 200, 300], 
     'classifier__max_depth': [1, 2, 3],
     'classifier__min_child_weight': [1, 2, 3],
     },
]

# 設置 Pipeline
estimators = [
    ('reduce_dim', PCA()), 
    ('scaler', StandardScaler()),
    ('classifier', LGBMClassifier())
]
pipeline = Pipeline(estimators)


scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s \n" % score)

    classifier = GridSearchCV(
        pipeline, 
        tuned_parameters, 
        scoring='%s' % score,
        cv=3
    ).fit(X_train, y_train)

    print("Best parameters set found on development set: \n")
    print(classifier.best_params_)
    print("\nGrid scores on development set:\n")
    
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    print("\nDetailed classification report:\n")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    
```
