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
