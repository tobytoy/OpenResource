

# 機率校準

問你一個問題，如果給你一個二分類問題，假設是正或負兩類，
然後你的模型對某個 sample 輸出預測為 $0.8$ ，
你會認為他有機會是正類是 $80%$ 嗎?


答案是，如果你沒有經過校準，那會是錯誤的結論。

## 動機

如果你面對的只是要做圖片分類等等問題，那你可能只會在意要選什麼閥值就好
但是如果你今天要做的是風險分析，或是給出預測報告，
你就會希望給出的是一個正確的機率。



## Calibration Curve

訓練好一個模型以後，有些模型可以給出機率預測 ``predict_proba``，
但有一些模型並沒有，只提供 ``decision_function``而已，這時你可以使用

$$
\frac{score - \min (score)}{\max (score) - \min (score)}
$$

將它縮放到 $[0, 1]$ 區間，
然後考慮 x 軸放 預測的平均， y 軸放 真實的正例的比例，
這樣我們也就可以知道誤差有多少了。



下面看一個 ``calibration_curve`` 的使用範例。



```python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

np.random.seed(5087)
X, y = datasets.make_classification(n_samples=100000, 
                                    n_features=20,
                                    n_informative=2, 
                                    n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train, X_test = X[:train_samples], X[train_samples:]
y_train, y_test = y[:train_samples], y[train_samples:]


# Create classifiers
classifier_lr  = LogisticRegression().fit(X_train, y_train)
classifier_gnb = GaussianNB().fit(X_train, y_train)
classifier_svc = LinearSVC(C=1.0).fit(X_train, y_train)
classifier_rfc = RandomForestClassifier().fit(X_train, y_train)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(8, 8))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for classifier, name in [(classifier_lr, 'Logistic'),
                         (classifier_gnb, 'Naive Bayes'),
                         (classifier_svc, 'Support Vector Classification'),
                         (classifier_rfc, 'Random Forest')]:
    
    if hasattr(classifier, "predict_proba"):
        prob_pos = classifier.predict_proba(X_test)[:, 1]
    else:  # use decision function
        decision_score = classifier.decision_function(X_test)
        prob_pos = (decision_score - decision_score.min()) / (decision_score.max() - decision_score.min())
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()

```


如果你還是不太清楚，下面去到[程式](https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/calibration.py#L873)裡面來看。


![calibration curve](../../images/scikit-learn_calibration.jpg)


下面給一個範例



```python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

np.random.seed(5087)
train_samples = 100
X, y = datasets.make_classification(n_samples=(train_samples*2), 
                                    n_features=20,
                                    n_informative=2, 
                                    n_redundant=2)

X_train, X_test = X[:train_samples], X[train_samples:]
y_train, y_test = y[:train_samples], y[train_samples:]
prob_pos        = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]


bins = np.linspace(0.0, 1.0, 10 + 1)     
print('bins: ', bins)               # 要分成幾份
binids = np.searchsorted(bins[1:-1], prob_pos)     
print('binids: ', binids)           # 把機率分到該去的那個 bins
bin_sums = np.bincount(binids, weights=prob_pos, minlength=len(bins))  
print('bin_sums: ', bin_sums)       # 每個 bin 區間的預測資料合
bin_true = np.bincount(binids, weights=y_test, minlength=len(bins))     
print('bin_true: ', bin_true)       # 每個 bin 區間的真實 positive 的數量
bin_total = np.bincount(binids, minlength=len(bins))        # 每個 bin 裡面有多少 data

nonzero = bin_total != 0
prob_true = bin_true[nonzero] / bin_total[nonzero]
print('prob_true: ', prob_true)       # 每個 bin 區間的真實 positive 的機率
prob_pred = bin_sums[nonzero] / bin_total[nonzero]
print('prob_pred: ', prob_pred)       # 每個 bin 區間的預測資料平均


```
