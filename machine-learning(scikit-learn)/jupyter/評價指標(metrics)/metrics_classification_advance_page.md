<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>




# 靈魂拷問


* 我們來看一個模型的預測，他的預測是直接就是答案嗎?
* 如果不是答案是機率，例如預測一張圖片是貓的機率，那多高算貓貓，$0.5$ ?
* 如果是機率那之前提的準確率、精確率、召回率怎麼算?




```python 
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=87)

classifier = SVC(probability=True).fit(X_train, y_train)

# 直接給信心最高的答案
y_pred = classifier.predict(X_test)
print('Predict: ', y_pred)

# 給所有類別的機率
y_pred_proba = classifier.predict_proba(X_test)
print('Predict Probability: ', y_pred_proba)

# 給所有類別的信心
# The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.
y_pred_decision = classifier.decision_function(X_test)
print('Confidence Score: ', y_pred_decision)

```


# Review

|  真實 \  預測 | **有病** （<font color=Blue>positive</font>） | **沒病** （negative） |
| :-----: | :----: | :----: |
| **有病** | TP （true <font color=Blue>positive</font>） <br> 判斷是對的 有病 | FN （<font color=Red>false</font> negative） <br> 判斷是錯的 沒病 |
| **沒病** | FP （<font color=Red>false</font> <font color=Blue>positive</font>） <br> 判斷是錯的 有病 | TN （true negative） <br> 判斷是對的 沒有病 |



### FP （false positive） : 偽陽，型一錯誤
### FN （false negative） : 偽陰，型二錯誤


## Accuracy 準確率

$$
Accuracy = \frac{TP+TN}{TP+FN+FP+TN}
$$

## Precision 精確率

$$
Precision = \frac{TP}{TP + FP}
$$


## Recall 召回率

$$
Recall = \frac{TP}{TP+FN}
$$


# Precision-Recall Curve

我們可以再活用之前的知識把 Precision 跟 Recall 依據不同的閾值 畫在圖上。

* x-軸 : Recall
* y-軸 : Precision

線越靠右越好

## AP （Average precision）

$$
AP = \sum_{i} (R_i - R_{i-1}) P_i
$$

可以理解為 Precision 的平均，或是曲線下的面積。

## mAP （mean Average precision）

如果分類的任務不只兩類是多類，就把每一類的 AP 取平均。


### 下面給 Precision-Recall Curve 使用範例

會用下面兩個方法演示

* from_estimator
* from_predictions




```python 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
%matplotlib inline

plt.rcParams['figure.figsize'] = [20, 10]

fig = plt.figure()
ax_11 = fig.add_subplot(1, 2, 1)
ax_12 = fig.add_subplot(1, 2, 2)

X, y = load_iris(return_X_y=True)

# 加雜訊
random_state = np.random.RandomState(87)
n_samples, n_features = X.shape
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# 只用前兩類 跟 切分資料
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2], test_size=0.5, random_state=random_state)


classifier = LinearSVC(random_state=random_state).fit(X_train, y_train)


# 如果用 from_estimator 要把模型也傳進去
display_estimator = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test, name="LinearSVC", ax=ax_11
)
display_estimator.ax_.set_title("Precision-Recall Curve from Estimator")

# 如果預測出結果可以用 from_predictions
y_score = classifier.decision_function(X_test)

display_predictions = PrecisionRecallDisplay.from_predictions(
    y_test, y_score, name="LinearSVC", ax=ax_12
    )
display_predictions.ax_.set_title("Precision-Recall Curve from Predictions")


```


# ROC （Receiver operating characteristic）

我們來看看 ROC 曲線的 x 軸 

## FPR （False Positive Rate） 偽陽性率

$$
FPR = \frac{FP}{FP+TN}
$$

我們來看看 ROC 曲線的 y 軸

## TPR （True Positive Rate） 真陽性率

$$
TPR = \frac{TP}{TP+FN}
$$

ROC 曲線越靠左上越好。

## AUC（Area Under Curve）

AUC 代表在 ROC 曲線下的面積

* $AUC=1$ 代表非常完美
* $0.5<AUC<1$ 代表模型有學到東西可以調閾值
* $AUC=0.5$ 跟猜得一樣
* $AUC<0.5$ 比猜的還差


下面看實例



```python 
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
plt.rcParams['figure.figsize'] = [10, 10]

X, y = datasets.load_iris(return_X_y=True)

#  把數字轉 one hot
#  0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1] 
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# 加入雜訊使問題變難
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)

# 使用 SVC 加上 多輸出
classifier = OneVsRestClassifier(
                SVC(kernel="linear", 
                probability=True, 
                random_state=random_state)
            )
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# 對於每一類計算 ROC curve 跟 ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 計算 micro-average ROC curve 跟 ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# setosa, versicolor, virginica
index = 2
name_list = ['setosa', 'versicolor', 'virginica']
plt.figure()
lw = 2
plt.plot(
    fpr[index],
    tpr[index],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[index],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("For Iris %s Labels" % name_list[index])
plt.legend(loc="lower right")
plt.show()



```


```python 
# 一次畫多筆資料
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# 每類算平均
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

# 算 macro
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 畫每一類
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()
```


這邊多提一下 Precision-Recall Curve 只關心到 TP 沒考慮到 TN，
而 ROC Curve 有考慮到 TN。

