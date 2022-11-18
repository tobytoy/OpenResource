

# Decision Trees 決策樹

決策樹 是 非參數的監督學習，可以用來做分類與回歸，我們前面已經教過他的概念，下面會詳細說明如何用 Decision Trees 做 分類 （classification） 與回歸 （regression）。




## Decision Tree 分類


```python 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 


classifier = DecisionTreeClassifier(criterion='gini', max_depth=3).fit(X_train, y_train)

print('Training accuracy: ', classifier.score(X_train, y_train))
print('Testing accuracy: ', classifier.score(X_test, y_test))


```


```python 
??DecisionTreeClassifier
```


## Decision Tree 回歸



```python 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 


#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.fit_transform(X_test)

regression = DecisionTreeRegressor(criterion='squared_error', 
                                   min_samples_split=40, 
                                   min_samples_leaf=5).fit(X_train, y_train)


y_pred = regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))


```


```python 
??DecisionTreeRegressor
```


# 隨機森林樹 （Random Forest）

簡單的說，隨機森林樹就是決策樹加上 bagging
假設我們有訓練資料 $X, y$，下面來說說隨機森林樹的實現步驟。

* 我們先隨機抽取訓練資料 $n$ 筆，假設第 $i$ 次抽取出的資料叫 $X_i, y_i$ ，我們允許重複抽取。
* 拿每筆資料 $X_i$，隨機抽取 $m$ 個特徵 然後另取一個名子 $\bar{X}_i$。
* 拿每筆資料 $\bar{X}_i, y_i$，去訓練一個決策樹模型。
* 如果做分類問題時，把所有的模型做投票取最終結果，如果做回歸問題，把所有模型的輸出取平均

[參考資料](https://zhuanlan.zhihu.com/p/86263786)



# 極端隨機樹 （Extremely Randomized Trees）

極端樹可以看做隨機森林樹的變種，可以簡單的理解為更隨機的取樣，但這是計算量更大的隨機森林樹。

假設我們有訓練資料 $X, y$，下面來說說隨機森林樹的實現步奏。

* 拿取全部資料 $X$ （隨機樹會在這邊做一次隨機抽取）， 隨機抽取 $m$ 個特徵。
* 對於這 $m$ 個特徵，每個特徵選擇隨機分裂，再看這 $m$ 個分裂哪個分數最高。（跟隨機樹相比，分裂節點更隨機。）


[參考資料](https://zhuanlan.zhihu.com/p/380323376)




## 下面開始做分類



```python 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

classifier_decision_tree = DecisionTreeClassifier(criterion='gini', 
                                                  max_depth=3, 
                                                  min_samples_split=2).fit(X_train, y_train)

print('Decision Tree Test accuracy: ', classifier_decision_tree.score(X_test, y_test))



classifier_random_forest = RandomForestClassifier(n_estimators=10, 
                                                  max_depth=3,
                                                  min_samples_split=2).fit(X_train, y_train)
print('Random Forest Test accuracy: ', classifier_random_forest.score(X_test, y_test))

classifier_extra_tree = ExtraTreesClassifier(n_estimators=10, 
                                             max_depth=3,
                                             min_samples_split=2).fit(X_train, y_train)
print('ExtraTrees Test accuracy: ', classifier_extra_tree.score(X_test, y_test))


```


## 下面開始做回歸



```python 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

regression_decision_tree = DecisionTreeRegressor(criterion='squared_error', 
                                                  min_samples_split=20).fit(X_train, y_train)

y_pred = regression_decision_tree.predict(X_test)

print("Decision Tree Test Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))

regression_random_forest = RandomForestRegressor(criterion='squared_error', 
                                                min_samples_split=20).fit(X_train, y_train)

y_pred = regression_random_forest.predict(X_test)

print("Random Forest Test Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))

regression_extra_tree = ExtraTreesRegressor(criterion='squared_error',  
                                            min_samples_split=20).fit(X_train, y_train)
y_pred = regression_extra_tree.predict(X_test)

print("ExtraTrees Test Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))

```


```python 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 


#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.fit_transform(X_test)

regression = DecisionTreeRegressor(criterion='squared_error', 
                                   min_samples_split=40, 
                                   min_samples_leaf=5).fit(X_train, y_train)


y_pred = regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))

```


# Random Forest 與 Extremely Randomized Trees 的參數選擇

下面要來碎碎念參數的選擇，
主要的調整參數是 **n_estimators** 和 **max_features**，
第一個參數 **n_estimators** 是要取多少樹的數量，通常越多效果越好，但是計算量也就越大，
第二個參數 **max_features** 是要隨機取多少特徵，通常這個參數減小，Variance 減小，Bias 增大。

默認值參數建議，**max_features=sqrt(n_features)**，**max_depth=None** 和 **min_samples_split=2**。

另外在 Random Forest 默認自己隨機採樣 （bootstrap = True），
Extremely Randomized Trees 的默認為使用整個數據 （bootstrap = False）。






# XGBoost

[官網文件](https://xgboost.readthedocs.io/en/stable/)

下面介紹 XGBoost， XGBoost 是 eXtreme Gradient Boosting 的簡寫，
對於計算速度與模型的表現都有特別優化。




```python 
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

# create model instance
classifier = XGBClassifier(n_estimators=2, 
                            max_depth=2, 
                            learning_rate=1, 
                            objective='binary:logistic').fit(X_train, y_train)

print('Test accuracy: ', classifier.score(X_test, y_test))


```


```python 
from xgboost import XGBRegressor
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 


#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.fit_transform(X_test)

regression = XGBRegressor(n_estimators=2, 
                            max_depth=2, 
                            learning_rate=1).fit(X_train, y_train)

y_pred = regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))


```


## LightGBM

[官網說明](https://lightgbm.readthedocs.io/en/v3.3.2/)

LightGBM 跟 XGBoost 很相似， LightGBM 是微軟做的產品，他比 XGBoost 更節省記憶體，速度也更快。




```python 
# 分類
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

classifier = LGBMClassifier().fit(X_train, y_train)

print('Training accuracy: ', classifier.score(X_train, y_train))
print('Testing accuracy: ', classifier.score(X_test, y_test))

```


```python 
??LGBMClassifier

```


```python 
# 回歸
from lightgbm import LGBMRegressor
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.fit_transform(X_test)

regression = LGBMRegressor().fit(X_train, y_train)

y_pred = regression.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred))

```


```python 
??LGBMRegressor

```
