<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# Review
這邊先回顧一下前面討論的線性回歸，其實我們就是用$w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots $去估計 $y$。

# Polynomial
我們來說一下什麼是多項式
$$
w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots
$$
這就是多項式。 <br>
我們先假想一個場景，假設你收集到一些資料 $X$ 想要去預測 $y$，結果很不幸的你的 $X$ 只有一維的資料，我們上次講過那麼多的方法難道就不能用嗎? <br>
多項式就說話啦，我可以來幫你把一維資料生出好多維資料，那你又可以快樂的用之前教的 linear model 線性模型來預測估計 $y$ 啦。 <br>
下面來一個範例 
$$
x = (1,2,3,4,5)
$$
那要怎麼生出其他向量
$$
x^2 = (1,4,9,16,25)
$$
這不就生出來了，並且你想生多少就生多少。 <br>
下面再問一個問題納我們如果 $X$ 本來就不是一維的那你會生出更多資料嗎? <br>
那你是不是馬上又想到一個問題，我生出那麼多特徵，效果會變很好嗎? 計算量變很大? 特徵無限變大該怎麼辦? <br>
特徵的選取(feature-selection)，以後會講到可以移駕 輸入 feature-selection。


# Polynomial Regression

沒錯這次的主題就是多項式回歸，我們要怎麼生出更多的特徵 ``PolynomialFeatures`` 就是要用到他啦，下面進入實戰。




```python 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.array([1, 2, 3, 4])
poly = PolynomialFeatures(degree=4, include_bias=False)
poly.fit_transform(x[:, None])

```


如果現在 $x=(x_1, x_2)$，那生出 degree 不超過 $2$ 的特徵有多少?
$$
1, x_1, x_2, x_1^2, x_1x_2, x_2^2
$$


```python 
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.arange(6).reshape(3, 2)

display(X)

poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)

```


# Linear Polynomial Regression 

接下來我們就根據這次學到的 polynomial features，去做 linear regression。



```python 
# 先準備資料
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# 下載 糖尿病資料
X, y = datasets.load_diabetes(return_X_y=True)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 準備模型
regression = linear_model.LinearRegression()
# 訓練模型
regression.fit(X_train, y_train)
# 預測結果
y_pred = regression.predict(X_test)

#print('w 係數：', regression.coef_)
print('w_0 截距：', regression.intercept_)

# The mean squared error 我們以後會介紹 metrics 就會認識 mse，現在先用。
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
```


如果前面係數要求是正的


```python 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


X, y = datasets.load_diabetes(return_X_y=True)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 準備模型 這邊要多加一個參數 positive=True
regression = linear_model.LinearRegression(positive=True)

regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

#print('w 係數：', regression.coef_)
print('w_0 截距：', regression.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

```


# Ridge Poly Regression


```python 
# Ridge Regression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


X, y = datasets.load_diabetes(return_X_y=True)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 準備 Ridge 模型 
regression = linear_model.Ridge(alpha=0.5)

regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

#print('w 係數：', regression.coef_)
print('w_0 截距：', regression.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
```


# Ridge Poly Classification


```python 
# Ridge Classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


X, y = datasets.load_iris(return_X_y=True)
#y = LabelBinarizer().fit_transform(y)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87) 

classifier = RidgeClassifier().fit(X_train, y_train)

# The Score will Return the mean accuracy on the given test data and labels.
print('Training accuracy: ', classifier.score(X_train, y_train))
print('Testing accuracy: ', classifier.score(X_test, y_test))
```


# Lasso Poly Regression


```python 
# Lasso Poly Regression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


X, y = datasets.load_diabetes(return_X_y=True)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 準備 Lasso 模型
regression = linear_model.Lasso(alpha=0.1)

regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

#print('w 係數：', regression.coef_)
print('w_0 截距：', regression.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

```


# Elastic-Net Poly Regression


```python 
# Elastic-Net Poly Regression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


X, y = datasets.load_diabetes(return_X_y=True)

# poly feature
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)

# 準備 Elastic-Net 模型
regression = linear_model.ElasticNet(alpha=0.1)

regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

#print('w 係數：', regression.coef_)
print('w_0 截距：', regression.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
```
