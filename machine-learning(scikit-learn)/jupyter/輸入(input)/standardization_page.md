<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



在前處理(preprocessing)的問題也包含缺失值處理(Missing Values)，我們把這個問題，移到 "其他工具" 的 pandas 的教學裡面。




# Standardization
在scikit-learn 裡面提供了四種不同資料處理的方法

* StandardScaler (平均值和標準差)
* MinMaxScaler (最小最大值標準化)
* MaxAbsScaler (絕對值最大標準化)
* RobustScaler

## StandardScaler
將所有特徵標準化，減去 mean 再除以 std，使得數據的 mean 為 $0$，std 為 $1$。

## MinMaxScaler
數據會縮放到到 $[0,1]$ 之間。

$$
X_{scaled} = \frac{X - X.min}{X.max - X.min}
$$

默認是對每一列都去做縮放。

## MaxAbsScaler
數據會縮放到到 $[-1,1]$ 之間，方法類似 MinMaxScaler。


## RobustScaler
可以有效的縮放帶有outlier的數據，透過Robust如果數據中含有異常值在縮放中會捨去。



下面是使用範例




```python 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


X, y = datasets.load_iris(return_X_y=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scaled之後的資料零均值，單位方差  
print('資料集 X 的平均值 : ', X.mean(axis=0))
print('資料集 X 的標準差 : ', X.std(axis=0))

print('\nStandardScaler 縮放過後的平均值 : ', X_scaled.mean(axis=0))
print('StandardScaler 縮放過後的標準差 : ', X_scaled.std(axis=0))



```


```python 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# scaled 之後的資料最小值、最大值  
print('資料集 X 的最小值 : ', X.min(axis=0))
print('資料集 X 的最大值 : ', X.max(axis=0))

print('\nStandardScaler 縮放過後的最小值 : ', X_scaled.min(axis=0))
print('StandardScaler 縮放過後的最大值 : ', X_scaled.max(axis=0))


```


```python 
from sklearn.preprocessing import MaxAbsScaler

X, y = datasets.load_iris(return_X_y=True)
# X = X-3
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)


# scaled 之後的資料最小值、最大值  
print('資料集 X 的最小值 : ', X.min(axis=0))
print('資料集 X 的最大值 : ', X.max(axis=0))

print('\nStandardScaler 縮放過後的最小值 : ', X_scaled.min(axis=0))
print('StandardScaler 縮放過後的最大值 : ', X_scaled.max(axis=0))


```


```python 
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


# scaled 之後的資料最小值、最大值  
print('資料集 X 的最小值 : ', X.min(axis=0))
print('資料集 X 的最大值 : ', X.max(axis=0))

print('\nStandardScaler 縮放過後的最小值 : ', X_scaled.min(axis=0))
print('StandardScaler 縮放過後的最大值 : ', X_scaled.max(axis=0))


```
