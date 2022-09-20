<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# 資料視覺化

這次主要是介紹 [Matplotlib](https://matplotlib.org) 這個 Python 最常被使用到的繪圖套件。 <br>

當我們拿到資料，第一步一定會先看一下資料的狀態，我們會看資料的品質 **Data Quality** ，
像是有沒有缺失資料，資料差異有沒有非常大，以及分佈狀況阿等等，
之後才會決定採用何種分析方法。





```python 
import pandas as pd
sample = pd.read_csv('../../files/sample_corr.csv')

sample

```


## 視覺化 sample data

我們會使用 [Matplotlib](https://matplotlib.org) ，他是 python 非常好用的一個視覺化套件。




```python 
import matplotlib.pyplot as plt
%matplotlib inline
# 設定圖片大小
plt.rcParams['figure.figsize'] = [10, 10]
# plt.rcParams['axes.facecolor'] = 'w'


```


```python 
# 散點圖

sample.plot.scatter(x='x', y='y')

```


## 我們可以從資料視覺化的圖得到什麼?
- 快速吸收信息
- 提高洞察力
- 做出更快的決定



```python 
data = pd.read_csv('../../files/sample_height.csv')
data.head()
```


```python 
# 檢查有沒有缺失資料
data.isna().any()

```


這邊提一下 我們的配色可以[參考](https://matplotlib.org/stable/tutorials/colors/colormaps.html)。


```python 
# 直方圖 histogram
data.plot.hist()
#data.plot.hist(cmap="coolwarm")

```


```python 
# 我們可以 filter 資料

data[data['height'] < 50]

```


```python 
# 更多使用範例
import pandas as pd
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)

X_df = pd.DataFrame(X)
X_df.describe()

```


```python 

#X_df.plot.hist()
X_df.plot.hist(cmap="coolwarm")

```


```python 
X_df[0].plot()
```


## 加 title 跟 labels

- ```title='Tilte'``` 加 title
- ```xlabel='X label'``` 加 X 軸 label
- ```ylabel='X label'``` 加 Y 軸 label


```python 

X_df[0].plot(title='Iris Data', ylabel='sepal length (cm)', xlabel='samples')

```


## 加軸的區間
- ```xlim=(min, max)``` or ```xlim=min``` : 設定 x 軸 的區間
- ```ylim=(min, max)``` or ```ylim=min``` : 設定 y 軸 的區間
  


```python 
X_df[0].plot(title='Iris Data', ylabel='sepal length (cm)', xlabel='samples', ylim=0)
```


如果一次要顯示多筆資料，去比較


```python 
X_df[[0,1]].plot(ylim=0)
```


## 可以用 figsize 改變圖片大小
- ```figsize=(width, height)```


```python 
X_df[[0,1]].plot(ylim=0, figsize=(20,5))
```


## Bar plot 長條圖
- ```.plot.bar()```


```python 
X_df[0].plot.bar(figsize=(20,5))
```


```python 
X_df[[0,1]].plot.bar(figsize=(20,5))
```


### Pie chart 圓餅圖
- ```.plot.pie()```


```python 
from filecmp import cmp

data = pd.Series(data=[3, 5, 7], index=['Data1', 'Data2', 'Data3'])
data.plot.pie(cmap='autumn')

```


```python 
(X_df[0] < 7).value_counts().plot.pie(colors=['b', 'g'], labels=['>=7', '<7'], title='Iris per capita', autopct='%1.1f%%')
```


# subplot 與 subplots

下面要教怎麼一次畫多張圖
## subplot(行,列,位置)


```python 
import matplotlib.pyplot as plt
x_1 = [1,2,3,4,5]
x_2 = [5,4,3,2,1]
x_3 = [1,3,5,3,1]
x_4 = [5,3,1,3,5]

plt.figure(figsize=(10, 10))

plt.subplot(221)
plt.plot(x_1)
plt.subplot(222)
plt.plot(x_2)
plt.subplot(223)
plt.plot(x_3)
plt.subplot(224)
plt.plot(x_4)
plt.show()


```


```python 
import matplotlib.pyplot as plt
x_1 = [1,2,3,4,5]
x_2 = [5,4,3,2,1]
x_3 = [1,3,5,3,1]
x_4 = [5,3,1,3,5]

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(2,2)

ax[0][0].plot(x_1)
ax[0][1].plot(x_2)
ax[1][0].plot(x_3)
ax[1][1].plot(x_4)

plt.show()
```
