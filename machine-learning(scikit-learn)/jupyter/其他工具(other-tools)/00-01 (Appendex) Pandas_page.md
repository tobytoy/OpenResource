<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# Pandas

Pandas 是 python 的 library 用來處理資料分析，如果現在你對 pandas 一無所知，你可以把 pandas 想成 Excel。 <br>

在開始前，需要提一些 pandas 的好用資源。
* 有提供很好的 [Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) 。
* 也有很好的教學 [tutorials](https://pandas.pydata.org/docs/getting_started/index.html) 。
* 也可以參考 [Kaggle tutorials](https://www.kaggle.com/learn/pandas) 。




第一步，我們先看 pandas 怎的讀取 csv。

- ```pd.read_csv(filename, parse_dates, index_col)``` ([docs](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html))
    - ```filename```: 檔案位置
    - ```parse_dates=True```: 如果是 True -> 試著去解析文件 (預設是 False，因為可以加快文件載入速度) [更多參考](http://hshsh.me/post/2016-04-12-python-pandas-notes-01/)
    - ```index_col=0```: 設定 index 為第 0 列

下面簡單介紹一下 CSV。

### CSV(Comma-Separated Values)，逗號分隔值
例如
* cat,1,2,3
* dog,2,3,4




```python 
import pandas as pd

data = pd.read_csv('../../files/aapl.csv', parse_dates=False, index_col=0)
data.head()

```


```python 
# 資料的 shape 與 len
print('資料的shape: ', data.shape, ' 資料的 length: ', len(data))
```


## Index and columns
* ```.index```: 回傳 index
* ```.columns```: 回傳每個 列 (column) 的名子。


```python 
data.index
```


```python 
data.columns
```


我們可以用每一列的名子去抓出每一列的資料。


```python 
# data.High    #也可以用這個方法
# data['High']

# 如果要第 10 行的值
data['High'][10]
```


## 每列 column 有他的資料型態 data type
- ```.dtypes```: 回傳每列的 data types


```python 
data.dtypes
```


## 列與行的切片(Slicing rows and columns)
- ```data['Close']```: 取出 'Close' 那一列
- ```data[['Open', 'Close']]```: 取出多列
- ```data.loc['2020-05-01':'2021-05-01']```: 取出 2020-05-01 到 2021-05-01 的那幾行
- ```data.iloc[50:55]```: 取出 50 到 55 的那幾行，如果你不知道 index 的話可以用這個方法。


```python 
data['Close']
```


```python 
data[['Close', 'Open']]
```


```python 
data.loc['2021-05-03':'2021-05-14']
```


```python 
data.iloc[50:55]
```


```python 
# 也可以取出非 null 的值，如果你的資料有缺失值想要略過他，很有幫助。
data.loc[data.Close.notnull()]
```


還有更細的操作可以[參考](https://www.kaggle.com/code/residentmario/indexing-selecting-assigning)。


我們可以做更多的運算。


```python 
data['Close'] - data['Open']
```


```python 
data['New'] = data['Open'] - data['Close']
```


```python 
data.head()
```


我們也可以做資料的選取。



```python 
data['New'] > 0
```


```python 
data[data['New'] > 0]
```


我們也可以叫 pandas 幫我們做結論。


```python 
data.High.mean()
#data.High.unique()
#data.High.value_counts()
#data.High.describe()
```


# 分群(grouping) 與 排序(sorting)



```python 
# 建立一個類別
data['Category'] = data['New'] > 0

```


```python 
data.groupby('Category').High.count()
#data.groupby('Category').mean()
#data['Category'].value_counts()

```


```python 
# 根據哪一列排序
#data.sort_values(by='Low')
# 根據 index 排序
data.sort_index()

```


更多用法可以[參考](https://www.kaggle.com/code/residentmario/grouping-and-sorting) 。


# Missing Values 缺失值

我們可以轉換值的 資料類型



```python 
data.dtypes
```


```python 
data.High.astype('float32')

```


```python 
# dtype('O') 在 pandas 表示 string
data.index.dtype
```


缺失值 NaN 他是 Not a Number 的縮寫， NaN 是 float64 dtype。


```python 
data[pd.isnull(data.High)]
```


```python 
import numpy as np
# 我們自己令一個 NaN

data['High'][0] = np.NaN

data['High'].head

```


```python 
data[pd.isnull(data.High)]
```


```python 
# 我們來補上

data.High.fillna(0).head

# 我們可以補上任何想補上的值
```
