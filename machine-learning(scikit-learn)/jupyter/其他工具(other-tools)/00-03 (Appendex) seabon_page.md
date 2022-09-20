<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# seabon

seabon 是以 matplotlib 為底層的高階繪圖套件，下面會以 Iris 鳶尾花資料 為範例，
來看看 seaborn 是怎麼用的。



```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# 讀取資料
X, y = datasets.load_iris(return_X_y=True)

df_iris = pd.DataFrame(data= np.c_[X, y], columns= ['Sepal Length','Sepal Width','Petal Length','Petal Width','Species'])
df_iris


```


```python 
# 這邊提一下 c_, r_ 是串接 矩陣
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.r_[a,b])
print(np.c_[a,b])

```


下面展示如何用 seabon 畫多個直方圖， bins 是要分幾箱。


```python 

df_data.hist(alpha=0.6, layout=(3,3), figsize=(15, 10), bins=10) 
plt.tight_layout()
plt.show()

```


下面用 seabon 跟 subplots 展示 多個子圖，
還有 kde （Kernel Density Estimation） 是核密度估計屬於非參數檢驗方法。



```python 
fig, axes = plt.subplots(nrows=1, ncols=4)
fig.set_size_inches(20, 5)
sns.histplot(df_iris["Sepal Length"][:],ax=axes[0], kde=True)
sns.histplot(df_iris["Sepal Width"][:] ,ax=axes[1], kde=True)
sns.histplot(df_iris["Petal Length"][:],ax=axes[2], kde=True)
sns.histplot(df_iris["Petal Width"][:] ,ax=axes[3], kde=True)

```


我們也可以看兩個特徵之間的關聯，可以用 height 跟 aspect 調大小。


```python 

sns.lmplot("Sepal Length", "Sepal Width", hue='Species', data=df_iris, fit_reg=False, legend=False, height=10, aspect=1)
plt.legend(title='Species', loc='upper right', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])

```


也可以直接看兩兩的關聯，中間對角線是用 kde，也可以用hist，[詳細參考](https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html)。



```python 
from pandas.plotting import scatter_matrix
#scatter_matrix( df_iris, figsize=(20, 20), color='b', diagonal='kde')
scatter_matrix( df_iris, figsize=(20, 20), color='b', diagonal='hist')


```


如果要用 seabon，[詳細參數參考](https://seaborn.pydata.org/generated/seaborn.pairplot.html) 。
- 畫圖的參數 ```kind : scatter, kde, hist, reg``` 
- 對角線畫圖參數 ```diag_kind : auto, hist, kde, None``` 



```python 
sns.pairplot(df_iris, hue="Species", height=2, kind="reg", diag_kind="kde")
sns.pairplot(df_iris, hue="Species", height=2, kind="scatter", diag_kind="hist")
```


下面展示如何看 correlation 關聯度，用 seabon 的 heatmap，[詳細參考](https://seaborn.pydata.org/generated/seaborn.heatmap.html)，如果要改顏色配置，seabon的底層是用matplotlib，所以顏色選擇可以[參考](https://matplotlib.org/stable/tutorials/colors/colormaps.html)。


```python 
# correlation 計算
correlation = df_iris[['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']].corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, square=True, annot=True, cmap="coolwarm")


```
