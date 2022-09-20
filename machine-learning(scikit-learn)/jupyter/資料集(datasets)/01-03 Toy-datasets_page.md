<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# 簡介
我們先來看看 Scikit learn 提供哪些 "玩具資料集"，
為什麼稱為玩具資料，是因為實際處理的資料格式不會這麼統一，可能會有缺失資料。
在開始訓練模型之前有一份工作叫做 data cleaning 資料清洗，要把真實收到的資料格式統一化，
再把缺失資料補上等等前置操作。
而玩具資料的特點一是資料量不多，特點二是格式都已經統一也沒有什麼缺失資料，
非常適合用來練習機器學習。



1. Boston house prices dataset. 波士頓房價數據集 <br>
  Number of Instances: <font color=#0000FF>$506$</font>, 
  Number of Attributes: 13
2. Iris plants dataset. 鳶尾植物數據集 <br>
  Number of Instances: <font color=#0000FF>$50$</font>, 
  Number of Attributes: 4
3. Diabetes dataset. 糖尿病數據集 <br>
  Number of Instances: <font color=#0000FF>$442$</font>, 
  Number of Attributes: 10
4. Optical recognition of handwritten digits dataset. 
  手寫數字數據集的光學識別 <br>
  Number of Instances: <font color=#0000FF>$1797$</font>, 
  Number of Attributes: 64
5. Linnerrud dataset. Linnerrud 數據集 <br>
  Number of Instances: <font color=#0000FF>$20$</font>, 
  Number of Attributes: 3
6. Wine recognition dataset. 葡萄酒識別數據集 <br>
  Number of Instances: <font color=#0000FF>$178$</font>, 
  Number of Attributes: 13
7. Breast cancer wisconsin (diagnostic) dataset. wisconsin 
  州乳腺癌（診斷）數據集 <br>
  Number of Instances: <font color=#0000FF>$569$</font>, 
  Number of Attributes: 30


下面給個簡單分類


| ```回歸 regression``` | ```分類 classification``` |
| :----: | :----: |
| 波士頓房價數據集 | 鳶尾植物數據集 |
| 糖尿病數據集 | 手寫數字數據集 |
| Linnerrud 數據集 (多輸出) | 葡萄酒識別數據集 |
|  | 乳腺癌（診斷）數據集 |


之後會詳細介紹這些數據集。我們先來看 Scikit learn package 裡面。



```python 
from sklearn import datasets

index = 0
for item in dir(datasets):
  if item.split('_')[0] == 'load':
    index += 1
    print(index, '\t', item)

```


# Boston house prices dataset. 波士頓房價數據集

* Number of Instances : <font color=#0000FF>$506$</font> <br>
  這個資料集有 $506$ 筆資料。
* Number of Attributes : 13 <br>
  每筆資料有 $13$ 個特徵。
* Category : regression

## Attributes 特徵分別是

- CRIM : 按城鎮劃分的人均犯罪率
- ZN : 比例超過25,000平方英尺的住宅用地比例。
- INDUS : 每個城鎮非零售業務英畝的比例
- CHAS : 查爾斯河虛擬變量（如果邊界是河流，則為1；否則為0）
- NOX : 一氧化氮濃度（百萬分之幾）
- RM : 每個住宅的平均房間數
- AGE : 1940年之前建造的自有住房的年齡比例
- DIS : 與五個波士頓就業中心的加權距離
- RAD : 高速公路通行能力指数
- TAX : 每10,000美元的稅全額財產稅稅率
- PTRATIO : 按鎮劃分的師生比率
- B : 1000（Bk-0.63）^ 2 其中Bk是按城鎮劃分的黑人比例
- LSTAT : 人口地位降低百分比
- MEDV : 自有住房的價值中位數（以1000美元計）




```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

# 忽略警告訊息
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X, y = datasets.load_boston(return_X_y=True)
    print('The shape of X: ', X.shape, '\n')
    print('The shape of y: ', y.shape)

  
```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

# 忽略警告訊息
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    
    data = datasets.load_boston()

    print('The data features:')
    display(data.feature_names)

    X = data.data
    print('The shape of X: ', X.shape, '\n')

    y = data.target
    print('The shape of y: ', y.shape)

```


# Iris plants dataset. 鳶尾植物數據集

* Number of Instances: <font color=#0000FF>$50$</font> <br>
  這個資料集有 $50$ 筆資料。
* Number of Attributes: 4 <br>
  每筆資料有 $4$ 個特徵。
* Category : classification

## Attributes 特徵分別是
我們來英文翻譯，花萼（sepal）和花瓣（petal），長度（length）和寬度（width）。

1. sepal length in cm， 花萼 的 長度。
2. sepal width in cm， 花萼 的 寬度。
3. petal length in cm， 花瓣 的 長度。
4. petal width in cm， 花瓣 的 寬度。


## class
要分類的是鳶尾花屬下的三個亞屬，分別是山鳶尾（setosa）、變色鳶尾（verscicolor）和維吉尼亞鳶尾（virginica）。

1. Setosa， 山鳶尾
2. Versicolour， 變色鳶尾
3. Virginica， 維吉尼亞鳶尾




```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)

```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_iris()

print('The data features:')
display(data.feature_names)

X = data.data
print('The shape of X: ', X.shape, '\n')

print('The class names:')
display(data.target_names)

y = data.target
print('The shape of y: ', y.shape)
```


# Diabetes dataset. 糖尿病數據集

* Number of Instances: <font color=#0000FF>$442$</font> <br>
  這個資料集有 $442$ 筆資料。
* Number of Attributes: 10 <br>
  每筆資料有 $10$ 個特徵。
* Category : regression


## Attributes 特徵分別是

- age in years : 年紀。
- sex : 性別。
- bmi : body mass index 體重指數。
- bp : average blood pressure 平均血壓。
- s1 tc : T細胞（一種白細胞）
- s2 ldl : 低密度脂蛋白
- s3 hdl : 高密度脂蛋白
- s4 tch : 甲狀腺刺激激素
- s5 ltg : 拉莫三嗪
- s6 glu : blood sugar level 血糖水平




```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)

```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_diabetes()

print('The data features:')
display(data.feature_names)

X = data.data
print('The shape of X: ', X.shape, '\n')

y = data.target
print('The shape of y: ', y.shape)

```


# Optical recognition of handwritten digits dataset. 手寫數字數據集的光學識別

* Number of Instances: <font color=#0000FF>$1797$</font> <br>
  這個資料集有 $1797$ 筆資料。
* Number of Attributes: 64 <br>
  每筆資料有 $64$ 個特徵。
* Category : classification

## Attributes 特徵分別是
範圍為0..16的整數像素的8x8圖像。

## class
10 個類別，0 ~ 9。



```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_digits(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)
```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_digits()

X = data.data
print('The shape of X: ', X.shape, '\n')

print('The class names:')
display(data.target_names)

y = data.target
print('The shape of y: ', y.shape)
```


# Linnerrud 數據集

* Number of Instances: <font color=#0000FF>$20$</font> <br>
  這個資料集有 $20$ 筆資料。
* Number of Attributes: 3 <br>
  每筆資料有 $3$ 個生理變量。
* Number of Output: 3 <br>
  每筆資料有 $3$ 個運動變量。
* Category : regression




```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_linnerud(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)
```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_linnerud()

print('The data features:')
display(data.feature_names)

X = data.data
print('The shape of X: ', X.shape, '\n')

print('The target features:')
display(data.target_names)

y = data.target
print('The shape of y: ', y.shape)

```


# Wine recognition dataset 葡萄酒識別數據集

* Number of Instances: <font color=#0000FF>$178$</font> <br>
  這個資料集有 $178$ 筆資料。
* Number of Attributes: 13 <br>
  每筆資料有 $13$ 個特徵。
* Category : classification

## Attributes 特徵分別是
- alcohol 醇
- malic_acid 蘋果酸
- ash 灰
- alcalinity_of_ash 灰的鹼度
- magnesium 鎂
- total_phenols 總酚
- flavanoids 類黃酮
- nonflavanoid_phenols 非類黃酮酚
- proanthocyanins 原花青素
- color_intensity 色彩強度
- hue 色調
- od280/od315_of_diluted_wines 稀釋酒的 OD280 / OD315
- proline 脯氨酸

## class
- class_0 (59)
- class_1 (71)
- class_2 (48)



```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_wine(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)
```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_wine()

print('The data features:')
display(data.feature_names)

X = data.data
print('The shape of X: ', X.shape, '\n')

print('The class names:')
display(data.target_names)

y = data.target
print('The shape of y: ', y.shape)
```


# Breast cancer wisconsin (diagnostic) dataset. wisconsin 州乳腺癌（診斷）數據集

* Number of Instances: <font color=#0000FF>$569$</font> <br>
  這個資料集有 $569$ 筆資料。
* Number of Attributes: 30 <br>
  每筆資料有 $30$ 個特徵。
* Category : classification



```python 
# 得到資料 方法一: 直接得到 X, y 
from sklearn import datasets

X, y = datasets.load_breast_cancer(return_X_y=True)

print('The shape of X: ', X.shape, ' The shape of y: ', y.shape)
```


```python 
# 得到資料 方法二: 先拿到資料物件，再得到 X, y
from sklearn import datasets

data = datasets.load_breast_cancer()

print('The data features:')
display(data.feature_names)

X = data.data
print('The shape of X: ', X.shape, '\n')

print('The class names:')
display(data.target_names)

y = data.target
print('The shape of y: ', y.shape)
```
