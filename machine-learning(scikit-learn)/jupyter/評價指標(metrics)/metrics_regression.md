

# Absolute Error 絕對誤差 與 Relative Error 相對誤差

先來上定義，假設 $y_i$ 是真實的值， $\hat{y}_i$ 是你預測的值 <br>

Absolute Error:
$$
|y_i - \hat{y}_i|
$$

Relative Error:
$$
\frac{y_i - \hat{y}_i}{y_i}
$$

看公式沒什麼感覺，下面來舉實際的例子，假設你要預估自己的身高，你的真實身高是 170 公分，但你心中認定你身高為完美的 180，我們來看看絕對誤差
$$
|170-180| = 10
$$
你跟你想像的只差10，那我們說不要不要，我要用公尺為單位你的絕對誤差 $|1.7-1.8|=0.1$，挖阿你一下數值就進步了，你的預測什麼都沒變，
你也說的都是事實，但是一下就變好了，所以有人會說統計可以騙人，最好騙的人是什麼人，半桶水的人最好騙，
因為他懂一點但是又不深入，你給他一些數據結果，他也不深入了解裡面的算法與數據，就可以迷惑他的眼睛與思想。 <br>

如果我們用相對誤差來看，不管用哪種單位都是相同的
$$
\frac{170 - 180}{170} = \frac{1.7 - 1.8}{1.7}
$$
那你會不會想說相對誤差就比絕對誤差還好，我只能跟你說不一定，下面來說點小故事。





假設你是做行銷的，某天你要去面試，你想要把你的能力展現給下一家的老闆看，如果你之前在大公司做有很高的知名度，那你可以展現真實數據，例如你做的行銷案，有千萬觀看，賣出百萬，雖然你只是照著公司SOP做的案子，你毫無新意，大公司老闆要推的產品，你就只是小編，照著模板修修改改，就推出了，你不能有個人意志去做你的創意，但是之前累積的名聲與聲望就可以讓你做的專案效果良好，但是跟之前的成果比可能差距不大，誰來做都是千萬觀看，賣出百萬。 <br>

如果你之前待的是家小公司，前面一個產品只賣出10個，你的新專案無前例可以參考，之前案子的前輩還離職了，你的宣傳就是寫幾個字PO在FB上，你大部分時間都在親朋好友間宣傳你的產品，還讓大家告訴大家，天天參加聚會去推廣最後賣出100個，這時候你不要難過，你可以跟下一個老闆講述你的成長性，你的策略病毒式傳播等等，你要說在你領導下業績翻了 10 倍，這時候是不是相對的數據就好看很多，這次不光教你們怎麼看評價指標，也教你們怎麼宣傳自己。 <br>

下面我們就看 scikit learn 提供那些指標




# Mean Absolute Arror (MAE)

公式

$$
MAE(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$


這邊要提一下 scikit learn 有支援多輸出(multi output)





```python 
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

display(mean_absolute_error(y_true, y_pred))

# multi output
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

display(mean_absolute_error(y_true, y_pred))

display(mean_absolute_error(y_true, y_pred, multioutput='raw_values'))

display(mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))

```


# Mean Squared Error (MSE)

公式

$$
MSE(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$


這邊要提一下 scikit learn 有支援多輸出(multi output)




```python 
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

display(mean_squared_error(y_true, y_pred))

# multi output
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

display(mean_squared_error(y_true, y_pred))

display(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

display(mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7]))
```


# Mean Squared Logarithmic Error (MSLE)

公式

$$
MSLE(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n ( \log_e (1+y_i) - \log_e (1+\hat{y}_i) )^2
$$

我們可以看一下
$$
\log_e (1+y_i) - \log_e (1+\hat{y}_i) = \log_e \frac{1+y_i}{1+\hat{y}_i}
$$
他其實是相對的誤差，我們下面看一下差別，假設真實的值 $ \hat{y}_i = 10$，不同的 $\hat{y}_i$ 差別是什麼。



```python 
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 10))

x = np.linspace(5, 15, 100)
y_1 = abs(x-10)
y_2 = abs(np.log((1+x)/(1+10)))

# plot
plt.subplot(121)
plt.plot(x, y_1, linewidth=2.0)
plt.subplot(122)
plt.plot(x, y_2, linewidth=2.0)

plt.show()


```


我們可以看到他的差別，這個 MSLE 適合用來估計目標有可能指數成長的結果，例如人口或是薪水，下面我們進入實例。



```python 
from sklearn.metrics import mean_squared_log_error
y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]
display(mean_squared_log_error(y_true, y_pred))


y_true = [[0.5, 1], [1, 2], [7, 6]]
y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
display(mean_squared_log_error(y_true, y_pred))

display(mean_squared_log_error(y_true, y_pred, multioutput='raw_values'))

display(mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7]))


```


# Mean Absolute Percentage Error (MAPE)

就是估計相對誤差的，下面來看公式

$$
MAPE(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\max (\epsilon,|y_i|)}
$$

裡面的 $\epsilon$ 是任意小的正數，去避免分母為零。




```python 
from sklearn.metrics import mean_absolute_percentage_error
y_true = [1, 10, 1e6]
y_pred = [0.9, 15, 1.2e6]

display(mean_absolute_percentage_error(y_true, y_pred))

# multi output
y_true = [[1, 2], [10, 9], [1e6, 1.1e6]]
y_pred = [[0.9, 1.1], [15, 17], [1.2e6, 1.2e7]]

display(mean_absolute_percentage_error(y_true, y_pred))

display(mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values'))

display(mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7]))

```


# Median absolute error (MedAE)

接下來要介紹的是絕對值誤差的中位數，下面來看公式

$$
MedAE(y, \hat{y}) = median(|y_1 - \hat{y}_1|, \cdots, |y_n - \hat{y}_n|)
$$



```python 
from sklearn.metrics import median_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

median_absolute_error(y_true, y_pred)

```


# Max error

接下來要介紹的是最大誤差，下面來看公式

$$
MaxError(y, \hat{y}) = \max (|y_1 - \hat{y}_1|, \cdots, |y_n - \hat{y}_n|)
$$



```python 
from sklearn.metrics import max_error

y_true = [3, 2, 7, 1]
y_pred = [9, 2, 7, 1]

max_error(y_true, y_pred)

```


# Explained variance score

可解釋的 variance 的公式

$$
ExplainedVariance(y, \hat{y}) = 1 - \frac{Var\{y-\hat{y}\}}{Var\{y\}} 
= 1 - \frac{\sum_{i=1}^n(y_i-\hat{y}_i)^2}{\sum_{i=1}^n(y_i-0)^2}
$$


# $R^2$ score

$R^2$ 的公式

$$
ExplainedVariance(y, \hat{y}) = 1 - \frac{Var\{y-\hat{y}\}}{Var\{y\}} 
= 1 - \frac{\sum_{i=1}^n(y_i-\hat{y}_i)^2}{\sum_{i=1}^n(y_i-\mu_y)^2}
$$


我們要提一下如果 $\mu_y=0$，則上面兩個相等。




```python 
# Explained variance score

from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(explained_variance_score(y_true, y_pred))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(explained_variance_score(y_true, y_pred, multioutput='raw_values'))

print(explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7]))

```


```python 
# R2

from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print(r2_score(y_true, y_pred))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(r2_score(y_true, y_pred, multioutput='variance_weighted'))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(r2_score(y_true, y_pred, multioutput='uniform_average'))

print(r2_score(y_true, y_pred, multioutput='raw_values'))

print(r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))

```
