<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>




# optimization

[pytorch optimization algorithms](https://pytorch.org/docs/stable/optim.html)
[tensorflow optimization algorithms](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
[最佳化](https://zh.wikipedia.org/zh-tw/%E6%9C%80%E4%BC%98%E5%8C%96)





# 希臘三哲
* 蘇格拉底-Socrates
* 柏拉圖-Plato
* 亞里士多德-Aristotle


## 什麼是愛情？
有一天，柏拉圖問蘇格拉底：「什麼是愛情？」

蘇格拉底說：「我請你穿越這片麥田，去摘一株最大最金黃的麥穗回來，但是有個規則：你只能摘一次且不能走回頭路。」

於是柏拉圖就朝麥田走去。許久之後，他卻空著雙手回來了。

蘇格拉底問他：「怎麼了？為何空手回來了？」

柏拉圖說道：「當我走在田間的時候，曾看到過好幾株特別大特別燦爛的麥穗，可是，我總想著也許前面還會有更大更好的，於是就放過沒有摘；但是，我繼續往下走的時候，看到的麥穗都不滿意，總覺得還不如先前看到的好，所以最後什麼都沒有摘到…。」

蘇格拉底意味深長地說：「這，就是愛情。」

## 什麼是婚姻？

有一天，柏拉圖又問蘇格拉底：「什麼是婚姻？」

蘇格拉底說：「我請你穿越這片樹林，去砍一棵最粗最結實的樹回來放在屋子裏做聖誕樹，但是有個規則：你不能走回頭路，而且你只能砍一次。」

於是，柏拉圖就往樹林走去。

許久之後，他帶了一 棵還算可以但並不是最高大粗壯的樹回來了。

蘇格拉底問他，怎麼只砍了這樣不太起眼的一棵樹回來？

柏拉圖說道：當我穿越樹林的時候，看到過幾棵非常好的樹，這次，我吸取了上次摘麥穗的教訓，看到這棵樹還不錯，就乾脆選它了，以免像上次一樣又錯過了砍樹的機會而空手而歸，儘管它並不是我碰見的最棒的一棵，但有總比沒有好。

這時，蘇格拉底意味深長地說：「這，就是婚姻。」

## 什麼是幸福

還有一次，柏拉圖問蘇格拉底：「什麼是幸福？」

蘇格拉底說：「我請你穿越這片田野，去摘一朵最美麗的花，但是有個規則：你不能走回頭路，而且你只能摘一次。」

於是，柏拉圖去做了。許久之後，他捧著一朵比較美麗的花回來了。

蘇格拉底問他：「這就是最美麗的花了？」

柏拉圖說道：「當我穿越田野的時候，我一眼看到了這朵美麗的花，我就認定了它是最美麗的並將它摘下，雖然我後來又看見很多其它更美麗的花的時候，我依然堅持著我這朵是心中最美的信念而不再動搖。所以這就是我選定的最美麗的花。」

這時，蘇格拉底意味深長地說：「這，就是幸福。」






## 數學化柏拉圖問題

稻田中總共有݊ $n$ 株稻穗，且每株稻穗都有一個隨機的相應值以代表這株稻穗有多好。
我們要取得最好一株的策略為「前幾株稻穗只觀察而不拾取，之後只要一看到比觀察值更好的稻穗就將它拾取並走到盡頭」。 
我們令只觀察不拾取的稻穗為前݇ $k$ 株，在此策略下拾到最好的稻穗的機率為ܲ $P_k$ 。
則我們的目的是設法找到 $k$ 去得到最大值的݇ $P_k$。


## 範例
假設有 $5$ 個隨機數 ， 假定為 $10, 100, 20, 1000, 1$ ， 
一次一個拿到你面前，你可以決定要或不要，但是起手無回不能後悔重拿，在此情況下你的策略是什麼？
如果現在假設有 $100$ 個隨機數 ， 你的拿取策略是什麼？






## 符號定義

假設第 $i$ 個位置的稻穗大小為 $x_i$ ， 最大的值為 $x_{max}$
令事件 $E_i$ 為取到第 $i$ 個位置的稻穗

# 解

我們可以考慮怎麼取到最大的稻穗，就是如果我們取到第 $i$ 個稻穗，
還剛好第 $i$ 個稻穗是最大的

$$
P_k = \sum_{i=1}^n Prob(x_i = x_{max}) \cdot Prob(E_i | x_i = x_{max})
$$

我們下一步可以簡單簡化，因為假設是隨機的所以 $Prob(x_i = x_{max}) = \frac{1}{n}$，
又因為我們的策略是前 $k$ 個稻穗只觀察不拿取 ， 所以 $Prob(E_i) = 0, i \leq k$。

$$
P_k = \frac{1}{n} \sum_{i=k+1}^n  Prob(E_i | x_i = x_{max})
$$

我們怎麼知道 $Prob(E_i | x_i = x_{max})$ 的值，
如果在前 $i-1$ 個稻穗，第二大的稻穗在我們觀察的前 $k$ 個 ， 那我們就一定可以找到最大的那個稻穗。

$$
Prob(E_i | x_i = x_{max}) = \frac{k}{i-1}
$$

where $i > k$。

所以

$$
P_k = \frac{1}{n} \sum_{i=k+1}^n  \frac{k}{i-1}
$$

我們整理一下可以得到

$$
P_k = \frac{k}{n} \sum_{i=k}^{n-1} \frac{1}{i}
$$







假設 $t = \frac{k}{n}$ ， 如果我們令 $n$ 趨近於無窮大 ，
利用微積分可以得到

$$
P_k = \lim_{n \rightarrow \infty} t \sum_{i=k}^{n-1} \frac{1}{i} 
\geq \lim_{n \rightarrow \infty} t \int_{i=k}^{n-1} \frac{1}{x} dx
$$

下面的不等號可以從積分定義還有分別代表的面積得到。
所以我們可以得到

$$
P_k \geq \lim_{n \rightarrow \infty} t \ln \frac{n-1}{k} = t \ln \frac{1}{t}
$$

令 

$$
p(t) := -t \ln t
$$

我們可以用微分求出函數 $p$ 的極值在 $t=\frac{1}{e}$




所以我們得到一個策略，就是先觀察前 $\frac{1}{e}$ 的資料，再挑之後看到比前面大的策略。



# 我們可以先簡單地看看


```python 
import numpy as np

max_num = 50
iteration_times = 10
number_k = 0

accumulation_max_number = 0
accumulation_value = 0

for t in range(iteration_times):
    choose = 0
    permuted = np.random.permutation(range(1,max_num+1))
    print(permuted)
    if number_k == 0:
        ob_max = 0
    else:    
        ob_max = max(permuted[:number_k])

    
    for i in range(number_k, max_num):
        if permuted[i] > ob_max:
            choose = permuted[i]
            break
    print('OB max: ', ob_max, ' Choose: ',  choose)
    if choose == max_num:
        accumulation_max_number += 1

    accumulation_value += (choose / max_num)


(accumulation_max_number / iteration_times), (accumulation_value / iteration_times)



```


# 如果改用比例來看


```python 
import numpy as np

max_num = 100
iteration_times = 100
rate = 0.3
number_k = int(max_num * rate)

accumulation_max_number = 0
accumulation_value = 0

for t in range(iteration_times):
    choose = 0
    permuted = np.random.permutation(range(1,max_num+1))
    ob_max = max(permuted[:number_k])

    for i in range(number_k, max_num):
        if permuted[i] > ob_max:
            choose = permuted[i]
            break
    
    if choose == max_num:
        accumulation_max_number += 1

    accumulation_value += (choose / max_num)


(accumulation_max_number / iteration_times), accumulation_value
```


# 我們把它寫成一函數


```python 
def wheat_estimate(max_num = 100, iteration_times = 100, number_k = 30, rate = None):
    if rate:
        number_k = int(max_num * rate)

    accumulation_max_number = 0
    accumulation_value = 0

    for t in range(iteration_times):
        choose = 0
        permuted = np.random.permutation(range(1,max_num+1))
        if number_k == 0:
            ob_max = 0
        else:    
            ob_max = max(permuted[:number_k])

        for i in range(number_k, max_num):
            if permuted[i] > ob_max:
                choose = permuted[i]
                break
        
        if choose == max_num:
            accumulation_max_number += 1

        accumulation_value += (choose / max_num)

    return (accumulation_max_number / iteration_times), (accumulation_value / iteration_times)


```


```python 
import matplotlib.pyplot as plt

max_number_list = []
accumulation_value_list = []
_range = range(0, 40)

for i in _range:
    _ = wheat_estimate(max_num = 50,  iteration_times = 1000, number_k = i)
    max_number_list.append(_[0])
    accumulation_value_list.append(_[1])


plt.figure(figsize=(20,10))

plt.plot(_range, max_number_list, 's-',color = 'r', label="max num")
plt.plot(_range, accumulation_value_list,'o-',color = 'g', label="acc val")

plt.xlabel("k")
plt.ylabel("value")
plt.legend()

plt.show()

```


```python 
(1/np.e) * 50
```


# 如果我們考慮其他策略

如果觀察前 $k$ 個， 如果比 ob_max $*$ rate_k 大我們就選他


```python 
def wheat_estimate(max_num = 100, iteration_times = 100, number_k = 30, rate_k = 1.0, rate = None):
    if rate:
        number_k = int(max_num * rate)

    accumulation_max_number = 0
    accumulation_value = 0

    for t in range(iteration_times):
        choose = 0
        permuted = np.random.permutation(range(1,max_num+1))
        if number_k == 0:
            ob_max = 0
        else:    
            ob_max = max(permuted[:number_k])

        for i in range(number_k, max_num):
            if permuted[i] > (ob_max * rate_k):       # 差在這邊
                choose = permuted[i]
                break
        
        if choose == max_num:
            accumulation_max_number += 1

        accumulation_value += (choose / max_num)

    return (accumulation_max_number / iteration_times), (accumulation_value / iteration_times)


    
```


```python 
import matplotlib.pyplot as plt

max_number_list = []
accumulation_value_list = []
_range = range(0, 40)

for i in _range:
    _ = wheat_estimate(max_num = 50,  iteration_times = 1000, number_k = i, rate_k = 0.8)
    max_number_list.append(_[0])
    accumulation_value_list.append(_[1])


plt.figure(figsize=(20,10))

plt.plot(_range, max_number_list, 's-',color = 'r', label="max num")
plt.plot(_range, accumulation_value_list,'o-',color = 'g', label="acc val")

plt.xlabel("k")
plt.ylabel("value")
plt.legend()

plt.show()
```


# 如果現在

你不知道面對的人生，你可以觀察幾次


```python 
import numpy as np
np.random.randint(10, size=5) + 1
```


```python 
def wheat_estimate(max_num = 100, iteration_times = 100, number_k = 30, rate_k = 1.0, rate = None):
    if rate:
        number_k = int(max_num * rate)

    accumulation_max_number = 0
    accumulation_value = 0

    max_num_list =  np.random.randint(max_num, size=iteration_times) + 1      # 差在這邊

    for max_number in max_num_list:
        choose = 0
        permuted = np.random.permutation(range(1,max_number+1))
        if number_k == 0:
            ob_max = 0
        else:    
            ob_max = max(permuted[:number_k])

        for i in range(number_k, max_number):
            if permuted[i] > (ob_max * rate_k):
                choose = permuted[i]
                break
        
        if choose == max_number:
            accumulation_max_number += 1

        accumulation_value += (choose / max_number)

    return (accumulation_max_number / iteration_times), (accumulation_value / iteration_times)






```


# 我們看看固定 rate_k 找 觀察 k 哪個好


```python 
import matplotlib.pyplot as plt

max_number_list = []
accumulation_value_list = []
_range = range(0, 40)

for i in _range:
    _ = wheat_estimate(max_num = 100,  iteration_times = 1000, number_k = i, rate_k = 1.0)
    max_number_list.append(_[0])
    accumulation_value_list.append(_[1])


plt.figure(figsize=(20,10))

plt.plot(_range, max_number_list, 's-',color = 'r', label="max num")
plt.plot(_range, accumulation_value_list,'o-',color = 'g', label="acc val")

plt.xlabel("k")
plt.ylabel("value")
plt.legend()

plt.show()
```


# 我們看看固定 k 找 rate_k 哪個好


```python 
import matplotlib.pyplot as plt

max_number_list = []
accumulation_value_list = []
fix_k = 50
divid_num = 100
_range = range(0, divid_num)

for i in _range:
    _ = wheat_estimate(max_num = 100,  iteration_times = 1000, number_k = fix_k, rate_k = (i/divid_num))
    max_number_list.append(_[0])
    accumulation_value_list.append(_[1])


plt.figure(figsize=(20,10))

plt.plot(_range, max_number_list, 's-',color = 'r', label="max num")
plt.plot(_range, accumulation_value_list,'o-',color = 'g', label="acc val")

plt.xlabel("k")
plt.ylabel("value")
plt.legend()

plt.show()
```
