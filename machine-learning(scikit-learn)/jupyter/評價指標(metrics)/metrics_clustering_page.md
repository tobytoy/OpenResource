<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# Clustering performance evaluation
這次要介紹在分群 clustering 任務上的評價指標。


## rand index

假設有集合 

$$
S=\{ s_1, s_2, \cdots, s_n \}
$$

我們有兩個分群的方法

$$
X=\{ X_1,X_2, \cdots, X_p\}, Y=\{ Y_1,Y_2, \cdots, Y_q\}
$$

我們定義下面四個數

* $a$ 表示在 $S$ 中的元素對，一起在 $X$ 中的同個子集，也一起在 $Y$ 中的同個子集，的元素對數量。
* $b$ 表示在 $S$ 中的元素對，在 $X$ 中的不同個子集，在 $Y$ 中的不同個子集，的元素對數量。
* $c$ 表示在 $S$ 中的元素對，一起在 $X$ 中的同個子集，在 $Y$ 中的不同個子集，的元素對數量。
* $d$ 表示在 $S$ 中的元素對，在 $X$ 中的不同個子集，一起在 $Y$ 中的同個子集，的元素對數量。

我們就可以定義 RI （rand index）

$$
RI := \frac{a+b}{a+b+c+d} = \frac{a+b}{C^n_2}
$$

我們可以用集合的概念表示 $a,b,c,d$ 。

$$
a = \#\{(s_i,s_j) | s_i, s_j  \in X_k, s_i,s_j \in Y_l \}
$$

$$
b = \#\{(s_i,s_j) | s_i \in X_{k_1}, s_j \in X_{k_2}, s_i \in Y_{l_1}, s_j \in Y_{l_2} \}
$$

$$
c = \#\{(s_i,s_j) | s_i, s_j  \in X_k, s_i \in Y_{l_1}, s_j \in Y_{l_2} \}
$$

$$
d = \#\{(s_i,s_j) | s_i \in X_{k_1}, s_j \in X_{k_2}, s_i,s_j \in Y_l \}
$$

我們也可以用以前的概念來看，

* $a$ 就是 TP （true positive）的數量。
* $b$ 就是 TN （true negatives）的數量。
* $c$ 就是 FN （false negatives）的數量，也叫偽陰，型二錯誤。
* $d$ 就是 FP （false positive）的數量，也叫偽陽，型一錯誤。

那 RI 就可以理解為

$$
RI := \frac{TP+TN}{TP+TN+FN+FP}
$$





再來要介紹 Adjusted Rand Index ， 他可以理解為修正機率版的 Rand Index ， 為何要提出 ARI 不能用 RI，
是因為有人提出在給隨機分群的時候 RI 不靠近 0 ，所以 Hubert 和Arabie 在1985年提出了，
可以[參考](https://i11www.iti.kit.edu/extra/publications/ww-cco-06.pdf)。


下面要定義一堆符號來講他的公式，
假設 $n_{ij} := \# X_i \cap Y_j$


| X\Y | $Y_1 \quad Y_2 \quad \cdots \quad Y_q$ | sum |
| :----:| :----: | :----: |
| $X_1$ | $n_{11} \quad n_{12} \quad \cdots \quad n_{1q}$ | $a_1$ |
| $X_2$ | $n_{21} \quad n_{22} \quad \cdots \quad n_{2q}$ | $a_2$ |
| $\vdots$ | $\vdots \quad \vdots \qquad \ddots \quad \vdots$ | $\vdots$ |
| $X_p$ | $n_{p1} \quad n_{p2} \quad \cdots \quad n_{pq}$ | $a_p$ |
| sum | $b_1 \quad b_2 \quad \cdots \quad b_q$ |  |


我們就可以定義 ARI 為

$$
ARI := \frac{RI -E[RI]}{\max(RI)-E[RI]} 
= \frac{\sum_{ij}C^{n_{ij}}_2 - [t_1 \cdot t_2]/C^n_2}{\frac{t_1 + t_2}{2} - [t_1 \cdot t_2]/C^n_2}
$$

where $t_1 = \sum_i C^{a_i}_2, t_2 = \sum_j C^{b_j}_2$。

下面我們實際舉個例子看怎麼計算，
假設 $S=\{1,2,3,4,5\}$

$$
X=\{ X_1=\{1,2,3\}, X_2=\{4,5\} \}
$$

$$
Y=\{ Y_1=\{1,2\}, Y_2=\{3,4,5\} \}
$$

可以得到 Table 長這樣


| X\Y | $Y_1$ | $Y_2$ | sum |
| :----: | :----: | :----: | :----: |
| $X_1$ | $\{1,2\}$ | $\{3\}$ | $3$ |
| $X_2$ | $\emptyset$  | $\{4,5\}$ | $2$ |
| sum | $2$ |  $3$ |  |

帶入公式可以算出 $t_1 = t_2 = (C^3_2 + C^2_2) = 4$，也知道 $C^5_2=10$，下面就可以帶入得到

$$
ARI := \frac{2-1.6}{4-1.6} \sim 0.16666666666666663
$$

下面來看看 scikit-learn 算的結果吧。



```python 
from sklearn.metrics import adjusted_rand_score

labels_true = [0, 0, 0, 1, 1]
labels_pred = [0, 0, 1, 1, 1]

display(adjusted_rand_score(labels_true, labels_pred))

```


下面來測試隨機給分群會接近 0 嗎?



```python 
import random
from sklearn.metrics import rand_score, adjusted_rand_score

n = 5

labels_true = [0] * n
labels_pred = [0] * n

_ = list(range(n))
random.shuffle(_)
break_index = random.randint(0, n)
# print(_[:break_index], _[break_index:])
for index in _[break_index:]:
    labels_true[index] = 1

_ = list(range(n))
random.shuffle(_)
break_index = random.randint(0, n)
for index in _[break_index:]:
    labels_pred[index] = 1

print('True: ', labels_true)
print('Pred: ', labels_pred)
print('Rand score: ', rand_score(labels_true, labels_pred))
print('Adjusted Rand score: ', adjusted_rand_score(labels_true, labels_pred))

```


```python 
import random
from sklearn.metrics import rand_score, adjusted_rand_score

n = 10
test_time = 1000
accumulation_RI  = 0
accumulation_ARI = 0

def random_split(n=5):
    _ = list(range(n))
    random.shuffle(_)
    break_index = random.randint(0, n)
    split_list = [1 if i in _[break_index:] else 0 for i in range(n)]
    return split_list

for i in range(test_time):
    labels_true = random_split(n)
    labels_pred = random_split(n)

    accumulation_RI += rand_score(labels_true, labels_pred)
    accumulation_ARI += adjusted_rand_score(labels_true, labels_pred)

accumulation_RI /= test_time
accumulation_ARI /= test_time

print('Average Rand score: ', accumulation_RI)
print('Average Adjusted Rand score: ', accumulation_ARI)

```


下面是 scikit-learn 上面的 rand_score 的實際使用 ， 根據上面的說明也可以看出這個方法跟給標記的順序無關。


```python 
from sklearn.metrics import rand_score

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

display(rand_score(labels_true, labels_pred))

labels_pred = [1, 1, 0, 0, 3, 3]

display(rand_score(labels_true, labels_pred))

```


## adjusted_rand_score


```python 
from sklearn.metrics import adjusted_rand_score

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

display(adjusted_rand_score(labels_true, labels_pred))

labels_pred = [1, 1, 0, 0, 3, 3]

display(adjusted_rand_score(labels_true, labels_pred))
```


## mutual information

下面介紹另一個指標，我們沿用上面的定義假設全部的集合是 $S$ 有兩個 分群分別是 $X$ 跟 $Y$，假設 $n_{ij} := \# X_i \cap Y_j$

| X\Y | $Y_1 \quad Y_2 \quad \cdots \quad Y_q$ | sum |
| :----:| :----: | :----: |
| $X_1$ | $n_{11} \quad n_{12} \quad \cdots \quad n_{1q}$ | $a_1$ |
| $X_2$ | $n_{21} \quad n_{22} \quad \cdots \quad n_{2q}$ | $a_2$ |
| $\vdots$ | $\vdots \quad \vdots \qquad \ddots \quad \vdots$ | $\vdots$ |
| $X_p$ | $n_{p1} \quad n_{p2} \quad \cdots \quad n_{pq}$ | $a_p$ |
| sum | $b_1 \quad b_2 \quad \cdots \quad b_q$ |  |

再來要引用 entropy 的概念，中文叫 熵。

$$
H(X) = - \sum_{i=1}^p \frac{\# X_i}{n} \log( \frac{\# X_i}{n} )
$$

$$
H(Y) = - \sum_{j=1}^q \frac{\# Y_j}{n} \log( \frac{\# Y_j}{n} )
$$

然後我們就能定義 MI

$$
MI(X,Y) = \sum_{i=1}^p \sum_{j=1}^q \frac{\# X_i \cap Y_j}{n} \log( \frac{N \# X_i \cap Y_j}{\# X_i \cdot \# Y_j} )
$$

再來要定義 normalized mutual information

$$
NMI(X,Y) = \frac{MI(X,Y)}{mean (H(X), H(Y))}
$$

如果有跟上看到這邊也會發現出了什麼問題，所以有提出adjusted mutual information

$$
AMI(X,Y) = \frac{MI - E[MI]}{mean(H(X), H(V)) - E[MI]}
$$






```python 
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

print('MI: ', mutual_info_score(labels_true, labels_pred))
print('NMI: ', normalized_mutual_info_score(labels_true, labels_pred))
print('AMI: ', adjusted_mutual_info_score(labels_true, labels_pred))

```


更多的指標可以[參考](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

