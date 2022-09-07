

# Expection and Variance

期望值或稱均值 (expection or mean) 是用來描述一組數據的中心位置，但是常常會有極端資料影響期望值，
所以也有其他指標(中位數 Median，四分位數 Quartile)來描述中心位置。
對於離散的隨機變量 $X$ (discrete random variables)，我們定義他的期望值為:

$$E(X) := \sum_x x P(x)$$

如果 $X$ 是連續的隨機變數，存在一個對應的機率密度函數(Probability Density Function) $P(x)$ 則我們定義期望值為

$$\int_{-\infty}^{\infty} x P(x) dx$$

在機率上我們常常用希臘字母 $\mu$ 來標記。




方差 Variance 是衡量一組數據變化的幅度 (variability) 的指標之一，
Variance 的定義是

$$Var(X) := E \Big( (X - \mu)^2 \Big)$$

還有另一個公式

$$Var(X) = E(X^2) - E(X)^2$$

大家自己展開隨便倒一下就會出來，我們在來說一個常用相關的定義標準差，也被稱為均方差(Standard Deviation)

$$SD(X) := \sqrt{Var(X)}$$

我們常用希臘字母 $\sigma$ 來表示 Standard Deviation


## Variance 的特性
下面說說 Variance 的特性。
- $Var(X + b) = V(X)$
- $Var(aX) = a^2 Var(X)$
- $Var(ax + b) = a^2 Var(X)$






下面我們會介紹偏態(Skewness)與峰度(Kurtosis)，偏態可以判斷隨機變量是往左或往右偏，峰度可以判斷隨機變量是高高尖尖的還是矮矮胖胖的，但是為了可以更系統的描述隨機變量，下面要介紹動差(moment)。

# Moment
假設隨機變量 $X$ 的機率密度函數(pdf) 為 $P(x)$ ，如果 $P$ 為離散的那我們定義對於 $c$ 點的 $n$ 階動差為

$$\mu_n(c) := \sum_{x} (x-c)^n P(x)$$

如果 $P$ 為連續函數則我們定義

$$\int_{-\infty}^{\infty} (x-c)^n P(x) dx$$

我們可以先看怎麼用動差描述 Expectation。

$$E(X) = \mu_1(0)$$

Variance 為 $2$ 階主動差

$$Var(X) = \mu_2(E(X))$$

## (Fisher's) Skewness 偏態

Skewness 的定義需要 $3$ 階主動差

$$Skew(X) := E \Big[ (\frac{X -\mu}{\sigma})^3 \Big]=\frac{\mu_3(E(X))}{\sigma^3}$$


## (Fisher's) Kurtosis 峰度

Kurtosis 的定義需要 $4$ 階主動差

$$Kurt(X) := \frac{\mu_4(E(X))}{\sigma^4}$$

### excess kurtosis

但是我們常常希望以常態分布為標準，所以我們會定義

$$Kurt(X) := \frac{\mu_4(E(X))}{\sigma^4} - 3$$

