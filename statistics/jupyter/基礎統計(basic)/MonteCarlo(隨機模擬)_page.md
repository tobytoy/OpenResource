<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# 三門問題

也被稱為 Monty Hall problem 蒙提霍爾問題。

**假設你正在參加一個遊戲節目，你被要求在三扇門中選擇一扇：
其中一扇後面有一輛車；其餘兩扇後面則是山羊。
你選擇了一道門，假設是一號門，然後知道門後面有什麼的主持人，
開啟了另一扇後面有山羊的門，
假設是三號門。他然後問你：「你想選擇二號門嗎？」
轉換你的選擇對你來說是一種優勢嗎？**


## Monte Carlo method

下面我們介紹一種常用的方法，去解決未知的問題叫 Monte Carlo method 蒙地卡羅方法。




```python 
import numpy as np
from sklearn.metrics import accuracy_score

size_number = 10

# 0 表示你堅持不換
# 1 表示你堅持要換
# 2 你也隨機要換不換
strategy = 2

car_number  = np.random.randint(3, size=size_number)
if size_number < 20:
    print('車在哪個門後面:\t', car_number)
your_choose = np.random.randint(3, size=size_number)
if size_number < 20:
    print('你選的門:\t', your_choose)
host_open = np.array([np.setdiff1d(range(3), [car_number[i], your_choose[i]])[0] for i in range(size_number)])
if size_number < 20:
    print('主持人開的門:\t', host_open)

    
if strategy == 0:
    your_new_choose = your_choose
elif strategy == 1:
    your_new_choose = np.array([np.setdiff1d(range(3), [host_open[i], your_choose[i]])[0] for i in range(size_number)])
elif strategy == 2:
    your_new_choose = np.array([np.setdiff1d(range(3), [host_open[i], your_choose[i]])[0] if np.random.randint(2)==1 else your_choose[i] for i in range(size_number)])
    

if size_number < 20:
    print('你新選的門:\t', your_new_choose)


print('你獲勝機率:\t', accuracy_score(car_number, your_new_choose)) 


```


## 直接解答


<details>
<summary> Case 不換 </summary>
我們假設我們堅持不換，下面可以分為兩種情況

### Case 1 : 第一次就猜中

這個情況機率是 $1/3$ ， 因為有三個門，只有一個門後面是車。

### Case 2 : 第一次沒猜中

這個情況機率是 $2/3$ ， 因為有三個門，只有一個門後面是車。

</details>


<details>
<summary> Case 要換 </summary>
我們假設我們堅持要換，下面可以分為兩種情況

### Case 1 : 第一次就猜中

這個情況機率是 $1/3$ ， 如果堅持要換那最後結局是不中。

### Case 2 : 第一次沒猜中

這個情況機率是 $2/3$ ， 如果堅持要換那最後結局是中。

### Example

如果不能想像我們來舉個例子，假設有 $A,B,C$ 三個門
，車子在 $C$ 門，那 Case 1，就是你猜 $A$ 或 $B$，
假設是 $A$ ，主持人打開 $B$ ，那你要不要換。

</details>


<details>
<summary> Case 隨機要不要換 </summary>
因為主持人會打開一個不中的門，所以如果你是隨機的話，中與不中機率就變為一半。
</details>




## 算命

大家有去算過命嗎? 大家相信算命嗎? <br>

我個人對算命的看法是心理學加統計學(用蒙地卡羅)，
去推算出某個時間你的狀況如何，
這可以包含星座、血型等等。
可能背後有很複雜的原因，我們不用去理解，直接用統計去看結果。





```python 
https://wiki.mbalib.com/zh-tw/%E8%92%99%E7%89%B9%E5%8D%A1%E7%BD%97%E6%96%B9%E6%B3%95
http://www.ablmcc.edu.hk/~scy/cprogram/Monte-Carlo.pdf
https://zh.wikipedia.org/zh-tw/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95
```
