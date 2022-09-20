

# 三門問題

也被稱為 蒙提霍爾問題 （Monty Hall problem） 。

**假設你正在參加一個遊戲節目，要在三扇門中選擇一扇：
其中一扇後面有大獎一輛車；其餘兩扇後面則只是一隻山羊。
當你選擇了一道門 （假設是一號門），
然後知道答案的主持人，
開啟了另一扇後面有羊的門（假設是三號門）。
他然後問你：「你想改變選擇換二號門嗎？」
此時你一定會想，
轉換選擇會增加勝率嗎？
還是說其實並沒有什麼差別，只剩兩個門二選一的機率都是 $1/2$ 。**


## Monte Carlo method

下面介紹一種我們常用的方法，去解決一些渾沌一些渾沌未知的問題叫 蒙地卡羅方法
（Monte Carlo method）。




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
我們假設我們堅持不換，下面可以分為兩種情況 <br>

Case 1 : 第一次就猜中 <br>

這個情況機率是 $1/3$ ， 因為有三個門，只有一個門後面是車。 <br>

Case 2 : 第一次沒猜中 <br>

這個情況機率是 $2/3$ ， 因為有三個門，只有一個門後面是車。

</details>


<details>
<summary> Case 要換 </summary>
我們假設我們堅持要換，下面可以分為兩種情況 <br>

Case 1 : 第一次就猜中 <br>

這個情況機率是 $1/3$ ， 如果堅持要換那最後結局是不中。 <br>

Case 2 : 第一次沒猜中 <br>

這個情況機率是 $2/3$ ， 如果堅持要換那最後結局是中。 <br>

Example  <br>

如果不能想像我們來舉個例子，假設有 $A,B,C$ 三個門
，車子在 $C$ 門，那 Case 1，就是你猜 $A$ 或 $B$，
假設是 $A$ ，主持人打開 $B$ ，那你要不要換。

</details>


<details>
<summary> Case 隨機要不要換 </summary>
因為主持人會打開一個不中的門，所以如果你是隨機的話，中與不中機率就變為一半。
</details>




## 算命

大家有去算過命嗎? 大家覺得算命可信嗎? <br>

我個人對算命的理解，它是心理學加統計學使用蒙地卡羅法，
去推算出你在某個時間你的狀況如何，
這可以包含星座、血型、生辰八字等等資料。
其背後可能有很複雜的理論依據，這部份我們先不用去理解，
直接用統計去看結果就好。



