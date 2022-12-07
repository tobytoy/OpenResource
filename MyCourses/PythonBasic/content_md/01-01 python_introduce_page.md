<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\(","\)"] ],
    processEscapes: true
    }
});
</script>



# [教學目標]

* 程式語言的運作原理 
* 程式語言如何實現跨平台
* 什麼是 python ， 為何用 python。
* python 的環境
* python 的虛擬環境



# 程式語言

計算機的硬件系統通常由五大部件構成，包括：運算器、控制器、存儲器、輸入設備和輸出設備。
其中，運算器和控制器放在一起就是我們通常所說的中央處理器，它的功能是執行各種運算和控制指令以及處理計算機軟件中的數據。
我們通常所說的程序實際上就是指令的集合，我們程序就是將一系列的指令按照某種方式組織到一起，然後通過這些指令去控制計算機做我們想讓它做的事情。
今天我們大多數時候使用的計算機，雖然它們的元器件做工越來越精密，處理能力越來越強大，但究其本質來說仍然屬於“馮·諾依曼結構”的計算機。 “馮·諾依曼結構”有兩個關鍵點，一是指出要將存儲設備與中央處理器分開，二是提出了將數據以二進制方式編碼。
二進制是一種"逢二進一"的計數法，跟我們人類使用的"逢十進一"的計數法沒有實質性的區別，人類因為有十根手指所以使用了十進制。
對於計算機來說，二進制在物理器件上來說是最容易實現的（高電壓表示1，低電壓表示0），於是在 "馮·諾依曼結構" 的計算機都使用了二進制。
雖然我們並不需要每個程序員都能夠使用二進制的思維方式來工作，但是了解二進制以及它與我們生活中的十進制之間的轉換關係，以及二進制與八進制和十六進制的轉換關係還是有必要的。


## 程式語言的編譯

![編譯](https://www.sitesbay.com/cpp/images/cpp-compiling.png)

精簡指令集簡稱（RISC），常被拿來和複雜指令集（CISC）做比較，兩者都是指令集架構（ISA）的不同設計方法。
指令集架構決定電腦如何執行指令，支援哪種數據類型和暫存器，如何管理記憶體容量，如何和其他裝置互動，等等。
電腦的處理器（含中央處理器CPU、圖形處理器GPU，等等）會遵循某種指令集架構來設計，而精簡指令集的概念就是，所有指令都要能在處理器的一個時脈週期內完成；這有別於複雜指令集的架構，支援更複雜的指令，需要好幾個時脈週期才能完成。


## 跨平台
實作跨平台性的方法是大多數編譯器在進行Java語言程式的編碼時候會生成一個用位元組碼寫成的「半成品」，這個「半成品」會在Java虛擬機器（解釋層）的幫助下執行，虛擬機器會把它轉換成當前所處硬體平台的原始程式碼。之後，Java虛擬機器會開啟標準庫，進行資料（圖片、執行緒和網路）的存取工作。 

![Java](https://openhome.cc/Gossip/JavaEssence/images/WhyJVM-3.jpg)



# 關於 Python

Python 是目前世界上最流行的程式語言之一，也是相當容易入門且功能強大的程式語言。它除了可以透過簡單邏輯進行程式設計，更具備有高效能的資料結構處理能力。Python 簡單易懂的程式語法，結合由 C 語言建構的特性，使得 Python 能跨平台開發，也幾乎能在所有作業系統中運作。


# Python 名稱的由來 

* 在 1989 年，一位荷蘭的程式設計師吉多。范羅蘇姆 ( Guido van Rossum )，在 1989 年的聖誕節期間，花了三個月的時間，創造出一套以 ABC 程式語言為基礎，作為替代 Unix shell 和 C 語言進行系統管理的程式語言：Python。

![Guido van Rossum](https://images0.cnblogs.com/blog/413416/201302/06100633-c2ce8755002945df846b5dad1dc25cdd.jpg)

* 范羅蘇姆是 BBC 電視劇 Monty Python's Flying Circus ( 蒙提派森的飛行馬戲團 ) 的愛好者，於是他就將這套程式語言命名為 Python，由於 Python 是「蟒蛇」的英文，所以在許多教學或文件中，都會使用一藍一黃的蟒蛇圖案作為 Python 的形象代表。

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" width="500px" />



* 1991年2月：第一個Python編譯器（同時也是解釋器）誕生，它是用C語言實現的（後面），可以調用C語言的庫函數。在最早的版本中，Python已經提供了對“類”，“函數”，“異常處理”等構造塊的支持，還有對列表、字典等核心數據類型，同時支持以模塊為基礎來構造應用程序。

* 1994年1月：Python 1.0正式發布。

* 2000年10月16日：Python 2.0發布，增加了完整的垃圾回收，提供了對Unicode的支持。與此同時，Python的整個開發過程更加透明，社區對開發進度的影響逐漸擴大，生態圈開始慢慢形成。

* 2008年12月3日：Python 3.0發布，它並不完全兼容之前的Python代碼。




# Python 的特色 
近幾年來，Python 已經逐漸變成最最熱門的程式語言之一，也是一直是最流行的程式語言前五名，主要有下列幾個原因：

* 語法簡潔、結構簡單，程式碼可讀性強，學習起來更加簡單 ( 閱讀好的 Python 程式碼，就好比在看英文文章 )。
* 免費且開源，擁有非常豐富的開發者社群支援。
* 完善的基礎程式庫，涵蓋網路、文件、資料庫、GUI...等。
* 非常強大的第三方程式庫，任何電腦可以實現的功能，都能透過 Python 實現。
* 應用範圍廣泛，能和絕大多數的程式語言 ( C/C++、C#、Java、JavaScript...等 ) 共同使用。

# Python 可以做什麼？

* 網路爬蟲與擷取資訊
* 數據處理分析與視覺化應用
* 機器學習與人工智慧
* 自動化測試
* 網站開發
* 軟體開發
* 商業應用
* 多媒體應用


# Python 之禪



```python 
import this

# this.__file__
# !cat c:\Users\ .. \this.py
# !type c:\Users\ .. \this.py

```



# Python 的執行原理

![編譯直譯](https://ithelp.ithome.com.tw/upload/images/20190905/201173994qFDAVHc0R.png)



# python 的環境

查看你的 python 版本

```python
!python -V
```

或是

```python
!python --version
```



## python 的搜尋路徑

PYTHONPATH 是 python 的搜尋路徑

* consoloe 命令提示界面

```bash
export PYTHONPATH=$PYTHONPATH:/your/path
```

* 在 python 中添加

```python
import sys

sys.path.append('/your/path')
```




# Anaconda


# Virtualenv





# [作業目標]

* python 可以拿來幹嘛? (複選題)

<ol type="A">
  <li>可以煮來吃</li>
  <li>可以做數據分析</li>
  <li>可以練打字速度</li>
  <li>可以做網路爬蟲</li>
  <li>可以向別人炫耀</li>
  <li>可以拿來當枕頭</li>
</ol>

* 如何查看你的 python 版本? 




# 參考資料

* https://docs.python.org/zh-tw/3/tutorial/index.html
* https://zh.wikipedia.org/zh-tw/Python
* https://www.cnblogs.com/vamei/archive/2013/02/06/2892628.html
* http://rogerchang.logdown.com/posts/5410082-does-python-need-to-compile-or-interpret
* https://zh.wikipedia.org/wiki/%E4%BA%8C%E8%BF%9B%E5%88%B6


