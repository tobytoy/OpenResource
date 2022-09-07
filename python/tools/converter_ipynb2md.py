from pathlib import Path
from tqdm import tqdm
import json


path_dictionary_list = []

##############################################################
####                     ML 機器學習
##############################################################


# 簡介
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/')
path_dictionary['list'] = [
                '簡介(Introduction)/01-01 Introduction.ipynb',
                '簡介(Introduction)/01-02 First_Try.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 其他工具
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/')
path_dictionary['list'] = [
                '其他工具(other-tools)/00-01 (Appendex) Pandas.ipynb',
                '其他工具(other-tools)/00-02 (Appendex) Visualization.ipynb',
                '其他工具(other-tools)/00-03 (Appendex) seabon.ipynb',
                '其他工具(other-tools)/00-04 (Appendex) Save_Load.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 資料集
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/資料集(datasets)/')
path_dictionary['list'] = [
                '01-03 Toy-datasets.ipynb',
                '01-04 Generated-Datasets.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 線性模型
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/線性模型(Linear)/')
path_dictionary['list'] = [
                '02-01 Linear_Regression.ipynb',
                '02-02 Linear_Regression_Polynomial.ipynb',
                '02-03 Linear_Regression_Bayesian.ipynb',
                '02-04 Linear_Regression_Logistic.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 貝氏
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/貝氏(bayes)/')
path_dictionary['list'] = [
                '03_01 Bayes_Theory.ipynb',
                '03_02 Naive_Bayes.ipynb',
                ]
path_dictionary_list.append(path_dictionary)



# 基本模型 的 貝氏
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/貝氏(bayes)/')
path_dictionary['list'] = [
                '03_01 Bayes_Theory.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 SVM
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/支持向量機(SVM)/')
path_dictionary['list'] = [
                '04_01 Support_Vector_Machines.ipynb',
                '04_02 SVM_Kernel_Trick.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 SGD
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/隨機梯度下降(stochastic-gradient-descent)/')
path_dictionary['list'] = [
                '05_01 Stochastic_Gradient_Descent.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 NearestNeighbors
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/NearestNeighbors/')
path_dictionary['list'] = [
                '06_01 KNN.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 評價指標 metrics 
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/評價指標(metrics)/')
path_dictionary['list'] = [
                'introduce.ipynb',
                'metrics_classification.ipynb',
                ]
path_dictionary_list.append(path_dictionary)


# 基本模型 的 Clustering
path_dictionary = dict()
path_dictionary['root'] = Path('../../machine-learning(scikit-learn)/jupyter/基本模型(basic-model)/Clustering/')
path_dictionary['list'] = [
                '07_01 Introduction_and_Kmeans.ipynb',
                ]
path_dictionary_list.append(path_dictionary)



##############################################################
####                     匯入清單
##############################################################
path_list = []
for dictionary in path_dictionary_list:
    path_root = dictionary['root']
    path_end_list = dictionary['list']
    for item in path_end_list:
        path_list.append(path_root / item)


def converter(path_ipynb):
    text_header = """<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
    inlineMath: [ ["$","$"], ["\\(","\\)"] ],
    processEscapes: true
    }
});
</script>\n\n"""

    with open(path_ipynb, encoding = "UTF-8") as file:
        data = json.load(file)

    file_name = path_ipynb.name.split('.ipynb')[0] + '.md'
    file_page_name = path_ipynb.name.split('.ipynb')[0] + '_page.md'
    file_path = path_ipynb.parent / file_name
    file_page_path = path_ipynb.parent / file_page_name
    file_text = ""

    for _cells in data['cells']:
        flag = False
        cells_source = _cells['source']
        cells_type   = _cells['cell_type']

        for item in cells_source:
            if len(item.strip()) > 0:
                flag = True
                break
        
        if flag:
            if cells_type == 'code':
                file_text += "\n\n```python \n" + "".join(cells_source) + "\n```\n"
            elif cells_type == 'markdown':
                file_text += "\n\n" + "".join(cells_source) + "\n"

    with open(file_path, "w", encoding = "UTF-8") as file:
        file.write(file_text)

    with open(file_page_path, "w", encoding = "UTF-8") as file:
        file.write(text_header+file_text)
    

def main():
    for path in tqdm(path_list, desc = "Converter Progress:"):
        converter(path)

if __name__ == '__main__':
    main()


