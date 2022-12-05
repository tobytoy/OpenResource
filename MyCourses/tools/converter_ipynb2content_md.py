from pathlib import Path
from tqdm import tqdm
import os
import json





##############################################################
####                     匯入清單
##############################################################
path_source_root      = Path('../PythonBasic/content/')
path_destination_root = Path('../PythonBasic/content_md/')





def replace_function(string_item):
    # 要更換在這邊
    string_output = string_item.replace('$', '$$')
    return string_output

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





    for path_ipynb in list(path_source_root.glob('*')):
    
        with open(path_ipynb, encoding = "UTF-8") as file:
            data = json.load(file)

    file_name      = path_ipynb.name.split('.ipynb')[0] + '.md'
    file_page_name = path_ipynb.name.split('.ipynb')[0] + '_page.md'
    file_path      = path_destination_root / file_name
    file_page_path = path_destination_root / file_page_name
    file_text      = ""
    file_page_text = ""

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
                file_page_text += "\n\n```python \n" + "".join(cells_source) + "\n```\n"
            elif cells_type == 'markdown':
                file_text += "\n\n" + "".join(list(map(replace_function, cells_source))) + "\n"
                file_page_text += "\n\n" + "".join(cells_source) + "\n"

    if file_path.exists():
        os.remove(file_path)
    if file_page_path.exists():
        os.remove(file_page_path)

    with open(file_path, "w", encoding = "UTF-8") as file:
        file.write(file_text)

    with open(file_page_path, "w", encoding = "UTF-8") as file:
        file.write(text_header+file_page_text)
    

def main():
    for path in tqdm(path_list, desc = "Converter Progress:"):
        converter(path)

if __name__ == '__main__':
    main()


