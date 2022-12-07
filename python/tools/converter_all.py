import os
import subprocess

# Blog
cmd_string = 'python converter_ipynb2md.py'
comp_process = subprocess.run(cmd_string, shell=True, check=True)

# Courses
os.chdir('../../MyCourses/tools/')
cmd_string = 'python converter_ipynb2content_md.py'
comp_process = subprocess.run(cmd_string, shell=True, check=True)

