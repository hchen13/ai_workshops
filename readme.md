# 环境搭建

在正式开始开发我们的图像分类系统之前, 我们需要先安装开发环境. 

## 1. Python 3

### 安装conda

若已经安装过conda, 请略过这部分.

一种简单的方式是用[miniconda](https://docs.conda.io/en/latest/miniconda.html)来安装Python语言, 通过它可以方便的创建虚拟环境, 管理Python版本和各种库类等. 先下载安装文件, 下载完成后执行脚本:
~~~bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
~~~

安装完毕后根据提示需要重启terminal或者手动加载才能使用: 
~~~bash
source ~/.bash_profile
~~~

验证安装:
~~~bash
conda -h
~~~

### 安装python3.6.5环境

创建虚拟环境并查看已经安装了的所有环境, 若已经创建过python 3.6.5的环境请略过:
~~~bash
conda create -n deeplearning python=3.6.5 matplotlib -y
conda-env list
~~~

开始使用所创建的环境deeplearning
~~~bash
source activate deeplearning
~~~

## 2. 依赖

在开发中我们会用到各种库类依赖, 需要先把他们安装起来
~~~bash
pip install scikit-image opencv-contrib-python numpy tensorflow==1.11.0 keras -i https://pypi.doubanio.com/simple/
~~~

在上面的命令中, 我们通过豆瓣的镜像源下载了scikit-image, opencv, numpy, tensorflow, keras这几个开发库. 不同的项目需要不同的库类, 根据需要安装即可