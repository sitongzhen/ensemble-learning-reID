
# Installation

I use Python 2.7 and Pytorch 1.0.1. For installing Pytorch, follow the [official guide](http://pytorch.org/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```
Then clone the repository:

'''bash

git clone https://github.com/sitongzhen/ensemble-learning-reID.git

'''

# Dataset Preparation

Transformed dataset has following features
- All used images, including training and testing images, are inside the same folder named `images`
- The train/val/test partitions are recorded in a file named `partitions.pkl`


You can download what I have transformed for the project from (https://pan.baidu.com/s/1nvOhpot) [market1501] or (https://pan.baidu.com/s/1hsB0pIc) [cuhk03] or (https://pan.baidu.com/s/1miIdEek) [duke]. Otherwise, you can download the original dataset and transform it using my script, described below.

```bash
python script/dataset/transform_market1501.py 
or 
python script/dataset/transform_cuhk03.py 
or
python script/dataset/transform_duke.py
```

The project requires you to configure the dataset paths. In `package/dataset/__init__.py`.


## Train Model

```bash
python ensemble_training.py
```

