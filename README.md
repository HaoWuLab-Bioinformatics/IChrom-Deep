# IChrom-Deep
IChrom-Deep is an attention-based deep learning model for identifying chromatin interactions

## Framework
![image](https://github.com/HaoWuLab-Bioinformatics/IChrom-Deep/blob/main/Figure/Figure.png)

## Overview
 
The folder "data" is the data of the promoter, containing the sequences and labels of the independent tesst sets and training sets.  
The folder "EPdata" is the data of the enhancer and promoter, containing the sequences of the independent tesst sets and training sets. The first half of each file is labeled 1, and the second half is labeled 0.  
The file "index.txt" and "word2vec.txt" are benchmark files used to extract word2vec features.  
The file "model.py" is the code of the IChrom-Deep model.  
The file "main.py" is the code of the entire model and will import RF, CNN and Weighted_average.  
The file "feature_code.py" is the code used to extract word2vec features.  

## Dependency
Python 3.6   
keras  2.3.1  
tensorflow 2.0.0  
sklearn  
numpy  
h5py 

## Usage
First, you should extract features of promoters, you can run the script to extract word2vec-based features as follows:  
`python feature_code.py`  
The extraction of other features is done using iLearnPlus [1].  
Then run the script as follows to compile and run iPro-WAEL:  
`python main.py`  
Note that the variable 'cell_lines' needs to be manually modified to change the predicted cell line.  
## Reference
1. Chen Z, Zhao P, Li C, et al. ILearnPlus: A comprehensive and automated machine-learning platform for nucleic acid and protein sequence analysis, prediction and visualization. Nucleic Acids Res. 2021; 49:1–19
