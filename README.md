# IChrom-Deep
IChrom-Deep is an attention-based deep learning model for identifying chromatin interactions

## Framework
![image](https://github.com/HaoWuLab-Bioinformatics/IChrom-Deep/blob/main/Figure/Figure.png)

## Overview
 
The folder "data" is the data of the promoter, containing the sequences and labels of the independent tesst sets and training sets.  

The file "index.txt" and "word2vec.txt" are benchmark files used to extract word2vec features.  
The file "feature_code.py" is the code used to extract word2vec features. Note that changing the variable 'cell_lines' on line 48 to extract the different cell lines.    
The file "model.py" is the code of the IChrom-Deep model. Note that changing the variable 'cell_lines' on line 142 to predict the different cell lines.  
The file "cross_cell_all.py" is the code of the IChrom-Deep model on cross-cell lines validation. The variable 'cell_lines1' on line 172 is the training cell line and the variable 'cell_lines2' on line 173 is the testing cell line.  


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
