# IChrom-Deep
IChrom-Deep is an attention-based deep learning model for identifying chromatin interactions

## Framework
![image](https://github.com/HaoWuLab-Bioinformatics/IChrom-Deep/blob/main/Figure/Figure.png)

## Overview

".feather" files are raw data files from https://github.com/shwhalen/tf2.  
The folder "data" contains the sequences and labels.  
The file "feature.rar" contains other features.  
The file "index.txt" and "word2vec.txt" are benchmark files used to extract word2vec features.  
The file "feature_code.py" is the code used to extract word2vec features. Note that changing the variable 'cell_lines' on line 48 to extract the different cell lines.    
The file "data_process.py" is the code uesd to filter from raw data.  
The file "model.py" is the code of the IChrom-Deep model. Note that changing the variable 'cell_lines' on line 142 to predict the different cell lines.  
The file "cross_cell_all.py" is the code of the IChrom-Deep model on cross-cell lines validation. The variable 'cell_lines1' on line 172 is the training cell line and the variable 'cell_lines2' on line 173 is the testing cell line.  
The folder "generative model" is the code of DCGAN and VAE, using for promoting the study of chromatin interactions.  

## Dependency
Python 3.6   
keras  2.3.1  
tensorflow 2.0.0  
sklearn  
numpy  
h5py 

## Usage
First, you should extract features of sequences, you can run the script to extract word2vec-based features as follows:  
`python feature_code.py`  
Then run the script as follows to compile and run iPro-WAEL:  
`python model.py`  

