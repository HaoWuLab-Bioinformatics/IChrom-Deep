# IChrom-Deep
IChrom-Deep is an attention-based deep learning model for identifying chromatin interactions

## Framework
![image](https://github.com/HaoWuLab-Bioinformatics/IChrom-Deep/blob/main/Figure/Figure.png)

## Webserver
Given that the long prediction time of IChrom-Deep and to ensure user experience, we developed a simple version of IChrom-Deep's weberver and made it freely available at http://hwclnn.sdu.edu.cn/firstapp/ichrom-deep. In this version, we utilizes only the eight most important genomic features (performance is comparable to the full version, see paper for details). Of course, if you have high performance requirements, you can use the full version provided by github.

## Overview of dataset
The raw data files are from https://github.com/shwhalen/tf2.  
The folder "data" contains the DNA sequences in ".bed" format and labels. 
The file "feature.rar" contains other features, including genomic features, distance, CTCF motif and conservation score.  
The file "K562_sequence.rar" is an example file for the K562 cell line, containing DNA sequences in ".fasta" format and labels.  
The file "K562_genomics.rar" is an example file for the K562 cell line, containing genomic features and other features.  

## Overview of code
The folder "IChrom-Deep-focal" contains the code of IChrom-Deep using focal loss.
The folder "IChrom-Deep-transformer" contains the code of IChrom-Deep using transformers.
The code "data_load.py" is the code uesd to read data.  
The code "model.py" is the code uesd to declare model architecture.  
The code "feature_code.py" is the code uesd to encode the DNA sequence.  
The code "main.py" is the code uesd to train ICrhom-Deep and evaluate the performance of model. This code automatically calls the above three codes.  
The folder "generative model" is the code of DCGAN and VAE for promoting the study of chromatin interactions.  
The code "SHAP.py" is the code uesd for analyzing feature importance.

## Dependency
Python 3.6   
keras 2.3.1  
tensorflow 2.0.0  
sklearn  
numpy  
h5py 

## Usage of example files (K562 cell line)
Run the script as follows to compile and run ICrhom-Deep:  
`python main.py`  
Note that at least 64G memory may be required.
