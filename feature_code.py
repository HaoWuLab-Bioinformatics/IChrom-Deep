import numpy as np
import torch
import torch.nn as nn
from keras.layers.convolutional import Conv2D

def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence.append(line.strip('\n'))
    f.close()
    k = 4
    kmer_list = []
    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            ind = index.index(sequence[number][i:i + k])
            seq.append(ind)
        kmer_list.append(seq)

    feature_word2vec = np.zeros((len(kmer_list), 5000-k+1, 8))
    for number in range(len(kmer_list)):
        #print(number)
        for i in range(len(kmer_list[number])):
            kmer_index = kmer_list[number][i]
            feature_word2vec[number,i,:] = word2vec[kmer_index]

    return feature_word2vec


'''cell_lines = 'NHEK'
sets = 'train'
filename = 'data/' + cell_lines + '/' +sets + '/data.fasta'
# filename = 'data/' + cell_lines + '/' + element + '/test/test.fasta'
f = open('index_promoters.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')
word2vec = np.loadtxt('word2vec_promoters.txt')
feature_word2vec = word_embedding(filename, index, word2vec)
feature_word2vec = np.array(feature_word2vec)
print(feature_word2vec.shape)
np.savetxt('feature/' + cell_lines + '/' +sets + '/word2vec.txt', feature_word2vec)'''


cell_lines = 'IMR90'
filename = 'data/' + cell_lines + '/y.fasta'

f = open('index.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')
word2vec = np.loadtxt('word2vec.txt')

feature_word2vec = word_embedding(filename, index, word2vec)
print(feature_word2vec.shape)
np.save('feature/' + cell_lines + '/y_word2vec.npy', feature_word2vec)
