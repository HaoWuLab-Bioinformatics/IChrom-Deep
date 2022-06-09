from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Dropout, Layer, Reshape, Concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from keras.models import load_model
from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from keras import backend as K
from keras import initializers
import heapq
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
from keras.layers import Bidirectional
from keras.layers import LSTM, BatchNormalization
class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def nor_train_test(x_train, x_test):
    x = np.concatenate((x_train, x_test), axis=0)
    x = noramlization(x)
    x_train = x[0:len(x_train)]
    x_test = x[len(x_train):]
    return x_train, x_test


def construct_model(input_shape_x, input_shape_y, input_shape_genomics):
    input_data_x = Input(shape=input_shape_x)
    input_data_y = Input(shape=input_shape_y)
    input_data_genomics = Input(shape=input_shape_genomics)

    x = Conv1D(32, kernel_size=50, padding='same', activation='relu')(input_data_x)
    x = MaxPooling1D(pool_size=25, strides=25)(x)
    x = Dropout(0.5)(x)

    y = Conv1D(32, kernel_size=50, padding='same', activation='relu')(input_data_y)
    y = MaxPooling1D(pool_size=25, strides=25)(y)
    y = Dropout(0.5)(y)

    merge1 = Concatenate(axis=1)([x, y])
    merge1 = BatchNormalization()(merge1)
    merge1 = Dropout(0.5)(merge1)
    merge1 = AttLayer(50)(merge1)

    merge2 = Concatenate(axis=1)([merge1, input_data_genomics])
    merge2 = BatchNormalization()(merge2)
    merge2 = Dropout(0.5)(merge2)
    merge2 = Dense(32, activation='relu')(merge2)
    z = Dense(1, activation='sigmoid')(merge2)
    model = Model([input_data_x, input_data_y, input_data_genomics], z)
    return model


def get_name():
    name = []
    f = open('data/' + cell_lines + '/x.bed')
    for i in f.readlines():
        if i[0] != ' ':
            name.append(i.strip().split('\t')[0])
    f.close()
    return name


''' focal loss '''


def binary_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


# data load
cell_lines = 'HeLaS3'
genomics = np.loadtxt('feature/' + cell_lines + '/genomics.csv', delimiter=',')
print(genomics.shape)
label = np.loadtxt('data/' + cell_lines + '/label.txt')
x_word2vec = np.load('feature/' + cell_lines + '/x_word2vec.npy')
y_word2vec = np.load('feature/' + cell_lines + '/y_word2vec.npy')
genomics = np.loadtxt('feature/' + cell_lines + '/genomics.csv', delimiter=',')[:, 4:]
'''genomics1 = np.loadtxt('feature/' + cell_lines + '/genomics.csv', delimiter=',')[:, 0:4]
genomics2 = np.loadtxt('feature/' + cell_lines + '/genomics.csv', delimiter=',')[:, 6:]
genomics = np.concatenate((genomics1, genomics2),axis=1)'''
# genomics = np.expand_dims(genomics, 2)


# input shape
input_shape_x = (4997, 8)
input_shape_y = (4997, 8)
input_shape_genomics = (len(genomics[0]), )


# model construction

filepath = "model.hdf5"

chr_name = get_name()
k = 10
i = 0
gkf = GroupKFold(n_splits=k)
auroc = np.zeros(k)
auprc = np.zeros(k)
mcc = np.zeros(k)
precision = np.zeros(k)
recall = np.zeros(k)
f1 = np.zeros(k)
bacc = np.zeros(k)
acc = np.zeros(k)
MAX_EPOCH = 20
BATCH_SIZE = 50

learning_rate = 0.001

for index_train, index_test in gkf.split(label, groups=chr_name):
    # print(len(index_train))
    print(len(index_test))

for index_train, index_test in gkf.split(label, groups=chr_name):
    # print(len(index_train))
    print(len(index_test))
    feature_train1 = x_word2vec[index_train, :, :]
    feature_train2 = y_word2vec[index_train, :, :]
    feature_train3 = genomics[index_train]
    label_train = label[index_train]

    # 不平衡的测试集
    feature_test1 = x_word2vec[index_test, :, :]
    feature_test2 = y_word2vec[index_test, :, :]
    feature_test3 = genomics[index_test]
    label_test = label[index_test]

    # 平衡的测试集
    feature_test1_pos = feature_test1[label_test == 1]
    feature_test2_pos = feature_test2[label_test == 1]
    feature_test3_pos = feature_test3[label_test == 1]
    pos_num = len(feature_test1_pos)
    print(pos_num)
    feature_test1_neg = feature_test1[label_test == 0][0:pos_num, ]
    feature_test2_neg = feature_test2[label_test == 0][0:pos_num, ]
    feature_test3_neg = feature_test3[label_test == 0][0:pos_num, ]

    feature_test1_balance = np.concatenate((feature_test1_pos, feature_test1_neg), axis=0)
    feature_test2_balance = np.concatenate((feature_test2_pos, feature_test2_neg), axis=0)
    feature_test3_balance = np.concatenate((feature_test3_pos, feature_test3_neg), axis=0)
    label_test_balance = np.concatenate((np.ones(pos_num), np.zeros(pos_num)), axis=0)

    model = construct_model(input_shape_x, input_shape_y, input_shape_genomics)
    # print(model.summary())
    model.compile(loss=[binary_focal_loss(alpha=0.75, gamma=3)], optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    model.fit(x=[feature_train1, feature_train2, feature_train3], y=label_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH)

    # 不平衡测试集评估
    model_pro = model.predict([feature_test1, feature_test2, feature_test3])
    model_class = np.around(model_pro)
    bacc[i] = balanced_accuracy_score(label_test, model_class)

    # 平衡测试集评估
    model_pro = model.predict([feature_test1_balance, feature_test2_balance, feature_test3_balance])
    model_class = np.around(model_pro)
    fpr, tpr, _ = roc_curve(label_test_balance, model_pro)
    auroc[i] = auc(fpr, tpr)
    acc[i] = accuracy_score(label_test_balance, model_class)
    mcc[i] = matthews_corrcoef(label_test_balance, model_class)
    precision[i], recall[i], f1[i], _ = precision_recall_fscore_support(label_test_balance, model_class,
                                                                        average='binary')

    print('bacc:', bacc)
    print('auc:', auroc)
    print('acc:', acc)
    print('mcc:', mcc)
    print('precision:', precision)
    print('recall', recall)
    print('fscore:', f1)

    i += 1

print('10-fold cross-validation')
print('bacc:', bacc.mean())
print('auc:', auroc.mean())
print('acc:', acc.mean())
print('mcc:', mcc.mean())
print('precision:', precision.mean())
print('recall', recall.mean())
print('fscore:', f1.mean())
print(bacc.mean(), auroc.mean(), acc.mean(), mcc.mean(), precision.mean(), recall.mean(), f1.mean())

print('----------------------------------------------------------------')
for i in range(10):
    print(bacc[i], auroc[i], acc[i], mcc[i], precision[i], recall[i], f1[i])

print(bacc.mean(), auroc.mean(), acc.mean(), mcc.mean(), precision.mean(), recall.mean(), f1.mean())