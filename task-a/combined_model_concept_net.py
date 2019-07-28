import numpy as np
import pandas as pd
import emoji
import string
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from gensim.models import KeyedVectors
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Concatenate
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import merge, Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import concatenate
import time

print("import done")

def load_data():

    data_training=pd.read_csv('../data/offenseval-training-v1.tsv',delimiter='\t',encoding='utf-8')

    # BASIC CLEANING
    data_training['tweet'] = data_training['tweet'].transform(lambda x: x.lower())
    corpus = data_training['tweet']
    corpus = corpus.transform(lambda x: emoji.demojize(x)) 
    corpus = corpus.transform(lambda x: x.replace(':', ' '))
    exclude = """!#$%&\()*+,-./:;<=>?[\\]^"'"-”“''`{|}~""" + '️—‘…–'
    corpus = corpus.transform(lambda x: ''.join(ch for ch in x if ch not in exclude))
    corpus = corpus.transform(lambda x: x.lower())

    # SPECIAL CHARACTER
    corpus = corpus.transform(lambda x: x.replace('★', 'star'))
    corpus = corpus.transform(lambda x: x.replace('☆', 'star'))
    corpus = corpus.transform(lambda x: x.replace('\xa0', u' '))
    corpus = corpus.transform(lambda x: x.replace('’', ' '))
    corpus = corpus.transform(lambda x: x.replace('£', 'dollar'))
    corpus = corpus.transform(lambda x: x.replace('_', ' '))

    alpha = string.ascii_lowercase + ' ' + '@' + '0123456789'

    # FINAL REFINING (JUST REMOVE THE REST)
    tempnot = dict()
    for i in corpus:
        for j in i:
            if j not in alpha:
                if j in string.ascii_uppercase:
                    print(j)
                if j in tempnot:
                    tempnot[j] +=1
                else:
                    tempnot[j] = 1

    exclude_refine = ''
    for i in tempnot:
        exclude_refine += i
    for i in exclude_refine:
        tempnot.pop(i)

    corpus = corpus.transform(lambda x: ''.join(ch for ch in x if ch not in exclude_refine))
    sentences = [i.split() for i in corpus]

    temp_a = {'OFF':1, 'NOT':0}
    temp_b = {'UNT':1, 'TIN':2}
    temp_c = {'IND':1, 'GRP':2, 'OTH':3}

    data_training['subtask_a'] = data_training['subtask_a'].map(temp_a)
    data_training['subtask_b'] = data_training['subtask_b'].map(temp_b)
    data_training['subtask_c'] = data_training['subtask_c'].map(temp_c)

    data_training = data_training.fillna(0)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(data_training['subtask_a']).reshape(-1, 1))
    one_hot_subtask_a = enc.transform(np.array(data_training['subtask_a']).reshape(-1, 1)).toarray()

    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    stopWords.add('@user')
    stopWords.add('url')
    corpus_no_stop = [[i for i in j.split() if i not in stopWords] for j in corpus]
    nostoplen = [len(i) for i in corpus_no_stop]

    corpus = corpus_no_stop

    label = [i for i in data_training['subtask_a']]

    print("data loaded")

    with open('abusive_words.txt', 'r') as f:
        x = f.readlines()

    abusive_words = [i[:-1] for i in x[1:]]

    with open('abusive_words_2.txt', 'r') as f:
        x = f.readlines()

    abusive_words += [i[:-1] for i in x]

    inlist_abu, inlist_noabu = 0, 0
    abu_list, noabu_list = set(), set()
    for i in range(len(corpus)):
        for j in corpus[i]:
            if j in abusive_words:
                if label[i] == 1:
                    abu_list.add(j)
                else:
                    noabu_list.add(j)

    r_abu = abu_list.copy()
    for i in abu_list:
        if i in noabu_list:
            r_abu.remove(i)

    abus, noabus = 0, 0
    for i in range(len(corpus)):
        for j in r_abu:
            if j in corpus[i]:
                if label[i]==1:
                    abus+=1
                else:
                    noabus+=1
                break

    count = dict()
    for i in r_abu:
        count[i] = 0
    for i in corpus:
        for j in i:
            if j in r_abu:
                count[j] += 1

    remove_from_abu = ['baptist', 'colored', 'diseases', 'dive', 'dopey','itch',
     'jesuschrist', 'nasty', 'redneck', 'robber',
     'rubbish',
     'schizo',
     'servant']
    for i in remove_from_abu:
        r_abu.remove(i)

    print("abu dict done")

    X_unpadded = np.load('../Flair/outfile.npy')

    print("loaded X")

    temp = [len(i) for i in X_unpadded]

    contains_abu_word = []
    for i in corpus:
        flag = 0
        for j in i:
            if j in r_abu:
                flag = 1
        contains_abu_word.append(flag)

    max_sentence_len = 50
    padded = pad_sequences(X_unpadded, padding='pre', dtype='object', maxlen=max_sentence_len, truncating="post")
    X_word_embeddings_padded = padded

    print("padding done")

    X_1 = []
    for i in corpus:
        temp = []
        for j in r_abu:
            if j in i:
                temp.append(1)
            else:
                temp.append(0)
        X_1.append(temp)

    X_2 = X_word_embeddings_padded
    Y = one_hot_subtask_a
    X_1 = np.array(X_1)

    return X_1, X_2, Y


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul


def make_model():

    model_1_input = Input(shape=(124, ))
    model_1_dense = Dense(2, activation="sigmoid")(model_1_input)
    
    inputs_layer_2 = Input(shape=(TIME_STEPS, INPUT_DIM, ))
    lstm_out_2 = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer_2)
    attention_mul_2 = attention_3d_block(lstm_out_2)
    attention_mul_2 = Flatten()(attention_mul_2)
    output_2 = Dense(2, activation="sigmoid")(attention_mul_2)
    #model_2_dense = Model(inputs=[inputs_layer_2], outputs=output_2)

    model_merge = Concatenate(axis=-1)([model_1_dense, output_2])
    model_out = Dense(2, activation="tanh")(model_merge)

    model = Model(inputs=[model_1_input, inputs_layer_2], outputs=model_out)

    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    filepath="weights-improvement-a-att-abu-conceptnet-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit([X_1, X_2], Y,validation_split=0.33, batch_size=64, epochs=30, callbacks=callbacks_list, verbose=0) 

    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['val_loss'], label = 'val_loss')
    plt.plot(model.history.history['loss'], label = 'loss')
    plt.title("att-abu-words \n concept-net")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['acc'], label = 'acc')
    plt.plot(model.history.history['val_acc'], label = 'val_acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('abu-att-conceptnet')

print(1)
X_1, X_2, Y = load_data()
print(2)
TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 50, len(X_2[0][0]), 10, False
make_model()
print(3)

    


