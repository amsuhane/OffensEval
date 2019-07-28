import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras import regularizers
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
import emoji
import string


def load_data():
    
    print("loading data")
    
    print('load embed')

    t = time.time()
    word_embed = dict()
    with open('../embeddings/conceptnet/numberbatch-en-17.06.txt') as f:
        f1 = f.readlines()
        for x in f1:
            word_embed[x.split()[0]] = [float(i) for i in x.split()[1:]]

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
    
    print(time.time() - t)
    
    X_unpadded = []
    for i in corpus:
        temp = []
        for j in i:
            if j in word_embed:
                temp.append(word_embed[j])
        X_unpadded.append(temp)
    
    print("data loaded")
    
    return X_unpadded, one_hot_subtask_a


def padder(X_unpadded):
    
    print("start padding")
    t = time.time()
    max_sentence_len = 50
    padded = pad_sequences(X_unpadded, padding='pre', dtype='object', maxlen=max_sentence_len, truncating='post')
    X_word_embeddings_padded = padded
    print(time.time() - t)
    print("padding done")
    
    return X_word_embeddings_padded


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


def make_model(X, Y):
    
    inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
    lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(2, activation="sigmoid")(attention_mul)
    model = Model(inputs=[inputs_layer], outputs=output)

    #complie model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #checkpoint
    filepath="weights-improvement-final-2-conceptnet-a-attention-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    #print(model.summary())
    
    #fit the model
    model.fit(x=X, y=Y,	batch_size=64, epochs=100, callbacks=callbacks_list, verbose=0)
    
    #plot
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['loss'], label = 'loss')
    plt.title("Attention conceptnet")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['acc'], label = 'acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    #plt.show()
    plt.savefig('Final_2_Bi_LSTM_a attention | conceptnet')

    for i in model.history.history['acc']:
        print(i)

X_unpadded, Y = load_data()
X = padder(X_unpadded)
max_sentence_len = 50
TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = max_sentence_len, len(X[0][0]), 10, False
#make_model(X, Y)
from keras.models import load_model
model_conceptnet = load_model('weights-improvement-final-2-conceptnet-a-attention-99-0.84.hdf5')
filepath="weights-improvement-final-2-conceptnet-a-attention-100-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model_conceptnet.fit(x=X, y=Y, batch_size=64, epochs=50, callbacks=callbacks_list, verbose=0)    
