import numpy as np
import pandas as pd
import emoji
import string
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from gensim.models import KeyedVectors
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

def load_data():
    
    print('a')
    fasttext_vectors = KeyedVectors.load_word2vec_format('../embeddings/fasttext_wiki.en.vec')
    data_training=pd.read_csv('../data/offenseval-training-v1.tsv',delimiter='\t',encoding='utf-8')
    print('b')
    
    #Remove if not offensive
    not_off = []
    for i in range(len(data_training['subtask_a'])):
        if data_training['subtask_a'][i]=='NOT':
            not_off.append(i)
    data_training = data_training.drop([i for i in not_off])
    data_training = data_training.reset_index(drop=True)
    
    # BASIC CLEANING
    data_training['tweet'] = data_training['tweet'].transform(lambda x: x.lower())
    corpus = data_training['tweet']
    corpus = corpus.transform(lambda x: emoji.demojize(x)) 
    corpus = corpus.transform(lambda x: x.replace(':', ' '))
    exclude = """!#$%&\()*+,-./:;<=>?[\\]^"'"-”“''`{|}~""" + '️—‘…–'
    corpus = corpus.transform(lambda x: ''.join(ch for ch in x if ch not in exclude))
    corpus = corpus.transform(lambda x: x.lower())
    print('c')
    
    # SPECIAL CHARACTER
    corpus = corpus.transform(lambda x: x.replace('★', 'star'))
    corpus = corpus.transform(lambda x: x.replace('☆', 'star'))
    corpus = corpus.transform(lambda x: x.replace('\xa0', u' '))
    corpus = corpus.transform(lambda x: x.replace('’', ' '))
    corpus = corpus.transform(lambda x: x.replace('£', 'dollar'))
    corpus = corpus.transform(lambda x: x.replace('_', ' '))

    alpha = string.ascii_lowercase + ' ' + '@' + '0123456789'
    print('d')
    
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
    
    print('e')
    temp_a = {'OFF':1}
    temp_b = {'UNT':1, 'TIN':2}
    temp_c = {'IND':1, 'GRP':2, 'OTH':3}

    data_training['subtask_a'] = data_training['subtask_a'].map(temp_a)
    data_training['subtask_b'] = data_training['subtask_b'].map(temp_b)
    data_training['subtask_c'] = data_training['subtask_c'].map(temp_c)
    
    data_training = data_training.fillna(0)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(data_training['subtask_a']).reshape(-1, 1))
    one_hot_subtask_a = enc.transform(np.array(data_training['subtask_a']).reshape(-1, 1)).toarray()
    enc.fit(np.array(data_training['subtask_b']).reshape(-1, 1))
    one_hot_subtask_b = enc.transform(np.array(data_training['subtask_b']).reshape(-1, 1)).toarray()
    enc.fit(np.array(data_training['subtask_c']).reshape(-1, 1))
    one_hot_subtask_c = enc.transform(np.array(data_training['subtask_c']).reshape(-1, 1)).toarray()
    
    return fasttext_vectors, data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c


def sentence_word_to_vector():
    
    print('a')
    #Find words not in vocab but in corpus
    not_in_vocab = dict()
    for i in sentences:
        for j in i:
            if j not in fasttext_vectors.vocab:
                if j not in not_in_vocab.keys():
                    not_in_vocab[j]=1
                else:
                    not_in_vocab[j]+=1
    print('b')
    #Filter those words
    corpus_no_unknown = []
    for i in corpus:
        corpus_no_unknown.append(list(filter(lambda x: x not in not_in_vocab, i.split())))
    print('c')
    X_fastext = [[fasttext_vectors[j] for j in i] for i in corpus_no_unknown]
    print('d')
    #Pad sentences
    X_word_embeddings_padded_a = pad_sequences(X_fastext, padding='post', dtype='object')
    max_sentence_len = len(X_word_embeddings_padded_a[0])
    print('e')
    return X_word_embeddings_padded_a, max_sentence_len


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


def make_model(X, one_hot_subtask_a):

    inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
    lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(2, activation="sigmoid")(attention_mul)
    model = Model(input=[inputs_layer], output=output)

    #complie model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #checkpoint
    filepath="weights-improvement-attention-elmo-b-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print(model.summary())

    #fit the model
    model.fit(x=X, y=one_hot_subtask_a, validation_split=0.33,  batch_size=64, epochs=50, callbacks=callbacks_list, verbose=1)
    #model.fit(x=X, y=one_hot_subtask_a, validation_split=0.33,  batch_size=64, epochs=5, verbose=0)
    
    print("val loss: \n")
    print(model.history.history['val_loss'])
    print("loss: \n")
    print(model.history.history['loss'])
    print("val acc: \n")
    print(model.history.history['val_acc'])
    print("acc: \n")
    print(model.history.history['acc'])
    #plot
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['val_loss'], label = 'val_loss')
    plt.plot(model.history.history['loss'], label = 'loss')
    plt.title("elmo - Attention (subtask_a) model (loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) \n fasttext pre-trained-vectors")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['acc'], label = 'acc')
    plt.plot(model.history.history['val_acc'], label = 'val_acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('elmo attention  a | fasttext')

print(1)
fasttext_vectors, data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c = load_data()
print(2)
X, max_sentence_len = sentence_word_to_vector()
print(3)
TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = max_sentence_len, 300, 10, False
print(4)
make_model(X, one_hot_subtask_b)


