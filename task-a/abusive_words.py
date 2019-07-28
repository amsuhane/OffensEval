from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import pandas as pd
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
import emoji
import string


def load_data():
    print(1)
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
    print(1)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(data_training['subtask_a']).reshape(-1, 1))
    one_hot_subtask_a = enc.transform(np.array(data_training['subtask_a']).reshape(-1, 1)).toarray()

    print(1)
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    stopWords.add('@user')
    stopWords.add('url')
    corpus_no_stop = [[i for i in j.split() if i not in stopWords] for j in corpus]
    nostoplen = [len(i) for i in corpus_no_stop]

    print(2)
    corpus = corpus_no_stop

    label = [i for i in data_training['subtask_a']]

    with open('../abusive_words/abusive_words.txt', 'r') as f:
        x = f.readlines()

    abusive_words = [i[:-1] for i in x[1:]]

    with open('../abusive_words/abusive_words_2.txt', 'r') as f:
        x = f.readlines()

    abusive_words += [i[:-1] for i in x]

    print(2)
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
    print(3)

    X_unpadded = np.load('../Flair/outfile.npy')
    print(4)

    temp = [len(i) for i in X_unpadded]

    contains_abu_word = []
    for i in corpus:
        flag = 0
        for j in i:
            if j in r_abu:
                flag = 1
        contains_abu_word.append(flag)
    print(5)

    max_sentence_len = 150
    padded = pad_sequences(X_unpadded, padding='pre', dtype='object', maxlen=max_sentence_len, truncating="post")
    X_word_embeddings_padded = padded
    print(6)

    X_1 = []
    for i in corpus:
        temp = []
        for j in r_abu:
            if j in i:
                temp.append(1)
            else:
                temp.append(0)
        X_1.append(temp)
    print(7)

    X_2 = X_word_embeddings_padded
    Y = one_hot_subtask_a
    X_1 = np.array(X_1)

    return X_1, X_2, Y


def make_model():

    print(8)

    model_1_input = Input(shape=(124, ))
    model_1_dense = Dense(2, activation="sigmoid")(model_1_input)

    model_2_input = Input(shape=(150, 1068, ))
    model_2_lstm = Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(model_2_input)
    model_2_dense = Dense(2, activation="sigmoid")(model_2_lstm)

    model_merge = Concatenate(axis=-1)([model_1_dense, model_2_dense])
    model_out = Dense(2, activation="tanh")(model_merge)

    model = Model(inputs=[model_1_input, model_2_input], outputs=model_out)

    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    filepath="weights-improvement-final-a-2-abu-word-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit([X_1, X_2], Y, batch_size=64, epochs=30, callbacks=callbacks_list, verbose=0) 
    
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['loss'], label = 'loss')
    plt.title("abu-words \n elmo|fasttext pre-trained-vectors")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['acc'], label = 'acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('final-abu-2-words-elmo-fasttext')
    
    for i in model.history.history['acc']:
        print(i)
    
X_1, X_2, Y = load_data()
make_model()




