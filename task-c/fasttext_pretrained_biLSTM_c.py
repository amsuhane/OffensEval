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

def plot_model():
    plt.subplot(1, 2, 1)
    plt.plot(model_Bi_LSTM.history.history['val_loss'], label = 'val_loss')
    plt.plot(model_Bi_LSTM.history.history['loss'], label = 'loss')
    plt.title("BiLSTM (subtask_b) model (loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) \n GloVe vectors")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model_Bi_LSTM.history.history['categorical_accuracy'], label = 'acc')
    plt.plot(model_Bi_LSTM.history.history['val_categorical_accuracy'], label = 'val_acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('Bi_LSTM_b | glove_6B_300d')

def load_data():

    fasttext_vectors = KeyedVectors.load_word2vec_format('../embeddings/fasttext_wiki.en.vec')
    data_training=pd.read_csv('../data/offenseval-training-v1.tsv',delimiter='\t',encoding='utf-8')

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
    enc.fit(np.array(data_training['subtask_b']).reshape(-1, 1))
    one_hot_subtask_b = enc.transform(np.array(data_training['subtask_b']).reshape(-1, 1)).toarray()
    enc.fit(np.array(data_training['subtask_c']).reshape(-1, 1))
    one_hot_subtask_c = enc.transform(np.array(data_training['subtask_c']).reshape(-1, 1)).toarray()

    data_training['tweet'] = data_training['tweet'].transform(lambda x: x.lower())
    corpus = data_training['tweet']
    exclude = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    corpus = corpus.transform(lambda x: ''.join(ch for ch in x if ch not in exclude))
    sentences = [i.split() for i in corpus]

    return fasttext_vectors, data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c

def sentence_word_to_vector():

    #Find words not in vocab but in corpus
    not_in_vocab = dict()
    for i in sentences:
        for j in i:
            if j not in fasttext_vectors.vocab:
                if j not in not_in_vocab.keys():
                    not_in_vocab[j]=1
                else:
                    not_in_vocab[j]+=1

    #Filter those words
    corpus_no_unknown = []
    for i in corpus:
        corpus_no_unknown.append(list(filter(lambda x: x not in not_in_vocab, i.split())))

    X_fastext = [[fasttext_vectors[j] for j in i] for i in corpus_no_unknown]

    #Pad sentences
    max_sentence_len = np.max([len(i) for i in X_fastext])
    padded = pad_sequences(X_fastext, padding='post', dtype='object')
    X_word_embeddings_padded = padded

    return X_word_embeddings_padded, max_sentence_len

def make_model(X, max_sentence_len, one_hot_subtask_c):

    # create model
    model_Bi_LSTM = Sequential()
    model_Bi_LSTM.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), input_shape=(max_sentence_len, 300), merge_mode='concat'))
    #model_Bi_LSTM_a.add(Bidirectional(LSTM(10, kernel_regularizer=regularizers.l2(0.01)), input_shape=(81, 300), merge_mode='concat'))
    model_Bi_LSTM.add(Dense(4, activation='softmax'))
    #model_Bi_LSTM_a.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.002), metrics=['categorical_accuracy'])

    #complie model
    model_Bi_LSTM.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    #checkpoint
    filepath="weights-improvement-c-biLSTM-fasttextpretrained-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    #fit the model
    model_Bi_LSTM.fit(x=X, y=one_hot_subtask_c, validation_split=0.33,	batch_size=64, epochs=50, callbacks=callbacks_list, verbose=0)

    #plot
    #plot_model()
    plt.subplot(1, 2, 1)
    plt.plot(model_Bi_LSTM.history.history['val_loss'], label = 'val_loss')
    plt.plot(model_Bi_LSTM.history.history['loss'], label = 'loss')
    plt.title("BiLSTM (subtask_c) model (loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) \n fasttext pre-trained-vectors")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model_Bi_LSTM.history.history['acc'], label = 'acc')
    plt.plot(model_Bi_LSTM.history.history['val_acc'], label = 'val_acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('Bi_LSTM_c | fasttext_trained')

fasttext_vectors, data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c = load_data()
X, max_sentence_len = sentence_word_to_vector()
make_model(X, max_sentence_len, one_hot_subtask_c)
