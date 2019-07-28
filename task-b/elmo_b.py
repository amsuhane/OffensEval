import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import regularizers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder

def load_data():

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

    return data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c

def generate_embeddings(corpus):

	#embed = hub.Module("https://tfhub.dev/google/elmo/2")
	embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

	messages = corpus.tolist()

	# Reduce logging output.
	#tf.logging.set_verbosity(tf.logging.ERROR)

	with tf.Session() as session:
	  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	  message_embeddings = session.run(embed(messages))

	#elmo_index_message = {}
	#elmo_emb_index = {}
	X = []
	for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
	    #a = "{}".format(messages[i])
	    #b = [x for x in message_embedding]
	    #elmo_emb[i] = [x for x in message_embedding]
	    #elmo_index_message["{}".format(messages[i])] = i
	    #elmo_emb_index[i] = [x for x in message_embedding]
	    X.append([int(x) for x in message_embedding])

	return X

def make_model(X, one_hot_subtask):

        model = Sequential()
        model.add(Dense(128, input_dim=1024, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        filepath="weights-improvement-elmo-b-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        #fit the model
        model.fit(x=X, y=one_hot_subtask, validation_split=0.33, batch_size=64, epochs=100, callbacks=callbacks_list, verbose=0)
        plt.subplot(1, 2, 1)
        plt.plot(model.history.history['val_loss'], label = 'val_loss')
        plt.plot(model.history.history['loss'], label = 'loss')
        plt.title(" elmo (subtask_b) model (loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) \n elmo")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(model.history.history['acc'], label = 'acc')
        plt.plot(model.history.history['val_acc'], label = 'val_acc')
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()
        plt.savefig('elmo_b.png')

print(1)
data_training, corpus, sentences, one_hot_subtask_a, one_hot_subtask_b, one_hot_subtask_c = load_data()
print(2)
X = np.array(generate_embeddings(corpus))
print(3)
make_model(X, one_hot_subtask_b)
print(4)
