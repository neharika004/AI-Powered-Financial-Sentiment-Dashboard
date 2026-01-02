import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import pickle
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
nltk.download('words')
from nltk.corpus import stopwords
from nltk.corpus import words
from textblob import TextBlob
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tokenize_text(text):
    tokens = []
    stop_words = set(stopwords.words('english'))
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            #if len(word) < 0:
            if len(word) <= 0:
                continue
            tokens.append(word.lower())

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens



def split_input(sequence):
    return sequence[:-1], tf.reshape(sequence[1:], (-1,1))

df = pd.read_csv(r'C:\Users\sharm\Desktop\EDAI\all-data.csv',delimiter=',',encoding='latin-1',header=None)
df.head()

df = df.rename(columns={0:'sentiment',1:'Message'})
df.info()

#To add indexing and word count 
df.index = range(df.shape[0])  # Automatically matches the DataFrame's current size.

df['Message'].apply(lambda x: len(x.split(' '))).sum()

sentiment  = {'positive': 0,'neutral': 1,'negative':2} 
df.sentiment = [sentiment[item] for item in df.sentiment] 


df['Message'] = df['Message'].apply(cleanText)

train, test = train_test_split(df, test_size=0.000001 , random_state=42)

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)


# The maximum number of words to be used. (most frequent)
max_features = 500000 

# Max number of words for each.
MAX_SEQUENCE_LENGTH = 50


tokenizer = Tokenizer(num_words=max_features, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Message'].values)

X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)



d2v_model = Doc2Vec(dm=1, dm_mean=1, vector_size=20, window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    d2v_model.alpha -= 0.002
    d2v_model.min_alpha = d2v_model.alpha

# save the vectors in a new matrix
embedding_matrix = np.zeros((len(d2v_model.wv)+ 1, 20))

for i, vec in enumerate(d2v_model.dv.vectors):
    while i in vec <= 1000:
          embedding_matrix[i]=vec


model = Sequential()

# emmbed word vectors
model.add(Embedding(len(d2v_model.wv)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True))

#hidden layer   
model.add(LSTM(50,return_sequences=False))
model.add(Dense(3,activation="softmax"))


# output model skeleton
model.summary()
model.compile(optimizer='adam',loss="binary_crossentropy", metrics=['accuracy'])

Y = pd.get_dummies(df['sentiment']).values
X_train_val, X_test, y_train_val, Y_test = train_test_split(X,Y, test_size=0.10, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, y_train_val, test_size=0.02, random_state=42)

print("Training data shape: X_train={},y_train={}".format(X_train.shape,Y_train.shape))
print("Testing data shape: X_test={},y_test={}".format(X_test.shape,Y_test.shape))
print("Validation data shape: X_val={},y_val={}".format(X_val.shape,Y_val.shape))


batch_size = 32
history=model.fit(X_train, Y_train,epochs = 50, batch_size=batch_size, verbose = 2, validation_data=(X_val,Y_val))

print(history.history)

# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=2)
_, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))

score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)

print("Test data score: %.2f" % (score))
print("Test data accuracy: %.2f" % (acc))

model.save('./financial_Pred_Model.h5',save_format='h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.save('tensor.npy',X)

# plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('loss.png')
plt.show()


# plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('accuracy.png')
plt.show()

print("End of training script!!")
