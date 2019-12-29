import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Conv2D,LeakyReLU, Flatten, K,Dense,Reshape,Activation,Conv2DTranspose,Embedding,Dropout,LSTM
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import np_utils
import re

filename="hbshort.txt"
with open(filename,encoding="utf-8-sig") as f:
    text =f.read()

seq_length=20
start_story='| '*seq_length

text=text.lower()
text=start_story+text
text=text.replace('\n\n\n\n\n',start_story)
text=text.replace('\n',' ')
text=re.sub(' +','. ',text).strip()
text=text.replace('..','.')
text=re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_{|}~])',r' \1 ',text)
text=re.sub('\s{2,}',' ',text)

tokenizer=Tokenizer(char_level=False,filters='')
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1
token_list=tokenizer.texts_to_sequences([text])[0]


p=1

def generate_sequences(token_list,step):
    X=[]
    y=[]

    for i in range(0,len(token_list)-seq_length,step):
        X.append(token_list[i:i+seq_length])
        y.append(token_list[i+seq_length])

    y=np_utils.to_categorical(y,num_classes=total_words)
    num_seq=len(X)

    print("number of sequences=",num_seq,"\n")
    return X,y,num_seq

step=1
seq_length=20
X,y,num_seq=generate_sequences(token_list,step)
X=np.array(X)
y=np.array(y)

p=2

embedding_size=100
n_units=256

text_in=Input(shape=(None,))
x=Embedding(total_words,embedding_size)(text_in)
x=LSTM(n_units,return_sequences=True)(x)
x=Dropout(0.2)(x)
x=LSTM(n_units)(x)
x=Dropout(0.2)(x)
text_out=Dense(total_words,activation='softmax')(x)

model=Model(text_in,text_out)
opti=RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opti)

epochs=5
batch_size=32
model.fit(X,y,epochs=epochs,batch_size=batch_size,shuffle=True)

model.save("dagmymodel.model")