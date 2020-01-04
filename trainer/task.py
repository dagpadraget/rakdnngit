#
# Copyright DLUAB 2020
#
# History
#
# DATE          SIGN                COMMENT
# 20200104      Dag Lundström       Created
#


# *********************************************************************************************************************
# Library imports
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Conv2D,LeakyReLU, Flatten, Dense,Reshape,Activation,Conv2DTranspose,Embedding,Dropout,LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import np_utils
import re
# *********************************************************************************************************************


# *********************************************************************************************************************
# Load data from file
def load_data(datapath):
    filename=datapath+"\\hbshort.txt"
    with open(filename,encoding="utf-8-sig") as f:
        text =f.read()

    seq_length=3 # sequence around sentence
    step=1 # prediction step forward

    start_story='| '*seq_length

    text=text.lower()
    text=start_story+text
    text=text.replace('\n\n\n\n\n',start_story)
    text=text.replace('\n',' ')
    #text=re.sub(' +','. ',text).strip()
    text=text.replace('..','.')
    text=re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_{|}~])',r' \1 ',text)
    text=re.sub('\s{2,}',' ',text)

    tokenizer=Tokenizer(char_level=False,filters='')
    tokenizer.fit_on_texts([text])
    total_words=len(tokenizer.word_index)+1
    token_list=tokenizer.texts_to_sequences([text])[0]

    return token_list, seq_length,step,total_words,start_story,tokenizer
# *********************************************************************************************************************


# *********************************************************************************************************************
def generate_sequences(token_list,step,seq_length,total_words):
    X=[]
    y=[]

    for i in range(0,len(token_list)-seq_length,step):
        X.append(token_list[i:i+seq_length])
        y.append(token_list[i+seq_length])

    y=np_utils.to_categorical(y,num_classes=total_words)
    num_seq=len(X)

    print("number of sequences=",num_seq,"\n")
    return X,y,num_seq
# *********************************************************************************************************************

def generate_text2(seed_text,next_words,model,max_sequence_len,temp,start_story,tokenizer):
    token_list = np.array(tokenizer.texts_to_sequences([seed_text])[0])
    probs = model.predict(token_list.reshape(1,len(token_list)), verbose=0)[0]
    y_class = sample_with_temp(probs, temperature=temp)
    output_word = tokenizer.index_word[y_class] if y_class > 0 else ''
    return output_word

# *********************************************************************************************************************
def generate_text(seed_text,next_words,model,max_sequence_len,temp,start_story,tokenizer):
    output_text=seed_text
    seed_text=start_story+seed_text
    for _ in range(next_words):
        token_list=tokenizer.texts_to_sequences([seed_text])[0]
        token_list=token_list[-max_sequence_len:]
        token_list=np.reshape(token_list,(1,max_sequence_len))

        probs=model.predict(token_list,verbose=0)[0]
        y_class=sample_with_temp(probs,temperature=temp)

        output_word=tokenizer.index_word[y_class] if y_class>0 else ''

        if output_word=="|":
            break

        seed_text+=output_word+' '
        output_text+=output_word+' '
    return output_text
# *********************************************************************************************************************

# *********************************************************************************************************************
def sample_with_temp(preds,temperature=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probs=np.random.multinomial(1,preds,1)
    return np.argmax(probs)
# *********************************************************************************************************************


# *********************************************************************************************************************
def train_and_evaluate(args):
    token_list,seq_length,step,total_words,start_story,tokenizer =load_data("localdata")
    X,y,num_seq=generate_sequences(token_list,step,seq_length,total_words)
    X=np.array(X)
    y=np.array(y)

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

    epochs=500
    batch_size=32
    model.fit(X,y,epochs=epochs,batch_size=batch_size,shuffle=True)

    # create path to jobdir
    export_path = os.path.join(args.job_dir, 'rak_dnngit')

    try:
        os.mkdir(args.job_dir)
    except:
        print("job_dir "+args.job_dir+"aleady existed")

    try:
        os.mkdir(export_path)
    except:
        print ("folder "+export_path+" already existed")

    model.save(export_path+"\\allincluded.model")

def test_model(args):
    fileandpath = os.path.join(args.job_dir, 'rak_dnngit')+"\\allincluded.model"

    try:
        model=load_model(fileandpath)
        model.summary()

        token_list, seq_length, step, total_words,start_story,tokenizer = load_data("localdata")
        seedtext="till handelsman och"
        newtext=generate_text2(seedtext,1,model,10,0.01,start_story,tokenizer)

        print("output from loaded model : (",seedtext,") => ", newtext)
    except:
        print("could not find "+fileandpath)