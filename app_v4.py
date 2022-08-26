#%%
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd


# %%
import matplotlib.pyplot as plt
import pandas as pd

import unicodedata

# %%
import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl


import pandas as pd


# %%
#分類スコアを求めるために変換する関数
def encoding_plus_for_logits(dataset_List, batch, model):#dataset_List=[text, label], batch=入れすぎるとメモリ不足
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    
    dataset_encoding_list = []
    for text in dataset_List:
        encoding = tokenizer.encode_plus(
            text,
            max_length = 128,
            padding='max_length',
            truncation=True,
        )
        #encoding['labels'] = label # ラベルを追加
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_encoding_list.append(encoding)
    #このままだとoutput = bert_sc(**dataset_encoding)に入らないので，
    #以下で整形する
    #dictから抜き出す
    labels_predicted = []
    
    for k in tqdm(range(len(dataset_List)//batch + 1)):
        encoding_input_ids = []
        encoding_token_type = []
        encoding_attention_mask = []
        # print(k,len(dataset_List)//batch)
        if k < len(dataset_List)//batch:
            for i in range(batch):
                encoding_input_ids.append(dataset_encoding_list[k*batch + i]['input_ids'])
                encoding_token_type.append((dataset_encoding_list[k*batch + i]['token_type_ids']))
                encoding_attention_mask.append((dataset_encoding_list[k*batch + i]['attention_mask']))
        else:
            # print('amari')
            for i in range(len(dataset_List)%batch):
                print(k*batch + i)
                encoding_input_ids.append(dataset_encoding_list[k*batch + i]['input_ids'])
                encoding_token_type.append((dataset_encoding_list[k*batch + i]['token_type_ids']))
                encoding_attention_mask.append((dataset_encoding_list[k*batch + i]['attention_mask']))

        #tensorをまとめる
        try:
            dataset_encoding = {}
            dataset_encoding = {'input_ids': torch.stack(encoding_input_ids).cpu(),
                            'token_type_ids':torch.stack(encoding_token_type).cpu(),
                            'attention_mask':torch.stack(encoding_attention_mask).cpu(),
                            }
        except:
            print('failed')
        
        #分類ラベルを得る
        try:
            with torch.no_grad():
                output = model(**dataset_encoding)
                scores = output.logits
            labels_predicted.append(scores.argmax(-1).tolist())
        except:
            print('failed')
            
    
    return [l for m in labels_predicted for l in m]


def predict(
            dir_model = 'Tomohiro/MediA_C',
            dir_test_file = 'lasix_processed_v3.csv'):

    # %%
    
    bert_sc = BertForSequenceClassification.from_pretrained(
        dir_model
    ).cpu()
    print('test')
    df = pd.read_table(dir_test_file, sep=',', index_col='Unnamed: 0')
    print('test2')
    df = df.dropna(subset=['text'])
    print('test3')
    

    # %%
    d_test_text = df['text'].to_list()
    print('test4')
    labels = encoding_plus_for_logits(d_test_text, 16, bert_sc)
    print('test5')

    # %%
    df['predicted']=labels
    return df
    


#%%
# サイドバー
# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv') 
med_name = st.sidebar.text_input('薬物名')

#%%


# メイン画面
st.header('読み込みデータ表示')
if uploaded_file is not None:
    # tweet = pd.read_csv(uploaded_file, sep=',', index_col='Unnamed: 0')
    # tweet
    # predict(dir_model='Tomohiro/MediA_C', dir_test_file=uploaded_file)
    # exec(open('BERT_for_streamlit.py').read())
    print('test')
    #%%
    df = predict(dir_model='Tomohiro/MediA_C', dir_test_file=uploaded_file)

    #%%
    df
    # %%

    df = df[~df['text'].duplicated()]
    df['created_at'] = pd.to_datetime(df['created_at'])
    tweet_df = df.groupby(pd.Grouper(key='created_at', freq='W', convention='start')).size()


    #%%
    labels = ["NC", "Sales NC", "Use", "Others"]
    #%%
    df_list = []

    for i, label in enumerate(labels):
        print(i, label)
        df['created_at'] = pd.to_datetime(df['created_at'])
        tweet_df = df[df['predicted']==i].groupby(pd.Grouper(key='created_at', freq='W', convention='start')).size()
        tweet_df = pd.DataFrame(tweet_df, columns=[label])
        df_list.append(tweet_df)

    #%%
    num_df = pd.concat(df_list, axis=1)


    st.title(med_name)\
        
    st.line_chart(num_df)

