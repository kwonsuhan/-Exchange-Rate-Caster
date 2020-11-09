#  모듈 및 데이터 로드

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

# 데이터 수집 시 사용
import requests

# 데이터 정제 시 사용
import re

# 형태소 분석(토큰화 )시 사용
from ekonlpy.tag import Mecab
from ekonlpy.sentiment import MPCK
mecab = Mecab()
mpck = MPCK()
from collections import Counter

# 벡터화할 때 사용
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec 

# Hyperparameter
max_sentences = 35
maxlen = 10


def change_date(df):
    # '2011.02.12.' -> '2011.02.12'로 변경
    df['DATE'] = df['DATE'].str.slice(0,11)
    # 날짜 형식 변환 (yyyy-mm-dd)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d', errors='raise')
    # 날짜순으로 정렬
    df = df.sort_values(by='DATE')
    df = df.reset_index(drop=True, inplace=False)
    
    return df

def duplicates(df):
    # 중복 기사 삭제
    df = df.drop_duplicates()
    df = df.reset_index(drop=True, inplace=False)
    
    return df

def load_data():
    # 데이터 불러오기 (input data)
    # 환율 데이터 프레임
    exchange_rate = pd.read_csv('../data/text_raw_data/naver_환율.csv')
    # 금리 데이터 프레임
    interest_rate = pd.read_csv('../data/text_raw_data/naver_금리.csv')
    # 외국인투자 데이터 프레임
    foreign_investment = pd.read_csv('../data/text_raw_data/naver_외국인투자.csv')
    # 미국증시 데이터 프레임
    american_stock  = pd.read_csv('../data/text_raw_data/naver_미국증시.csv')
    # 국내증시 데이터 프레임
    korea_stock = pd.read_csv('../data/text_raw_data/naver_국내증시.csv')

    # 데이터 병합하기 (위-아래로 붙이기)
    news_merge_df = pd.concat([exchange_rate, interest_rate, foreign_investment, american_stock, korea_stock])
    news_merge_df.columns = ['MEDIA','DATE','TITLE']

    # 날짜 형식 변환
    news_merge_df = change_date(news_merge_df)

    # 결측값 삭제
    news_merge_df = news_merge_df.dropna(how='any')
    news_merge_df = news_merge_df.reset_index(drop=True, inplace=False)

    # 중복 기사 삭제
    news_merge_df = duplicates(news_merge_df)

    # 날짜에 맞춰 하나의 컬럼으로 만들기
    news_merge_df = news_merge_df.groupby('DATE')['TITLE'].apply(lambda x : '\n'.join(x)).reset_index()

    # 공휴일/토요일/일요일 제거를 위한 날짜 컬럼 받아오기
    usd_df = pd.read_csv('../data/numeric_pre_data/numeric_data_for_text.csv') # 매매기준율
    usd_df = usd_df[['DATE']]

    usd_df['DATE'] = pd.to_datetime(usd_df['DATE'], format='%Y.%m.%d', errors='raise')

    # 필요한 날짜에만 맞춰 데이터를 살린다.
    input_data = pd.merge(news_merge_df[['DATE','TITLE']],usd_df, how='right', on ='DATE')
    input_data = input_data.dropna(how='any')
    input_data = input_data.reset_index(drop=True, inplace=False)
    input_data.to_csv('../data/text_pre_data/naver_news_for_att.csv', index=False)

    return input_data


# 1. 특수문자 제거 & 형태소 분석

def clean_str(string):
    # (대)괄호와 (대)괄호안의 내용 삭제
    string = re.sub('\(.+?\)','', string)
    string = re.sub('\[.+?\]','', string)
    string = re.sub('\<.+?\>','', string)

    # 특수문자 제거
    string = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!『』「」\\‘|\(\)\[\]\<\>`\'…》▶→]',' ', string)
    string = re.sub('↑','상승',string)
    string = re.sub('↓','하락',string)
    
    # 한글, 영어빼고 나머지는 다 삭제
    string = re.sub('[^a-zA-Z가-힣\s\n]','',string)
    
    return string.lstrip()

def pos_tagging(string):
    # 명사만 추출
    # tokens = mecab.nouns(string)
    tokens = mpck.tokenize(string)

    result = []
    for token in tokens:
      token = token.split('/')[0]
      # 한 글자 짜리는 제거
      if len(token) > 1:
          result.append(token)
            
    return result

def add_cleaning(sentences):
    result = []
    for sent in sentences:
        sent = re.sub('\s+', ' ', sent).strip() # 다중공백제거
        sent = pos_tagging(sent)
        if len(sent) != 0:  # 공백만 있는 string 제거
            result.append(sent)

    return result

def first_clean(df):
    temp = []
    for title in tqdm(df):

        clean_title = clean_str(title)
        split_title = clean_title.split('\n')
        each_title = add_cleaning(split_title)
        temp.append(each_title)
        
    return temp


#  2. 단어사전 구축

def flat_list(array): 
    a=[]
    for i in array:
        if type(i) == type(list()):
            a+=(flat_list(i))       
        else:
            a.append(i)
    return a

def build_dictionary(string):
    # 사전 구축
    texts = flat_list(string)   # 모든 요일의 제목을 사전을 위해 1차원으로 변경
    tokenizer = Tokenizer()        # 토크나이저 객체 로드
    tokenizer.fit_on_texts(texts)  # 빈도수를 기반으로 단어사전 구축
    # print(tokenizer.word_index)

    # tokenizer 저장하기
    with open('../data/text_pre_data/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)

    return tokenizer


#  3. word2idx 및 패딩 (입력데이터 만듦)


def build_dataset(sentences,tokenizer):

    tokenized_sentences = tokenizer.texts_to_sequences(sentences)
    tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=maxlen)
    
    pad_size = max_sentences - tokenized_sentences.shape[0]
    
    if pad_size <= 0:  # tokenized_sentences.shape[0] < max_sentences
        tokenized_sentences = tokenized_sentences[:max_sentences]
    else:
        tokenized_sentences = np.pad(
        tokenized_sentences, ((0, pad_size), (0, 0)),
        mode='constant', constant_values=0
        )
        
    return tokenized_sentences

def main_cleaning(train_x_data, tokenizer):
  #  word2idx 및 패딩
    X_data = np.zeros((len(train_x_data), max_sentences, maxlen), dtype='int32')
    for idx, sentences in tqdm(enumerate(train_x_data)):
        tokenized_sentences = build_dataset(sentences, tokenizer)
        X_data[idx] = tokenized_sentences[None, ...]
        
    return X_data

def split_train_valid_test(second_clean_data):
    text_train = second_clean_data[:1989]
    text_test = second_clean_data[1989:]

    text_valid = text_test[:249]
    text_test = text_test[249:]

    print(text_train.shape, text_valid.shape, text_test.shape)
    # 데이터 저장하기
    with open('../data/text_pre_data/text_data.pickle', 'wb') as f:
        pickle.dump(text_train, f)
        pickle.dump(text_valid, f)
        pickle.dump(text_test, f)

# 4.Word2Vec

class callback(CallbackAny2Vec): 
    """Callback to print loss after each epoch.""" 
    def __init__(self): 
        self.epoch = 0 
        self.loss_to_be_subed = 0 
        
    def on_epoch_end(self, model): 
        loss = model.get_latest_training_loss() 
        loss_now = loss - self.loss_to_be_subed 
        self.loss_to_be_subed = loss 
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now)) 
        self.epoch += 1

def word2vec_train(input_vec):
    
    print("학습 중") 
    model = Word2Vec(input_vec, size=512, window = 4, min_count=10, sg=1, compute_loss=True, iter=10, callbacks=[callback()])
    print('완료')

    # 위의 파라미터 바꾸면서 파일명 맞춰서 바꿔주기
    model.wv.save_word2vec_format('../data/text_pre_data/w2v_512features_mpck')
    print(model.wv.vectors.shape) 

# 5. word embedding

def load_word2vec():
    
    embedding_dir = '../data/text_pre_data/'

    embedding_path = os.path.join(embedding_dir, 'w2v_512features_mpck')
    embeddings_index = KeyedVectors.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
    
    return embeddings_index
    
def load_embedding(embedding_type, tokenizer, embedding_dim):
    
    if embedding_type == 'word2vec':
        embeddings_index = load_word2vec()
        
    # 사전에서 단어 수(embedding layer에서 사용)
    max_nb_words = len(tokenizer.word_index) + 1

    embedding_matrix = np.random.normal(0, 1, (max_nb_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = embeddings_index[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    print("embedding_matrix.shape: {}".format(embedding_matrix.shape))

    # embedding_matrix 저장하기
    with open('../data/text_pre_data/embedding_matrix.pickle', 'wb') as f:
        pickle.dump(embedding_matrix, f)

    return embedding_matrix


# __main__

input_data = load_data()

# 1. 특수문자 제거 & 형태소 분석(명사만 추출)
first_clean_data = first_clean(input_data['TITLE'])
input_vec = first_clean_data # w2v용

# 2. 단어사전 구축
tokenizer = build_dictionary(first_clean_data)
len(tokenizer.word_index)+1

# 3. word2idx 및 패딩 (입력데이터 만듦)
second_clean_data = main_cleaning(first_clean_data, tokenizer)

# 데이터 분할해서 저장하기
split_train_valid_test(second_clean_data)

# 4. Word2Vec
input_vec = sum(input_vec,[])
word2vec_train(input_vec)

# 5. embedding_matrix 생성
load_embedding('word2vec', tokenizer, 512)