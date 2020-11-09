# 코드 참고
# https://github.com/hist0613/keras-implementations/blob/main/IMDB-HieAtt.ipynb

import pickle
import pandas as pd
import numpy as np
import os
import re
import sys
import urllib.request
import json

import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sn

from keras.layers import Input, Embedding, Dense
from keras.layers import Lambda, Permute, RepeatVector, Multiply
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import GRU #CuDNNGRU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop

# 단어사전 & embedding matrix 불러오기
with open('../data/text_pre_data/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('../data/text_pre_data/embedding_matrix.pickle', 'rb') as f:
    embedding_matrix = pickle.load(f)

# Hyperparameter
max_sentences = 35
max_sentence_length = 10  # maxlen이랑 같음
# 사전에서 단어 수(embedding layer에서 사용)
max_nb_words = len(tokenizer.word_index) + 1

# Text 데이터 불러오기
def load_text():
    with open('../data/text_pre_data/text_data.pickle', 'rb') as f: 
        text_train = pickle.load(f)
        text_valid = pickle.load(f)
        text_test = pickle.load(f)

    # 입력데이터 train & test
    print(text_train.shape, text_valid.shape, text_test.shape)

    return text_train, text_valid, text_test


# y값 데이터(to_categorical인 경우)
def load_target():

    # 전일대비 수치데이터 로드
    c = pd.read_csv('../data/numeric_pre_data/numeric_data_for_text.csv') # 매매기준율

    # 타겟값만 추출
    target = c['USD_KRW'].to_frame()
    
    # <1> 값의 범위로 category 지정

    target.loc[(target['USD_KRW']> -40) & (target['USD_KRW']<= -30), 'label'] = 0       # -40 < x <= -31 : 0
    target.loc[(target['USD_KRW']> -30) & (target['USD_KRW']<= -20), 'label'] = 1       # -30 < x <= -21 : 1
    target.loc[(target['USD_KRW']> -20) & (target['USD_KRW']<= -10), 'label'] = 2       # -20 < x <= -11 : 2
    target.loc[(target['USD_KRW']> -10) & (target['USD_KRW']<= 0), 'label'] = 3         # -10 < x <=   0 : 3
    target.loc[(target['USD_KRW']> 0) & (target['USD_KRW']<= 10), 'label'] = 4          #   0 < x <=  10 : 4
    target.loc[(target['USD_KRW']> 10) & (target['USD_KRW']<= 20), 'label'] = 5         #  10 < x <=  20 : 5
    target.loc[(target['USD_KRW']> 20) & (target['USD_KRW']<= 30), 'label'] = 6         #  20 < x <=  30 : 6
    target.loc[(target['USD_KRW']> 30) & (target['USD_KRW']<= 40), 'label'] = 7         #  30 < x <=  40 : 7
    target.loc[target['USD_KRW']> 40, 'label'] = 8

    # 그림그릴때 사용
    target_test_using_predtest = target[-249:]

    # 카테고리 나누기

    y_data = target['label']
    nb_classes = len(set(y_data))
    target_Y = to_categorical(y_data, nb_classes)

    target_train = target_Y[:1989]
    target_test = target_Y[1989:]

    target_valid = target_test[:249]
    target_test = target_test[249:]

    # 데이터 저장하기
    with open('../data/text_pre_data/target_category_data.pickle', 'wb') as f:
        pickle.dump(target_train, f)
        pickle.dump(target_valid, f)
        pickle.dump(target_test, f)

    print(target_train.shape, target_valid.shape, target_test.shape)

    return target_train, target_valid, target_test, target_test_using_predtest


# def my_summary(x):
#     result = {
#         'sum': x.sum(), 
#         'count': x.count(), 
#         'mean': x.mean(), 
#         'variance': x.var()
#     }
#     return result
# target.groupby('label')['USD_KRW'].apply(my_summary).unstack() # grp_digitize / label


# HAN Model

def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    
    # TP = y_target_yn * y_pred_yn
    # FN = y_target_yn - (y_target_yn * y_pred_yn) 
    
    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # FP = y_pred_yn - (y_target_yn * y_pred_yn) 
    
    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score


class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], self.attention_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(self.attention_dim, ),
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(self.attention_dim, 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        u_it = K.tanh(K.dot(x, self.W) + self.b)
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)
        
        return a_it
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
    

def WeightedSum(attentions, representations):
    # from Shape(batch_size, len_units) to Shape(batch_size, rnn_dim * 2, len_units)
    repeated_attentions = RepeatVector(K.int_shape(representations)[-1])(attentions)
    # from Shape(batch_size, rnn_dim * 2, len_units) to Shape(batch_size, len_units, lstm_dim * 2)
    repeated_attentions = Permute([2, 1])(repeated_attentions)

    # compute representation as the weighted sum of representations
    aggregated_representation = Multiply()([representations, repeated_attentions])
    aggregated_representation = Lambda(lambda x: K.sum(x, axis=1))(aggregated_representation)

    return aggregated_representation
    
    
def HieAtt(embedding_matrix,
           max_sentences,
           max_sentence_length,
           nb_classes,
           embedding_dim=512,
           attention_dim=100,
           rnn_dim=150,
           include_dense_batch_normalization=False,
           include_dense_dropout=True,
           nb_dense=1,
           dense_dim=300,
           dense_dropout=0.2,
           optimizer = keras.optimizers.Adam(lr=0.001)):

    # embedding_matrix = (max_nb_words + 1, embedding_dim)
    max_nb_words = embedding_matrix.shape[0] - 1
    embedding_layer = Embedding(max_nb_words + 1, 
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sentence_length,
                                trainable=False)

    # first, build a sentence encoder
    sentence_input = Input(shape=(max_sentence_length, ), dtype='int32')
    embedded_sentence = embedding_layer(sentence_input)
    embedded_sentence = Dropout(dense_dropout)(embedded_sentence)
    contextualized_sentence = Bidirectional(GRU(rnn_dim, return_sequences=True))(embedded_sentence)
    
    # word attention computation
    word_attention = AttentionLayer(attention_dim)(contextualized_sentence)
    sentence_representation = WeightedSum(word_attention, contextualized_sentence)
    
    sentence_encoder = Model(inputs=[sentence_input], 
                             outputs=[sentence_representation])

    # then, build a document encoder
    document_input = Input(shape=(max_sentences, max_sentence_length), dtype='int32')
    embedded_document = TimeDistributed(sentence_encoder)(document_input)
    contextualized_document = Bidirectional(GRU(rnn_dim, return_sequences=True))(embedded_document)
    
    # sentence attention computation
    sentence_attention = AttentionLayer(attention_dim)(contextualized_document)
    document_representation = WeightedSum(sentence_attention, contextualized_document)
    
    # finally, add fc layers for classification
    fc_layers = Sequential()
    for _ in range(nb_dense):
        if include_dense_batch_normalization == True:
            fc_layers.add(BatchNormalization())
        fc_layers.add(Dense(dense_dim, activation='relu'))
        if include_dense_dropout == True:
            fc_layers.add(Dropout(dense_dropout))
    fc_layers.add(Dense(nb_classes, activation='softmax')) # sigmoid / softmax
    
    pred_sentiment = fc_layers(document_representation)

    model = Model(inputs=[document_input],
                  outputs=[pred_sentiment])
    
    ############### build word attention extractor ###############
    word_attention_extractor = Model(inputs=[sentence_input],
                                     outputs=[word_attention])
    word_attentions = TimeDistributed(word_attention_extractor)(document_input)
    attention_extractor = Model(inputs=[document_input],
                                     outputs=[word_attentions, sentence_attention])
    
    model.compile(loss=['categorical_crossentropy'], #  binary_crossentropy / categorical_crossentropy
              optimizer=optimizer,
              metrics=['accuracy', precision, recall, f1score])

    return model, attention_extractor


# 결과 예측 & 시각화

def model_accuracy(model, text_test,target_test_using_predtest):
    predictions = model.predict(text_test)
    pred_Y = [np.argmax(predictions[idx]) for idx in range(0,249)]
    cnt = 0
    for pre, real in zip(pred_Y, target_test_using_predtest['label'].tolist()):
      if pre == real:
        cnt += 1 
    print((cnt/249)*100)

    return str(round((cnt/249)*100, 1)).replace('.','_')

# 예측값 저장 및 시각화
def eval_pred_save(model, text_test, target_test_using_predtest, model_acc):

    #### categorical인 경우만 해당
    predictions = model.predict(text_test)

    plt.figure(figsize=(30, 10))
    xlabel = range(0,text_test.shape[0])
    plt.plot(xlabel, target_test_using_predtest['label'],'-', label='actual')
    plt.plot(xlabel, [np.argmax(predictions[idx]) for idx in range(0,text_test.shape[0])],'-' ,label='prediction')
    plt.legend()
    plt.grid()
    plt.savefig('../data/han_result/pred_graph_{}.png'.format(model_acc))

    # 예측한 결과 저장
    a = pd.DataFrame({'categorical_pred' : [np.argmax(predictions[idx]) for idx in range(0,text_test.shape[0])]})
    a.to_csv('../data/han_result/categorical_pred.csv', index=False)

# 모델 그림 및 모델 저장
def model_save(model, model_acc):

    # 모델 시각화
    plot_model(model, to_file='../data/han_result/model_img_{}.png'.format(model_acc), show_shapes=True)

    # 모델 저장
    model.save_weights("../data/han_result/hieatt_model_{}.h5".format(model_acc))
    print("Saved hieatt_model to disk")

    # 모델 저장
    attention_extractor.save_weights("../data/han_result/att_model_{}.h5".format(model_acc))
    print("Saved att_model to disk")

# loss & acc 시각화
def visual_loss_acc(history, model_acc):
    # Visualization loss & acc

    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('../data/han_result/model_loss_{}.png'.format(model_acc))


    plt.clf()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../data/han_result/model_acc_{}.png'.format(model_acc))

# 6. Word Attention Visualization
def word_att_data_load():
    news_merge_df = pd.read_csv('../data/text_pre_data/naver_news_for_att.csv')

    # TOPnews attention에 대한 실제 기사 제목을 뽑기 위함(나중에 깔끔하게 할 것!)---[1]
    test_raw = news_merge_df[-249:]
    test_raw = test_raw.reset_index(drop=True, inplace=False)
    test_title = test_raw.TITLE.tolist()
    test_date = test_raw.DATE.tolist()

    # TOPnews attention에 대한 실제 기사 제목을 뽑기 위함(나중에 깔끔하게 할 것!)---[2]
    testset = []
    for i, test in enumerate(test_title):
        test_sent = test.split('\n')
        testset.append(test_sent)

    return testset, test_date

def wor2idx(tokenizer):    
    word_rev_index = {}
    for word, i in tokenizer.word_index.items():
        word_rev_index[i] = word
    return word_rev_index

def sentiment_analysis(tokenized_sentences): 
    sum_att_dict = {}

    # word attention 가져오기
    pred_attention = attention_extractor.predict(np.asarray([tokenized_sentences]))[0][0]
    # print(pred_attention)
    for sent_idx, sentence in enumerate(tokenized_sentences):
        # print('sent_idx : {}, sentence : {}'.format(sent_idx, sentence))
        if sentence[-1] == 0:
            continue
        
        maxlen = 10 # 문장 내 포함 단어 개수
        for word_idx in range(maxlen):
            if sentence[word_idx] != 0:
                words = [word_rev_index[word_id] for word_id in sentence[word_idx:]]
                # print('words : {}'.format(words))
                pred_att = pred_attention[sent_idx][-len(words):]
                # print('pred_att : {}'.format(pred_att))
                # 하나의 헤드라인에 대한 Attention 값의 합
                sum_att_dict[sent_idx] = sum(pred_att)
                # print('sum_att : {}'.format(sum(pred_att)))
                pred_att = np.expand_dims(pred_att, axis=0)
                break

    # print('sum_att_dict : {}'.format(sum_att_dict))
    return sum_att_dict
        # fig, ax = plt.subplots(figsize=(len(words), 1))
        # plt.rc('xtick', labelsize=16)
        # midpoint = (max(pred_att[:, 0]) - min(pred_att[:, 0])) / 2
        # heatmap = sn.heatmap(pred_att, xticklabels=words, yticklabels=False, square=True, linewidths=0.1, cmap='coolwarm', center=midpoint, vmin=0, vmax=1)
        # plt.xticks(rotation=45)
        # plt.show()

def extract_top_news(text_test, testset, attention_extractor, word_rev_index):
    top_news = []
    for idx, test in enumerate(text_test):
        ranking_att = sentiment_analysis(test)
        # value값을 기준으로 정렬 후 TOP3 추출
        sorted_att = sorted(ranking_att.items(),reverse=True, key=lambda x : x[1])[:3]
        print('sorted_att : {}'.format(sorted_att))
        title = []
        for k in sorted_att:
          # TOP3의 실제 제목 추출
            title.append(testset[idx][k[0]])
        top_news.append(title)
    return top_news

# 7. 네이버 검색 API


def build_news_api(search_title, start, sort, disp):
  encText = urllib.parse.quote(search_title)
  param_start = "&start=" + str(start)
  param_sort = "&sort=" + sort
  param_disp = "&display=" + str(disp)
  url = "https://openapi.naver.com/v1/search/news.json?query=" + encText + param_disp +param_start + param_sort  # json 결과
  
  return url

def resquest_api(url, client_id, client_secret):
  request = urllib.request.Request(url)
  request.add_header("X-Naver-Client-Id", client_id)
  request.add_header("X-Naver-Client-Secret", client_secret)
  
  response = urllib.request.urlopen(request)
  if response.getcode() == 200:
      return response.read().decode('utf-8')
  else:
      return "Error Code:" + rescode

def remove_tag(content):
   cleantext = re.sub('<.*?>', '', content)
   cleantext = re.sub('&quot;', '', cleantext)
   return cleantext


# __main__

# 텍스트 데이터 로드하기
text_train, text_valid, text_test = load_text()
# 타겟 데이터 로드하기 
target_train, target_valid, target_test, target_test_using_predtest = load_target()

model_name = "HieAtt"
model_path = './gdrive/My Drive/{}.h5'.format(model_name)
checkpointer = ModelCheckpoint(filepath=model_path,
                               monitor='val_acc',
                               verbose=True,
                               save_best_only=True,
                               mode='max')

model, attention_extractor = HieAtt(embedding_matrix=embedding_matrix,
                                    max_sentences=max_sentences,
                                    max_sentence_length=max_sentence_length,
                                    nb_classes = 9, 
                                    embedding_dim=512,
                                    attention_dim=100,
                                    rnn_dim=150,
                                    include_dense_batch_normalization=False,
                                    include_dense_dropout=True,
                                    nb_dense=1,
                                    dense_dim=300,
                                    dense_dropout=0.2,
                                    optimizer = keras.optimizers.Adam(lr=0.00001))


model.summary()

# model fit
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(text_train,
                    target_train,
                    batch_size=10,
                    epochs=1,
                    verbose=True,
                    validation_data=(text_valid, target_valid),
                    callbacks=[early_stop])

# 평가
_loss, _acc, _precision, _recall, _f1score  = model.evaluate(text_test, target_test)
print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc, _precision, _recall, _f1score))

# 정확성 & 결과 시각화
model_acc = model_accuracy(model, text_test,target_test_using_predtest)
eval_pred_save(model, text_test, target_test_using_predtest, model_acc)
model_save(model, model_acc)
visual_loss_acc(history, model_acc)

# 주요 뉴스 추출하기
# 워드 어텐션에 사용될 데이터셋 로드
testset, test_date = word_att_data_load()
word_rev_index = wor2idx(tokenizer)
# attention을 통해 TOP 3의 중요 뉴스 추출
top_news = extract_top_news(text_test, testset, attention_extractor, word_rev_index)

# with open('../data/han_result/topnews_for_naver_api.pickle', 'wb') as f:
    # pickle.dump(top_news, f)

# 네이버 API로 검색하기
date, title, link, description = [], [], [], []
client_id = "1jjTllhWavLoG6b0L8Y2"
client_secret = "M2hkKBMSHt"

for d, top in zip(test_date, top_news):
  for news in top:
    print(news.replace('"',''))
    news = news.replace('"','')
    url = build_news_api(news, 1, 'sim', 1)
    res_api = resquest_api(url, client_id, client_secret)
    json_data = json.loads(res_api)
    print(json_data)
    try:
      date.append((str(d).split(' ')[0]))
      title.append(remove_tag(json_data['items'][0]['title']))
      link.append(json_data['items'][0]['link'])
      description.append(remove_tag(json_data['items'][0]['description']))
    except:
      date.append((str(d).split(' ')[0]))
      title.append(remove_tag(json_data['items'][0]['title']))
      link.append(json_data['items'][0]['link'])
      description.append(remove_tag(json_data['items'][0]['description']))

# 수집한 기사의 날짜 제목,설명,링크를 dataframe로 저장
crawl_data = pd.DataFrame(columns={'date','title','link','description'})
crawl_data['date'] = date
crawl_data['title'] = title
crawl_data['description'] = description
crawl_data['link'] = link
# 저장
crawl_data.to_csv('../data/han_result/top_news_for_telegram.csv', index=False)

