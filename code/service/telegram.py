
# 설치필요
# get_ipython().system('pip install python-telegram-bot')


# 처음 챗봇 만들 때 필요한 코드
# 우리는 이미 채널을 개설해서 추가 실행 필요없음

# import telegram

# telgm_token = '1314835737:AAFgivmVXKs9SDMvd5gv-ji7MGgy9EVk7WU'

# bot = telegram.Bot(token = telgm_token)

# updates = bot.getUpdates()

# print(updates)

# for i in updates:
#     print(i)

# print('start telegram chat bot')
# telegram chatbot ID : '676149244'


#  챗봇 메시지 작성 및 전송

import telegram
import pandas as pd
import numpy as np

crawl_data = pd.read_csv('../data/han_result/top_news_for_telegram.csv')
dallar_predict = pd.read_csv('../data/darnn_result/dallar_predict.csv')

pred_date = '2020-10-05'
text_date = '2020-09-29'
top_news = crawl_data.loc[crawl_data['date']==text_date]
top_news = top_news.reset_index()

sentence = "< "+pred_date+" 달러 예측 환율>\n\n"+str(np.round(dallar_predict['dallar'][0],2))+"원\n\n<TOP "+str(len(top_news))+" News >\n"
for i in range(len(top_news)):
    sentence = sentence + "\n <TITLE>\n"+top_news['title'][i]+"\n<DESCRIPTION>\n"+top_news['description'][i]+"\n<LINK>\n"+top_news['link'][i]+'\n'
# image_path = './plots/prediction_up/down.png'


def telegram_main():
    telgm_token = '1314835737:AAFgivmVXKs9SDMvd5gv-ji7MGgy9EVk7WU'
    bot = telegram.Bot(token = telgm_token)
    bot.sendMessage(chat_id = '@multi_campus_exchange_rate', text=sentence)
#     bot.send_photo(chat_id = '@multi_campus_exchange_rate', photo=open(image_path, 'rb'))
    print('성공')

telegram_main()

