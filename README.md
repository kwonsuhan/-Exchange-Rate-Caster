## Exchange Rate Catser

<< 환율 뉴스 리포트 >>

* 프로젝트 주제 :  환율 관련 수치 데이터와 텍스트 데이터를 함께 수집하여 미래 달러 ① **환율을 예측**하고 ② **관련 주요 뉴스를 제공**

* 프로젝트 기간 : 2020/09/11 ~ 2020/11/11 (2개월)
* 프로젝트 팀원 : 장연미, 이정민, 임예빈, 한권수

<img src="img/model.PNG" alt="model" style="zoom: 80%;" />

```bash
# 코드 구성 
code/
	darnn/             # Dual stage Attention
		constants.py
		custom_types.py
		main.py
		modules.py
		utils.py
		
	data_pre/          # Data Preprocessing 
		numeric_preprocessing.py
		text_preprocessing.py

	han/               # Hierarchical Attention Network
		HAN_model.py

	naver_crawler/     # Naver News Crawler using Scrapy

	service/           # Send message using Telegram
		telegram.py
		
	main.py            # Main code
```

