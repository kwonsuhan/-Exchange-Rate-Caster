# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import csv


class NaverCrawlerPipeline:
    def __init__(self):
        self.csvwriter = csv.writer(open('naver_환율.csv','w', encoding='utf-8'))
        self.csvwriter.writerow(['media','time','title'])
    def process_item(self, item, spider):
        row=[]
        row.append(item['media'])
        row.append(item['time'])
        row.append(item['title'])
        self.csvwriter.writerow(row)
        return item
