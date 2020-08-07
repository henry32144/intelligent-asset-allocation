import joblib
import nltk
import time
import random
import datetime
import requests
import grequests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from dateutil import parser
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
# nltk.download('stopwords')


class ReutersCrawlerV3:
    def __init__(self):
        self.sp_500 = open("../data/ticker_name.txt", 'r').read().split('\n')
        self.driver_path = r"./chromedriver.exe"
        self.driver = webdriver.Chrome(self.driver_path)
        self.next_button = '//*[@id="content"]/section[2]/div/div[1]/div[4]/div/div[4]/div[1]'
    
    def parse_to_dataframe(self, query):
        """
        Parameters:
            query: str
        """
        # Open driver
        self.query = query
        self.url = "https://www.reuters.com/search/news?blob={}&dateRange=all".format(query)
        self.driver.get(self.url)
        time.sleep(2)
        # Scroll down page
        self.scroll_to_bottom()
        # Parsing
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        self.driver.quit()
        news_list = soup.find_all(name="div", attrs={"class": "search-result-content"})
        news_list_generator = self.get_news_list(news_list)
        df = pd.DataFrame(list(news_list_generator), columns=["title", "date", "query", "url"])
        df = df.drop_duplicates(subset="title")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["date"] = df["date"].apply(lambda x: x.date())
        # Add ticker column
        sp500_dict = self.get_sp500_dict()
        df["ticker"] = df["query"].apply(lambda key: sp500_dict.get(key, np.nan))
        return df
    
    def parse_all_to_dataframe(self):
        query_list = [element.split("\t")[0] for element in self.sp_500]
        data = pd.DataFrame(columns=["title", "date", "query", "url", "ticker"])
        for query in progressBar(query_list):
            try:
                df = self.parse_to_dataframe(query)
                data = pd.concat([data, df], axis=0)
            except: 
                pass
        return data

    def scroll_to_bottom(self):
        while True:
            # self.driver.find_element_by_css_selector('.search-result-more-txt').click()
            element = WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'search-result-more-txt')))
            if element.text.lower() == 'no more results':
                break
            else:
                element.click()
    
    def get_news_list(self, news_list):
        for i in range(len(news_list)):
            title = news_list[i].find(name="a").text
            date = news_list[i].find(name="h5", attrs={"class": "search-result-timestamp"}).text
            date = parser.parse(date, tzinfos={"EDT": "UTC-8", "EST": "UTC-8"})
            url = news_list[i].find(name="a").get("href")
            url = "https://www.reuters.com" + url
            yield [title, date, self.query, url]
            
    def get_sp500_dict(self):
        sp_500_dict = dict()
        for element in self.sp_500:
            sp_500_dict[element.split("\t")[0]] = element.split("\t")[1]
        return sp_500_dict


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


class AsynchronousCrawler:
    def __init__(self, lst):
        self.urls = lst

    def exception(self, request, exception):
        print("Problem: {}: {}".format(request.url, exception))

    def asynchronous(self):
        return grequests.map((grequests.get(u) for u in self.urls), exception_handler=self.exception, size=5)

    def collate_responses(self, results):
        return [self.parse(x.text)for x in results if x is not None]

    def parse(self, response_text):
        soup = BeautifulSoup(response_text, "html.parser")
        paragraph = []
        for element in soup.find_all("p"):
            paragraph.append("".join(element.find_all(text=True)))
        return "".join(paragraph[1:-2])


def main():
	# Get data from Reuters
	crawler = ReutersCrawlerV3()
	data = crawler.parse_to_dataframe(query="Google")
	# data = crawler.parse_all_to_dataframe()
	data = data.drop_duplicates(subset="title")
	data = data.drop_duplicates(subset="url")

	# Get content from url
	crawler = AsynchronousCrawler(data.url.tolist())
	results = crawler.asynchronous()
	contents = crawler.collate_responses(results)

	content_list = []
	for content, url in zip(contents, data.url.tolist()):
		content_list.append([content, url])
	final_df = pd.merge(data, pd.DataFrame(content_list, columns=["content", "url"]), on="url")
	final_df = final_df.drop_duplicates(subset="content")
	joblib.dump(final_df, "../data/sp500_top100_content_v5.bin", compress=5)
	