import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import re
import time
from dateutil import parser
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

#https://www.reuters.com/search/news?blob=


class ArticleGetter:
    def __init__(self, base_url, query=""):
        self.base_url = base_url
        self.search_url = base_url + query + '&sortBy=date&dateRange=all'
    def get_daily_news(self, query):
        # get our search webpage
        search_url = self.base_url + query + '&sortBy=date&dateRange=all'
        # script = 'which Chrome'
        # a = os.system(script)
        driver = webdriver.Chrome('C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')
        reuters_url = 'https://www.reuters.com'
        driver.get(search_url)

        origin_soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 5 means the click time
        for i in range(1, 5):
            button = driver.find_element_by_class_name('search-result-more-txt')
            button.click()

            time.sleep(1)

        # get current time
        previous_date = str(datetime.now().date() - timedelta(days=1))

        article_title = []
        article_time = []
        article_query = []
        article_url = []

        expand_soup = BeautifulSoup(driver.page_source, 'html.parser')
        divs = expand_soup.find_all('div', class_='search-result-indiv')

        for div in tqdm(divs):
            try:
                title = div.find('h3', class_='search-result-title')
                time_ = div.find('h5', class_='search-result-timestamp').text
                news_utc8_datetime = parser.parse(time_, tzinfos={"EDT": "UTC-8"})
                news_utc8_date = str(news_utc8_datetime).split()[0]


                if news_utc8_date == previous_date:
                    article_title.append(title.text)
                    article_time.append(news_utc8_datetime)
                    article_query.append(query)
                    article_url.append(reuters_url + title.find('a')['href'])

            except:
                pass

        article_df = pd.DataFrame({
            'title': article_title,
            'time': article_time,
            'query': article_query,
            'url': article_url
        })

        return article_df


if __name__ == "__main__":
    base_url = 'https://www.reuters.com/search/news?blob='
    query = 'Google'

    article_getter = ArticleGetter(base_url)
    article_getter.get_daily_news('Google')