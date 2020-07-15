import time
import random
import datetime
import requests
import pandas as pd
import numpy as np
from dateutil import parser
from bs4 import BeautifulSoup
from selenium.common.exceptions import *
from selenium import webdriver

class Reuters_Crawler:
    """
    Parameters:
        query: str
        
    Example:
        RC = Reuters_Crawler()
        df = RC.parse_to_dataframe()
    """
    def __init__(self, query="google"):
        self.query = query
        self.url = "https://www.reuters.com/search/news?blob={}&sortBy=date&dateRange=all".format(query)
        self.driver_path = r"./chromedriver.exe"
        self.driver = webdriver.Chrome(self.driver_path)
        self.next_button = '//*[@id="content"]/section[2]/div/div[1]/div[4]/div/div[4]/div[1]'
    
    def parse_to_dataframe(self, parse_time=10):
        """
        Parameters:
            parse_time: int (seconds)
        """
        # Open driver
        self.driver.get(self.url)
        time.sleep(2)
        # Scroll down page
        start_time = time.time()
        while (int(time.time() - start_time) < parse_time):
            if self.check_exists_by_xpath(self.next_button): 
                self.driver.find_element_by_xpath(self.next_button).click()
                time.sleep(2 + random.random())
            else: 
                break
        # Parsing
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        self.driver.quit()
        news_list = soup.find_all(name="div", attrs={"class": "search-result-content"})
        news_list_generator = self.get_news_list(news_list)
        df = pd.DataFrame(list(news_list_generator), columns=["title", "date", "query"])
        return df
        
                
    def check_exists_by_xpath(self, xpath):
        try:
            self.driver.find_element_by_xpath(xpath)
        except NoSuchElementException:
            return False
        return True
    
    def get_news_list(self, news_list):
        for i in range(len(news_list)):
            title = news_list[i].find(name="a").text
            date = news_list[i].find(name="h5", attrs={"class": "search-result-timestamp"}).text
            date = parser.parse(date, tzinfos={"EDT": "UTC-8"})
            yield [title, date, self.query]

def main():
	RC = Reuters_Crawler()
	df = RC.parse_to_dataframe()
	print(df)

if __name__ == "__main__":
	main()