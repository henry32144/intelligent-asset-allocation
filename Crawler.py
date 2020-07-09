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
        df = RC.parse_to_dataframe(query="Google")
    """
    def __init__(self):
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
        return df
                
    def check_exists_by_xpath(self, xpath):
        try:
            self.driver.find_element_by_xpath(xpath)
        except NoSuchElementException:
            return False
        return True

    def scroll_to_bottom(self):

        old_position = 0
        new_position = None

        while new_position != old_position:
            # Get old scroll position
            old_position = self.driver.execute_script(
                    ("return (window.pageYOffset !== undefined) ?"
                     " window.pageYOffset : (document.documentElement ||"
                     " document.body.parentNode || document.body);"))
            # Sleep and Scroll
            time.sleep(1)
            self.driver.execute_script((
                    "var scrollingElement = (document.scrollingElement ||"
                    " document.body);scrollingElement.scrollTop ="
                    " scrollingElement.scrollHeight;"))
            # Get new position
            new_position = self.driver.execute_script(
                    ("return (window.pageYOffset !== undefined) ?"
                     " window.pageYOffset : (document.documentElement ||"
                     " document.body.parentNode || document.body);"))
            self.driver.find_element_by_xpath(self.next_button).click()
            time.sleep(2 + random.random())
    
    def get_news_list(self, news_list):
        for i in range(len(news_list)):
            title = news_list[i].find(name="a").text
            date = news_list[i].find(name="h5", attrs={"class": "search-result-timestamp"}).text
            date = parser.parse(date, tzinfos={"EDT": "UTC-8"})
            url = news_list[i].find(name="a").get("href")
            url = "https://www.reuters.com" + url
            yield [title, date, self.query, url]

def main():
	RC = Reuters_Crawler()
	df = RC.parse_to_dataframe()
	print(df)

if __name__ == "__main__":
	main()