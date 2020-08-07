# seq matcher
# spacy matcher
import sys
import ssl
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
import requests
from bs4 import BeautifulSoup
from database.database import db
import pandas as pd
from absl import logging

import numpy as np
import os
import pandas as pd
import re

import pysentiment as ps
from tqdm import tqdm
from database.tables.crawling_data import CrawlingData
from database.tables.output_news import OutputNews
import json
from pprint import pprint
from sqlalchemy import Column, Integer, String, DateTime, Date, and_
from datetime import datetime,timedelta
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


class KeysentGetter():
	def __init__(self):
		self.title = [] #all the titles
		self.doc = [] # all the documents wrt the title
		self.keysent_idx = [] # all the keysents
		self.url = [] #all url
		self.important_title = []
		self.polarity = []
		self.important_news = []
		self.title_polarity = []
		self.companys = []
		self.dates = []
		self.q_data = self._get_all_url()

	def _get_all_url(self):
		result = CrawlingData.query.filter(CrawlingData.date > datetime(2020,6,30))
		# result = CrawlingData.query.filter_by( date =  (datetime.now().date() - timedelta(days=1)))
		self.q_data = result
		for r in result:
			self.url.append(r.url)
			self.dates.append(r.date)
			self.companys.append(r.company)
			self.dates.append(r.date)

	def url2news(self):
		hiv4 = ps.hiv4.HIV4()
		_lst = []
		for url_idx,url in tqdm(enumerate(self.url),total = len(self.url), desc = 'get urls'):
			check_repeated=0
			resp = requests.get(url)
			soup = BeautifulSoup(resp.text, 'html.parser')
			paragraph = soup.find_all('p')
			paragraph = [p.text for p in paragraph]
			paragraph = paragraph[1:-1]
			title = soup.find('title').text
			title = title.lstrip()
			title_score = hiv4.get_score(hiv4.tokenize(  title ))
			title_score = title_score['Polarity']
			# print(title_score)
			if(title_score < 0.5 and title_score >= -0.5):
				continue
			po = []
			# print(title)
			paragraph_sents = ""
			for p in paragraph:
				paragraph_sents += p + "%%"
				tokens = hiv4.tokenize(p)
				s = hiv4.get_score(tokens)
				po.append(s['Polarity'])
			paragraph_sents = paragraph_sents[:-2]
			key_idx = ""
			for po_idx,polarity in enumerate(po):
				if(polarity >= 0.75 or polarity <= -0.75):
					key_idx+=str(po_idx) + "%%"
			key_idx = key_idx[:-2]
			# print("paragraph length" ,len(paragraph))
			# print("keyidx", key_idx)
			_lst.append(  OutputNews( title, date = self.dates[url_idx], company = self.companys[url_idx], paragraph = paragraph_sents, keysent = key_idx ))

		db.session.add_all(_lst)
		db.session.commit()



def test_url():
	result = CrawlingData.query.filter_by(date=(datetime.now().date() - timedelta(days=1)))
	return result
