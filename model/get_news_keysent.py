# seq matcher
# spacy matcher
import sys
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
from sqlalchemy import Column, Integer, String, DateTime, Date
from datetime import datetime,timedelta

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
		self.q_data = self._get_all_url()


	def _get_all_url(self):
		result = CrawlingData.query.filter_by( date =  (datetime.now().date() - timedelta(days=1)))
		self.q_data = result
		for r in result:
			self.url.append(r.url)
			self.companys.append(r.company)
		print(result)
		return result

	def url2news(self):
		company_idx = 0
		for url in tqdm(self.url):
		    resp = requests.get(url)
		    soup = BeautifulSoup(resp.text, 'html.parser')
		    paragraph = soup.find_all('p')
		    paragraph = [p.text for p in paragraph]
		    # while(paragraph[0][-8:] == "min read"):
		    # 	del paragraph[0]

		    title = soup.find('title').text
		    title = title.lstrip()
		    if (len(self.title) != 0):
		    	if( self.get_jaccard_sim(title, self.title[-1]) >0.8 ):
		    		del self.companys[company_idx+1]
		    		continue
		    # print(self.title)
		    company_idx+=1
		    self.title.append(title)
		    hiv4 = ps.hiv4.HIV4()
		    self.doc.append(paragraph)
		    self.title_polarity.append( hiv4.get_score(hiv4.tokenize(title))['Polarity']  )
		    
		    po = []
		    for p in paragraph[:-1]:
		    	tokens = hiv4.tokenize(p)
		    	s = hiv4.get_score(tokens)
		    	# print(tokens)
		    	po.append(s['Polarity'])
		    	# print(s)
		    self.polarity.append(po)
		# print(self.doc[0])
		# print(self.title_polarity)

	def get_jaccard_sim(self, str1, str2): 
	    a = set(str1.split()) 
	    b = set(str2.split())
	    c = a.intersection(b)
	    return float(len(c)) / (len(a) + len(b) - len(c))
				
	def get_news(self):
		result = []
		_dict = {
		"title": 0,
		"pargraph":0,
		"keysent":0
		}
		for i, po in tqdm(enumerate(self.polarity), total = len(self.polarity)):
			if (self.title_polarity[i] >= 0.85 or self.title_polarity[i] <= -0.85):
				key_idx = []
				for j, p in enumerate(po):
					if (p >= 0.85):
						key_idx.append(j)
					elif (p <= -0.85):
						key_idx.append(-1*j)
				if(len(key_idx) != 0):
					self.important_news.append(self.doc[i])
					self.keysent_idx.append(key_idx)
					self.important_title.append(self.title[i])
		print(len(self.doc))
		print(len(self.important_news))
				# temp = _dict
				# temp['title'] = self.title[i]
				# temp['paragraph'] = self.doc[i]
				# temp['keysent'] = key_idx
				# result.append(temp)

		# for i0, n in enumerate(self.important_news):
		# 	for k in self.keysent_idx[i0][1:]:
		# 		self.important_news[i0][k+1] = "**" + self.important_news[i0][k+1] + "**"
		# 	if(self.keysent_idx[i0][0] > 0):
		# 		print("------------NEWS : Positive------------")
		# 	else:
		# 		print("------------NEWS : Negative------------")
		# 	print("======  :  "+self.important_title[i0] + "  : ======")
		# 	pprint(self.important_news[i0])
		# 	l+=1

	def to_db(self):
		_lst = []
		# print(self.important_title)
		# print(len(self.important_title))
		for i, t in enumerate(self.important_title):
			s = ""
			for p in self.important_news[i]:
				s = s+p+'%%'
			s = s[:-2]
			p = s
			s = ""
			for k in self.keysent_idx[i]:
				s = s+str(k)+"%%"
			s = s[:-2]

			_lst.append(  OutputNews( t, date = datetime.now().date() - timedelta(days=1), company = self.companys[i], paragraph = p, keysent = s  )   )
		# print(_lst[3].news_title)
		db.session.add_all(_lst)
		db.session.commit()


def test_url():
	result = CrawlingData.query.filter_by(date=(datetime.now().date() - timedelta(days=1)))
	return result
# 		self.q_data = result
# 		for r in result:
# 			self.url.append(r.url)
# 			self.companys.append(r.company)
# 		return result
# get = KeysentGetter()
# get.url2news()
# print(len(get.title))
# get.get_news()



