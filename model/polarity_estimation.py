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
# import os
import pandas as pd
import re
import spacy
import pysentiment as ps
# from tqdm import tqdm
# from database.tables.crawling_data import CrawlingData
# from database.tables.output_news import OutputNews
# import json
from pprint import pprint
# from sqlalchemy import Column, Integer, String, DateTime, Date
# from datetime import datetime,timedelta



def cool_function(url):
	resp = requests.get(url)
	soup = BeautifulSoup(resp.text, 'html.parser')
	paragraph = soup.find_all('p')
	paragraph = [ p.text for p in paragraph  ]
	paragraph = paragraph[1:-1]
	res_paragraph = "".join(paragraph)
	# pprint(paragraph)
	hiv4 = ps.hiv4.HIV4()
	po = []
	for p in paragraph:
		tokens = hiv4.tokenize(p)
		s = hiv4.get_score(tokens)
		# print(tokens)
		po.append(s['Polarity'])
		# print(s)
	res = []
	for i, p in enumerate(po):
		if(	float(p) >= 0.85 or float(p) <=-0.85):
			res.append(paragraph[i])
	res = "".join(res)
	# print(res_paragraph)
	# print("===================================================")
	# print(res)
	#res_paragraph : complete paragraph string
	# res : important sentence string
	return res_paragraph, res


'''
['1 Min Read',
 'March 14 (Reuters) - Apple Inc: ',
 '* APPLE INC SAYS NOT ALLOWING ENTERTAINMENT OR GAME APPS WITH COVID-19 AS '
 'THEME; ONLY RECOGNIZED ENTITIES SHOULD SUBMIT APP FOR APP STORE EVALUATION '
 'Source text: apple.co/2IKlI7R Further company coverage:',
 'All quotes delayed a minimum of 15 minutes. See here for a complete list of '
 'exchanges and delays.',
 '© 2020 Reuters. All Rights Reserved.']
'''

url = 'https://www.reuters.com/article/idUSFWN2B61K2'
url2 = 'https://www.reuters.com/article/idUST9N27L01G'
url3 = 'https://www.reuters.com/article/idUSKBN1XB3AF'
url4 = 'https://www.reuters.com/article/idUSL1N0U01RW20141216'
cool_function(url3)
cool_function(url2)
cool_function(url)
cool_function(url4)