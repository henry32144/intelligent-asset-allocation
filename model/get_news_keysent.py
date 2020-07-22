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

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import spacy
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
# model = hub.load(module_url)
# print ("module %s loaded" % module_url)

class KeysentGetter():
	def __init__():
		self.title = []
		self.doc = []
		self.keysent_idx = []
		self.url = []

	def _get_all_url(self):
		

	def url2news(url):
	    resp = requests.get(url)
	    soup = BeautifulSoup(resp.text, 'html.parser')
	    paragraph = soup.find_all('p')
	    paragraph = [p.text for p in paragraph]
	    print(paragraph)
	    title = soup.find('title')
	    print(title.text)




def embed(input):
  return model(input)


def url2news(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    paragraph = soup.find_all('p')
    paragraph = [p.text for p in paragraph]
    print(paragraph)
    title = soup.find('title')
    print(title.text)

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)


def clean_text():



url = 'https://www.reuters.com/article/us-google-security/google-meet-to-roll-out-new-security-features-for-video-meetings-idUSKCN24M21M'

url2news(url)


