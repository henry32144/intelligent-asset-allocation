import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')


from database.tables.crawling_data import CrawlingData
from database.tables.price import StockPrice
from datetime import datetime




def get_bert_input():
	data = CrawlingData.query.filter_by(date = datetime.now().date())
	if(data == None):
		print("no new crawling data updated")
		return None
	else:
		return data



def get_price():
	price = StockPrice.query.filter_by( date = datetime.now().date())
	if(data == None):
		print("no new stock price data updated")
		return None
	else:
		return data

