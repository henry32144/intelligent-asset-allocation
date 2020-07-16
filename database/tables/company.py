import requests
from bs4 import BeautifulSoup
from database.database import db
import pandas as pd


class Company(db.Model):
    __tablename__ = 'company'

    id_ = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50))
    company_name = db.Column(db.String(50))
    industry = db.Column(db.String(50))

    def __init__(self, symbol, company_name, industry):
        self.symbol = symbol
        self.company_name = company_name
        self.industry = industry
    
    def __repr__(self):
        return "Get Company: {}!".format(self.symbol)


def crawl_sp500_info():
    sp500_wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(sp500_wiki_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='wikitable sortable')
    trs = table.find_all('tr')

    symbols = []
    company_names = []
    industries = []

    for idx, tr in enumerate(trs):
        if idx != 0:
            tds = tr.find_all('td')
            symbols.append(tds[0].text)
            company_names.append(tds[1].text)
            industries.append(tds[3].text)

    stock_data_df = pd.DataFrame({
        'symbol': symbols,
        'company': company_names,
        'industry': industries
    })
    
    return stock_data_df
            
def save_company():
    stock_data_df = crawl_sp500_info()

    stock_list = []
    for idx, row in stock_data_df.iterrows():
        stock_list.append(Company(row['symbol'], row['company'], row['industry']))

    db.session.add_all(stock_list)
    db.session.commit()

    # db.session.add_all(symbols)
    # db.session.add_all(company_names)
    # db.session.add_all(industries)
    # db.session.commit()