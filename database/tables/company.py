import requests
from bs4 import BeautifulSoup
from database.database import db
import pandas as pd


class Company(db.Model):
    __tablename__ = 'company'

    id_ = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50), unique=True)
    company_name = db.Column(db.String(50), unique=True)
    industry = db.Column(db.String(50))

    def __init__(self, symbol, company_name, industry):
        self.symbol = symbol
        self.company_name = company_name
        self.industry = industry
    
    def __repr__(self):
        return "Get Company: {}!".format(self.symbol)

    def to_json(self):
        return {
                'id_': self.id_,
                'symbol': self.symbol,
                'company_name': self.company_name,
                'industry': self.industry,
                }
                
def company_name2dict():
    with open('./database/tables/ticker_name.txt', 'r') as f:
        _dict = {}
        lines = f.readlines()
        for l in lines:
            c = l.split('\t')
            _dict[c[1].replace('\n','')] = c[0]
    f.close()
    return _dict



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
            symbols.append(tds[0].text.replace('\n',''))
            company_names.append(tds[1].text.replace('\n',''))
            industries.append(tds[3].text.replace('\n',''))

    stock_data_df = pd.DataFrame({
        'symbol': symbols,
        'company': company_names,
        'industry': industries
    })


    symbol = []
    company = []
    industry = []
    stock_dict = company_name2dict()

    sym = list(stock_dict.keys())
    com = list(stock_dict.values())


    for idx, row in stock_data_df.iterrows():
        if row[0] in sym:
            symbol.append(row[0])
            company.append(stock_dict[row[0]])
            industry.append(row[2])


    # for row in stock_data_df.iterrows():
    #     if (row[0] in com ):
    #         symbol.append( row[0] )
    #         company.append(stock_dict[ row[0] ])
    #         industry.append(row[2])

    cool_df = pd.DataFrame({
        'symbol' : symbol,
        'company':company,
        'industry':industry
        })

    return cool_df
            
def save_company():
    stock_data_df = crawl_sp500_info()

    stock_list = []
    for idx, row in stock_data_df.iterrows():
        stock_list.append(Company(row['symbol'], row['company'], row['industry']))

    db.session.add_all(stock_list)
    db.session.commit()




