import re
import sys
import time
import config
import warnings
import requests
import joblib
import pysentiment as ps
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from contractions import contractions_dict
from zero_shot_learner import extend_df_with_cos_sim
from summarizer import Summarizer
warnings.filterwarnings("ignore")
sys.setrecursionlimit(1000000)


def print_time(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        ret = func(*args, **kwargs)
        e = time.time()

        print(f"Spend {e - s:.3f} s")
        return ret

    return decorated_func


def add_content(url, ratio=0.8):
    """
    Return:
        res_origin: complete paragraph string
        res_ps: important sentence string
        res_bertsum: filtered string by BertSum
    """
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
    resp = requests.get(url, timeout=10, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    paragraph = soup.find_all('p')
    paragraph = [p.text for p in paragraph]
    paragraph = paragraph[1:-1]
    res_origin = "".join(paragraph)
    return res_origin


def summary(content, threshold=0.85):
    sent_text = nltk.sent_tokenize(content)
    hiv4_function = ps.hiv4.HIV4()
    po = []
    for p in sent_text:
        tokens = hiv4_function.tokenize(p)
        s = hiv4_function.get_score(tokens)
        po.append(s['Polarity'])
    res = []
    for i, p in enumerate(po):
        if(float(p) >= threshold or float(p) <= -threshold):
            res.append(sent_text[i])
    res_ps = "".join(res)
    return res_ps


class NewsPreprocessor:
    """
    Data preprocessing class.
    """
    def __init__(self, contractions_dict, lower=True, rm_stopwords=False):
        """
        :param contractions_dict: dict
        :param lower: bool
        :param rm_stopwords: bool
        """
        self.contractions_dict = contractions_dict
        self.lower = lower
        self.rm_stopwords = rm_stopwords

    def remove_unicode(self, text):
        """
        Removes unicode strings like "\u002c" and "x96"
        """
        text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        return text

    # Function for expanding contractions
    def expand_contractions(self, text, contractions_dict):
        """
        Finding contractions. (e.g. you've -> you have)
        """
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    def remove_stopwords(self, text):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        text = pattern.sub('', text)
        return text

    def remove_digits(self, text):
        nums = set(map(int, range(10)))
        text = ''.join(i for i in text if i not in nums)
        return text

    def ultimate_clean(self, text):
        if self.lower:
            text = text.lower()
        if self.rm_stopwords:
            text = self.remove_stopwords(text)
        text = self.remove_unicode(text)
        text = self.expand_contractions(text, self.contractions_dict)
        text = self.remove_digits(text)
        return text


@print_time
def transform_df(df, sort_by, k=10):
    """
    Transform dataframe into another dataframe with top k news using zero-shot learner.
    :param df: pandas dataframe
    :param sort_by: str
    :param k: int
    :return: pandas dataframe
    """
    # Group tweets by date and aggregate into a list
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["date"], utc=True)
    df_temp['date'] = df_temp['date'].apply(lambda x: x.date())
    df_temp = df_temp.sort_values(['date', sort_by], ascending=False).groupby('date').head(100)
    df_temp = df_temp.groupby("date")['clean_title'].agg(list)
    df_temp = df_temp.reset_index(drop=False, inplace=False)
    df_temp.columns = ["date", "agg_news"]

    # Create top k tweet columns
    new_cols = ["Top {} News".format(i + 1) for i in range(k)]
    df_temp = df_temp.assign(**dict.fromkeys(new_cols, np.NaN))

    # Update every columns
    print("Start transforming dataframe...")
    for index, row in df_temp.iterrows():
        try:
            i = 1
            for news in row["agg_news"]:
                column = "Top {} News".format(i)
                df_temp.loc[index, column] = news
                i += 1
                if i > k:
                    break
        except:
            pass
    df = df_temp.drop("agg_news", axis=1)
    print("Done!")

    return df

def load_fx(file_name="./data/EURUSD1440.csv"):
    fx = pd.read_csv(file_name, sep="\t", header=None, index_col=False)
    fx.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
    fx["date"] = pd.to_datetime(fx["date"], utc=True)
    fx['date'] = fx['date'].apply(lambda x: x.date())
    fx.sort_values(by='date', inplace=True)
    fx.reset_index(drop=True, inplace=True)
    fx["label"] = fx["Close"].diff(periods=1)
    fx.dropna(inplace=True)
    fx["label"] = fx["label"].map(lambda x: 1 if float(x) >= 0 else 0)
    return fx

def load_stock(ticker_name, start_date=config.TRAIN_START_DATE):
    ticker = yf.Ticker(ticker_name)
    hist = ticker.history(period="max", start=start_date)
    hist.index = hist.index.set_names(['date'])
    hist = hist.reset_index(drop=False, inplace=False)
    hist["date"] = pd.to_datetime(hist["date"], utc=True)
    hist['date'] = hist['date'].apply(lambda x: x.date())
    hist.sort_values(by='date', inplace=True)
    hist.reset_index(drop=True, inplace=True)
    hist["ticker"] = ticker_name
    hist["label"] = hist["Close"].diff(periods=1)
    hist.dropna(inplace=True)
    hist["label"] = hist["label"].map(lambda x: 1 if float(x) >= 0 else 0)
    return hist

def load_news(file_name, labels, sort_by, k):
    """
    :param file_name: str
    :param labels: list of str (for zero-shot learner)
    :param sort_by: str (str in labels)
    :param k: int (top k news)
    :return: pandas dataframe
    """
    df = joblib.load(file_name)
    df.drop_duplicates(subset="title", inplace=True)
    preprocessor = NewsPreprocessor(contractions_dict=contractions_dict)
    df["clean_title"] = df["title"].apply(lambda x: preprocessor.ultimate_clean(x))
    df = extend_df_with_cos_sim(df=df, col="clean_title", labels=labels, sort_by=sort_by)
    df = transform_df(df=df, sort_by=sort_by, k=k)
    df.reset_index(drop=True, inplace=True)
    return df

def load_data(ticker_name, news_filename, labels, sort_by, top_k):
    stock = load_stock(ticker_name)
    news = load_news(news_filename, labels=labels, sort_by=sort_by, k=top_k)
    news_and_fx = pd.merge(news, stock, on=["date"])
    news_and_fx.set_index('date', inplace=True)
    return news_and_fx


def main():
    # Load data
    df = joblib.load("./data/sp500_top100_v1.bin")
    df.drop_duplicates(subset="title", inplace=True)
    print("Get content...")
    tqdm.pandas()
    df[["content", "ps_content", "bs_content"]] = df.progress_apply(lambda row: pd.Series(add_content(row["url"])), axis=1)

    # Clean data
    preprocessor = NewsPreprocessor(contractions_dict=contractions_dict)
    print("Start cleaning title.")
    df["clean_title"] = df["title"].apply(lambda x: preprocessor.ultimate_clean(x))
    print("Start cleaning pysentiment content.")
    df["clean_ps_content"] = df["ps_content"].apply(lambda x: preprocessor.ultimate_clean(x))
    print("Start cleaning BertSum content.")
    df["clean_bs_content"] = df["bs_content"].apply(lambda x: preprocessor.ultimate_clean(x))
    df = extend_df_with_cos_sim(df=df, col="clean_ps_content", labels=["stock", "finance"], sort_by="stock")
    print(df)

    joblib.dump(df, "./data/sp500_top100_v2.bin", compress=5)


if __name__ == "__main__":
    main()