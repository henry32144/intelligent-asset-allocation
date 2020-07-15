import re
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from contractions import contractions_dict
from zero_shot_learner import extend_df_with_cos_sim
warnings.filterwarnings("ignore")
sys.setrecursionlimit(1000000)


class NewsPreprocessor:
    """
    Data preprocessing class.
    """
    def __init__(self, contractions_dict, lower=True):
        self.contractions_dict = contractions_dict
        self.lower = lower

    def remove_unicode(self, text):
        """ Removes unicode strings like "\u002c" and "x96" """
        text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        return text

    # Function for expanding contractions
    def expand_contractions(self, text, contractions_dict):
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
        text = self.remove_unicode(text)
        text = self.expand_contractions(text, self.contractions_dict)
        text = self.remove_stopwords(text)
        text = self.remove_digits(text)
        return text

def transform_df(df, sort_by, k=10):
    """
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
    for index, row in tqdm(df_temp.iterrows(), total=df_temp.shape[0]):
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

def load_news(file_name="./data/reuters_news_amazon_v1.joblib", labels=["finance", "forex"], sort_by="finance", k=3):
    df = joblib.load(file_name)
    df.drop_duplicates(subset="title", inplace=True)
    preprocessor = NewsPreprocessor(contractions_dict=contractions_dict)
    df["clean_title"] = df["title"].apply(lambda x: preprocessor.ultimate_clean(x))
    df = extend_df_with_cos_sim(df=df, col="clean_title", labels=labels, sort_by=sort_by)
    df = transform_df(df=df, sort_by="finance", k=k)
    df.reset_index(drop=True, inplace=True)
    return df

def load_data(fx_filename, news_filename, **kwargs):
    fx = load_fx(fx_filename)
    news = load_news(news_filename)
    news_and_fx = pd.merge(news, fx, on=["date"])
    news_and_fx.set_index('date', inplace=True)
    return news_and_fx

def main():
    df = load_data()
    print(df.index.min())
    print(df.index.max())

if __name__ == "__main__":
    main()