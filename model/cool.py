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
import joblib
import pandas as pd
import numpy as np
import pysentiment as ps
from tqdm import tqdm
def cat( sents ):
	lm = ps.LM()
	sents = sent_tokenize(sents)
	res = {}
	for s in sents:
		if(s != 'empty content'):
			tokens = lm.tokenize(s)
			score = lm.get_score(tokens)
			res[s] = score['Polarity']
		else:
			res[s] = 0
	res = sorted(res.items(), key=lambda x: x[1])  
	res = [ x[0] for x in res  ]
	return res

# df = joblib.load('sp500_top100_v4.bin')
# df = df.drop_duplicates(subset = 'content')

def transform(data):
	qq = 0
	k = 25
	new_cols = ["Top {} News".format(i + 1) for i in range(k)]
	new_cols.insert(0, 'date')
	new_cols.insert(0, 'ticker')
	result_df = pd.DataFrame(columns = new_cols)
	for ticker in data['ticker'].unique().tolist():
		new_df = pd.DataFrame(columns = new_cols)
		df = data[data['ticker'] == ticker]
		for dt in tqdm(df['date'].unique().tolist(), total = len(df['date'].unique().tolist())):
			context = ""
			df_date = df[df['date'] == dt]
			for i0, row0 in df_date.iterrows():
				context += row0['content'] 
			# for i, row in tqdm( df.iterrows(), total = len(df) , desc = ticker + 'get sent'):
			content = cat(context)
			if( len(content) < 25 ):
				for jj in range(25 - len(content)):
					content.append(np.NaN)
			else:
				content = content[:25]
			content.insert(0, dt)
			content.insert(0,ticker)
			content = [tuple(content)]
			new_df = pd.DataFrame( content, columns = new_cols )
			result_df = pd.concat([ new_df, result_df ],ignore_index=True)
	return result_df
			# joblib.dump(result_df , 'result.bin',compress=5)
# main(df)
# joblib.dump(main(df) , 'result.bin',compress=5)
# def test():
# 	a = joblib.load('result.bin')
# 	print(a)
# test()