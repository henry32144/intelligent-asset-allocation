# Cinnamon AI

## Crawl Data
Crawling data asynchronously from Reuters.

```python
class ReutersCrawlerV3:
    def __init__(self):
        self.sp_500 = open("../data/ticker_name.txt", 'r').read().split('\n')
        self.driver_path = r"./chromedriver.exe"
        self.driver = webdriver.Chrome(self.driver_path)
        self.next_button = '//*[@id="content"]/section[2]/div/div[1]/div[4]/div/div[4]/div[1]'
    
    def parse_to_dataframe(self, query):
        """
        Parameters:
            query: str
        """
        # Open driver
        self.query = query
        self.url = "https://www.reuters.com/search/news?blob={}&dateRange=all".format(query)
        self.driver.get(self.url)
        time.sleep(2)
        # Scroll down page
        self.scroll_to_bottom()
        # Parsing
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        self.driver.quit()
        news_list = soup.find_all(name="div", attrs={"class": "search-result-content"})
        news_list_generator = self.get_news_list(news_list)
        df = pd.DataFrame(list(news_list_generator), columns=["title", "date", "query", "url"])
        df = df.drop_duplicates(subset="title")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["date"] = df["date"].apply(lambda x: x.date())
        # Add ticker column
        sp500_dict = self.get_sp500_dict()
        df["ticker"] = df["query"].apply(lambda key: sp500_dict.get(key, np.nan))
        return df
    
    def parse_all_to_dataframe(self):
        query_list = [element.split("\t")[0] for element in self.sp_500]
        data = pd.DataFrame(columns=["title", "date", "query", "url", "ticker"])
        for query in progressBar(query_list):
            try:
                df = self.parse_to_dataframe(query)
                data = pd.concat([data, df], axis=0)
            except: 
                pass
        return data

    def scroll_to_bottom(self):
        while True:
            # self.driver.find_element_by_css_selector('.search-result-more-txt').click()
            element = WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'search-result-more-txt')))
            if element.text.lower() == 'no more results':
                break
            else:
                element.click()
    
    def get_news_list(self, news_list):
        for i in range(len(news_list)):
            title = news_list[i].find(name="a").text
            date = news_list[i].find(name="h5", attrs={"class": "search-result-timestamp"}).text
            date = parser.parse(date, tzinfos={"EDT": "UTC-8", "EST": "UTC-8"})
            url = news_list[i].find(name="a").get("href")
            url = "https://www.reuters.com" + url
            yield [title, date, self.query, url]
            
    def get_sp500_dict(self):
        sp_500_dict = dict()
        for element in self.sp_500:
            sp_500_dict[element.split("\t")[0]] = element.split("\t")[1]
        return sp_500_dict


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


class AsynchronousCrawler:
    def __init__(self, lst):
        self.urls = lst

    def exception(self, request, exception):
        print("Problem: {}: {}".format(request.url, exception))

    def asynchronous(self):
        return grequests.map((grequests.get(u) for u in self.urls), exception_handler=self.exception, size=5)

    def collate_responses(self, results):
        return [self.parse(x.text)for x in results if x is not None]

    def parse(self, response_text):
        soup = BeautifulSoup(response_text, "html.parser")
        paragraph = []
        for element in soup.find_all("p"):
            paragraph.append("".join(element.find_all(text=True)))
        return "".join(paragraph[1:-2])
```

### Usage of Functions

```python
# Get data from Reuters
crawler = ReutersCrawlerV3()
data = crawler.parse_to_dataframe(query="Google")
# data = crawler.parse_all_to_dataframe()
data = data.drop_duplicates(subset="title")
data = data.drop_duplicates(subset="url")

# Get content from url
crawler = AsynchronousCrawler(data.url.tolist())
results = crawler.asynchronous()
contents = crawler.collate_responses(results)

content_list = []
for content, url in zip(contents, data.url.tolist()):
	content_list.append([content, url])
final_df = pd.merge(data, pd.DataFrame(content_list, columns=["content", "url"]), on="url")
final_df = final_df.drop_duplicates(subset="content")
joblib.dump(final_df, "../data/sp500_top100_content_v5.bin", compress=5)
```

## Data Processing
Transform dataframe ("title", "date", "query", "url", "ticker", "content") into another dataframe ("ticker", "date", "Top K News", ...).

```python
def scoring(sentence):
	lm = ps.LM()
	sents = sent_tokenize(sentence)
	res = {}
	for s in sents:
		if(s != 'empty content'):
			tokens = lm.tokenize(s)
			score = lm.get_score(tokens)
			res[s] = score['Polarity']
		else:
			res[s] = 0
	res_list = sorted(res.items(), key=lambda x: x[1])  
	res_list = [x[0] for x in res_list]
	return res_list


def transform(data, k=25):
	# Create new column (k cols)
	new_cols = ["Top {} News".format(i + 1) for i in range(k)]
	new_cols.insert(0, 'date')
	new_cols.insert(0, 'ticker')

	# Insert into result_df
	result_df = pd.DataFrame(columns=new_cols)
	# For each different tickers
	for ticker in data['ticker'].unique().tolist():
		new_df = pd.DataFrame(columns=new_cols)
		df = data[data['ticker'] == ticker]
		# For each different dates
		for dt in tqdm(df['date'].unique().tolist(), total=len(df['date'].unique().tolist())):
			context = ""
			df_date = df[df['date'] == dt]
			# Aggregate every contents in the same date
			for i0, row0 in df_date.iterrows():
				context += str(row0['content'])
			# Calculate scores for each sentence in the paragraph and return a list
			content = scoring(context)
			if(len(content) < 25):
				for _ in range(25 - len(content)):
					content.append(np.NaN)
			else:
				content = content[:25]
			content.insert(0, dt)
			content.insert(0, ticker)
			content = [tuple(content)]
			new_df = pd.DataFrame(content, columns=new_cols)
			result_df = pd.concat([new_df, result_df], ignore_index=True)
	return result_df


def check_nan(df):
    # Calculate sparsity
    sparse = df.isna().sum().sum()
    total = df.shape[0] * df.shape[1]
    ratio = sparse / total
    print("Sparsity Ratio: {}".format(ratio))

    # Plot heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(df.isnull(), cbar=True, cmap=sns.color_palette("GnBu_d"))
    plt.title("Missing Values Heatmap")
    plt.show()
```

## Split Data into Train/Valid/Test

```python
def load_stock(ticker_name, start_date="2012-01-01"):
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

def load_news(news_path):
    news = joblib.load(news_path)
    news.reset_index(drop=True, inplace=True)
    return news
```

### Usage of Function

```python
TRAIN_START_DATE = "2012-01-01"
TRAIN_END_DATE = "2015-12-31"
VALID_START_DATE = "2016-01-01"
VALID_END_DATE = "2016-12-31"
TEST_START_DATE = "2017-01-01"
TEST_END_DATE = "2020-07-01"

train = pd.DataFrame()
valid = pd.DataFrame()
test = pd.DataFrame()

news_all = load_news(news_path="../data/sp500_top100_content_base_v1.bin")

for ticker in tqdm(news_all["ticker"].unique()):
    news = news_all[news_all["ticker"] == str(ticker)]
    news = news.drop(labels="ticker", axis=1)
    stock = load_stock(str(ticker), start_date="2012-01-01")
    news_and_stock = news.merge(stock, on="date")
    news_and_stock.set_index('date', inplace=True)
    news_and_stock = news_and_stock.sort_index()
    
    train_temp = news_and_stock.loc[
        pd.to_datetime(TRAIN_START_DATE).date():pd.to_datetime(TRAIN_END_DATE).date()]
    valid_temp = news_and_stock.loc[
        pd.to_datetime(VALID_START_DATE).date():pd.to_datetime(VALID_END_DATE).date()]
    test_temp = news_and_stock.loc[
        pd.to_datetime(TEST_START_DATE).date():pd.to_datetime(TEST_END_DATE).date()]
    
    train = pd.concat([train, train_temp], axis=0)
    valid = pd.concat([valid, valid_temp], axis=0)
    test = pd.concat([test, test_temp], axis=0)
```
