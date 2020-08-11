from model.markowitz import Markowitz
from model.black_litterman import Black_Litterman
from model.equal_weight import EqualWeight
from database.tables.user import User, get_companys
from database.tables.company import Company
from database.tables.portfolio import Portfolio
import numpy as np

def return_portfolio(mode='basic', portfolio_id=None, company_symbols=[]):
	if portfolio_id is None:
		companys = company_symbols
	else:
		data = Portfolio.query.filter(and_(portfolio_name == portfolio_id)  )
		companys = get_companys(data.portfolio_stocks)

	if (mode == 'basic'):
		print("Use basic model")
		marko = Markowitz(companys)
		all_weights = marko.get_all_weights()
		date, all_values, prices = marko.get_backtest_result()

		return all_weights, all_values, date, prices


	elif( mode == 'blacklitterman'):
		print("Use blacklitterman")
		BL = Black_Litterman(companys)
		all_weights = BL.get_all_weights()
		date, all_values, prices = BL.get_backtest_result()

		return all_weights, all_values, date, prices

	elif( mode == 'equalweight'):
		print("Use Equal weight")
		EW = EqualWeight(companys)
		ratio = 100 / len(companys)
		date, all_values, prices = EW.get_backtest_result()
		all_weights = np.ones((len(companys), len(date) + 1)) * ratio
		all_weights = np.around(all_weights, 2)

		return all_weights.tolist(), all_values, date, prices