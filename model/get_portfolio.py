from model.markowitz import Markowitz
from model.black_litterman import Black_Litterman
from database.tables.user import User, get_companys
from database.tables.company import Company
from database.tables.portfolio import Portfolio

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
		date, all_values= marko.get_backtest_result()

		return all_weights, all_values, date


	elif( mode == 'blacklitterman'):
		print("Use blacklitterman")
		BL = Black_Litterman(companys)
		all_weights = BL.get_all_weights()
		date, all_values = BL.get_backtest_result()

		return all_weights, all_values, date