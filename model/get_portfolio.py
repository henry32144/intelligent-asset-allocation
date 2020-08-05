from markowitz import Markowitz
from database.tables.user import User, get_companys
from database.tables.company import Company
from database.tables.portfolio import Portfolio

def return_portfolio(mode = 'basic',portfolio_id ):

	data = Portfolio.query.filter(and_(user_id == user_id, portfolio_name == portfolio_id)  )
	companys = get_companys(data.portfolio_stocks)

	if (mode == 'basic'):
		marko = Markowitz(companys)
	    all_weights = marko.get_all_weights()
	    date, all_values= marko.get_backtest_result()

	    return all_weights, all_values


	elif( mode == 'blacklitterman'):
		BL = Black_Litterman(companys)
    	all_weights = BL.get_all_weights()
    	date, all_values = BL.get_backtest_result()

    	return all_weights, all_values