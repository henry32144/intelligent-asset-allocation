from markowitz import Markowitz
from database.tables.user import User, get_companys
from database.tables.company import Company
from database.tables.portfolio import Portfolio

def return_portfolio(mode = 'basic', portfolio_id ):
	if (mode == 'basic'):
		data = Portfolio.query.filter(Portfolio.portfolio_name == portfolio_id)
		companys = get_companys(data.portfolio_stocks)
		marko = Markowitz(companys)
	    all_weights = marko.get_all_weights()
	    all_values, all_return = marko.get_backtest_result()

	    print('all_weights:', all_weights)
	    print('all_values:', all_values)
	    print('all_return:', all_return)
	    return all_weights, all_values, all_return

	elif(mode == 'blacklitterman'):
		pass