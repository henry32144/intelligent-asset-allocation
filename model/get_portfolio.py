from markowitz import Markowitz
from database.tables.user import User, get_companys
from database.tables.company import Company


def return_portfolio(mode = 'basic', user_name ):
	if (mode == 'basic'):
		data = User.query.filter_by( user_name = user_name ).first()
		companys = get_companys(data.user_company_selection)
		marko = Markowitz(companys)
	    all_weights = marko.get_all_weights()
	    all_values, all_return = marko.get_backtest_result()

	    print('all_weights:', all_weights)
	    print('all_values:', all_values)
	    print('all_return:', all_return)
	    return all_weights, all_values, all_return
