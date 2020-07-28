from database.database import db
from database.tables.company import Company, crawl_sp500_info, save_company

'''
user_style:
    0 : active
    1 : moderate
    2 : passive
'''

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(80), nullable=False)
    user_email = db.Column(db.String(120), unique=True, nullable=False)
    user_password = db.Column(db.String(120), nullable=False)
    user_style = db.Column(db.Integer, default=0)
    user_company_selection = db.Column(db.String)

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        self.user_name = user_name
        self.user_email = user_email
        self.user_password = user_password
        self.user_style = user_style
        self.user_company_selection = concat_company(user_company_selection)


    def __repr__(self):
        return '<User %r>' % self.user_name


def concat_company( _json ): #from json data to a concated string
    s = ""
    for c in _json['Company_Selection']:
        s = s + "%%" + str(c)
    return s


def get_companys(_str): # from a concated string to a list of company names
    _lst = _str.split("%%")
    res = []
    for l in _lst :
        data = Company.query.filter_by(  symbol = l  ).all()
        res.append( data[0].company_name  )
    return res