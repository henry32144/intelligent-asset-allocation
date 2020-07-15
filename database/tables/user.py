from database.database import db

'''
user_style:
    0 : active
    1 : moderate
    2 : passive
'''

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(80), unique=True, nullable=False)
    user_email = db.Column(db.String(120), unique=True, nullable=False)
    user_password = db.Column(db.String(120), nullable=False)
    user_style = db.Column(db.Integer, default=0)

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

    def __repr__(self):
        return '<User %r>' % self.user_name