from flask import Blueprint, jsonify, abort, request, render_template, send_from_directory
from database.tables.user import User
from database.tables.company import Company, save_company
from database.tables.price import StockPrice, save_history_stock_price_to_db
from database.database import db

def initialize():
  # Check company table is initialized
  if len(Company.query.limit(1).all()) == 0:
    save_company()
    print("save_company")
  if len(StockPrice.query.limit(1).all()) == 0:
    save_history_stock_price_to_db()
    print("save_history_stock_price_to_db")