from flask import Blueprint, jsonify, abort, request, render_template, send_from_directory
from database.tables.user import User
from database.tables.company import Company, save_company
from database.database import db

def initialize():
  # Check company table is initialized
  if len(Company.query.limit(1).all()) == 0:
    save_company()
    print("save_company")