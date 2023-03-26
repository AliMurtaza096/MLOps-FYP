import os

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
with app.app_context():
    CORS(app)

 
    UPLOAD_FOLDER = '/home/ali/Desktop/FYP/MLflow/mlops_fyp/dataset/train/files'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_DATABASE_URI']= 'mysql+pymysql://admin:ali.1397@mlops-project-db.chs6rlz4ojkg.ap-south-1.rds.amazonaws.com:3306/mlops_db'
    app.config['UPLOAD_FOLDER']  =UPLOAD_FOLDER

    from WebApp import models
    from WebApp import views
    