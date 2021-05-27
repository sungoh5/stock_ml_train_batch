import os

from pymongo import MongoClient

host = os.getenv('MONGO_HOST', 'localhost')
port = os.getenv('MONGO_PORT', 27017)
db = os.getenv('MONGO_DB', 'stocks')
username = os.getenv('MONGO_USERNAME', 'root')
password = os.getenv('MONGO_PASSWORD', 'mongodb')

client = MongoClient(host=host,
                     port=int(port),
                     username=username,
                     password=password,
                     authMechanism='SCRAM-SHA-1')[db]
