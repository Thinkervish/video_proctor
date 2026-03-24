from pymongo import MongoClient

MONGO_URL= "mongodb+srv://vishravi135_db_user:Virtusa_Hackathon@ai.6axnjyd.mongodb.net/?appName=AI"
try:
    client = MongoClient(MONGO_URL)
    db = client['proctoring']
    
    violation_logs_collection = db['violation_logs'] 
    CodeEvaluation_collection = db['Code_Detection_DB']

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    violation_logs_collection = None