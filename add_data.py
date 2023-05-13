from app import db
from models import Face
from tools import Predictor 

import pickle  # 用于序列化数据

face_db_path = 'face_db'
threshold = 0.7
mobilefacenet_model_path = 'save_model/mobilefacenet.pth'
mtcnn_model_path = 'save_model/mtcnn'

loader = Predictor(face_db_path=face_db_path, mobilefacenet_path=mobilefacenet_model_path, mtcnn_path=mtcnn_model_path,threshold=threshold)
for name, feature in loader.load_face_db():      
    serialized_feature = pickle.dumps(feature)        
    face = Face(name=name, feature=serialized_feature)
    db.session.add(face)   

db.session.commit()                        