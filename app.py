import os
import sys
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from tools import Predictor

from datetime import timedelta

face_db_path = 'face_db'
threshold = 0.7
mobilefacenet_model_path = 'save_model/mobilefacenet.pth'
mtcnn_model_path = 'save_model/mtcnn'

# SQLite URI compatible
WIN = sys.platform.startswith('win')
if WIN:
    prefix = 'sqlite:///'
else:
    prefix = 'sqlite:////'

app = Flask(__name__)

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


app.config['SQLALCHEMY_DATABASE_URI'] = prefix + os.path.join(app.root_path, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# @app.context_processor
# def inject_user():
#     # from models import Face
#     # user = User.query.first()
#     # return dict(user=user)

# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def index():
    if request.method == 'POST':
        f = request.files['file'] # 获取上传的图片
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        # 储存图片
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        img = cv2.imread(upload_path)

        predictor = Predictor(face_db_path=face_db_path, mobilefacenet_path=mobilefacenet_model_path, mtcnn_path=mtcnn_model_path,threshold=threshold)
        print("finished")
        boxes, names = predictor.recognition(img)
        print("2")
        predictor.draw_face(img, boxes, names)
    
        return render_template('upload_ok.html', userinput=names)

    return render_template('index.html')

app.run()