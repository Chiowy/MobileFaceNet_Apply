import os
import time
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from detection.face_detect import MTCNN


import torch
import numpy as np
import cv2
from PIL import ImageDraw, ImageFont, Image

engine = create_engine('sqlite:///database.db')
Session = sessionmaker(bind=engine)
session = Session()

class Predictor:
    def __init__(self, mtcnn_path, mobilefacenet_path, face_db_path, threshold):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_path) # 加载mtcnn人脸预测模型
        self.device = torch.device("cuda")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_path)
        self.model.to(self.device)
        self.model.eval()


    def load_face_db(self, face_db_path):
        faces_db = {}
        for dir_path in os.listdir(face_db_path):
            for path in os.listdir(os.path.join(face_db_path, dir_path)):
                name = os.path.basename(path).split('.')[0]
                image_path = os.path.join(face_db_path, dir_path, path)
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1) # 从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式
                imgs, _ = self.mtcnn.infer_image(img)
                if imgs is None or len(imgs) > 1:
                    print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                    continue
                imgs = self.process(imgs)
                feature = self.infer(imgs[0])
                faces_db[name] = feature[0][0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]

        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            # 执行预测
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def recognition(self, img):
        s = time.time()
        imgs, boxes = self.mtcnn.infer_image(img)
        print('人脸检测时间：%dms' % int((time.time() - s) * 1000))
        if imgs is None:
            return None, None
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        s = time.time()
        features = self.infer(imgs)
        print('人脸识别时间：%dms' % int((time.time() - s) * 1000))
        names = []
        probs = []

        from models import Face
        # 从数据库查询所有记录的 name 字段
        database_names = session.query(Face.name).all()

        for i in range(len(features)):
            feature = features[i][0]
            results_dict = {}
            for name in database_names:
                name = name[0]
                record = session.query(Face).filter_by(name=name).first()
                serialized_feature1 = record.feature
                feature1 = pickle.loads(serialized_feature1)
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            print('人脸对比结果：', results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simfang.ttf', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, img, boxes_c, names):
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                                (corpbbox[2], corpbbox[3]), (0, 0, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] -15, color=(255, 0, 255), size=12)

        cv2.imwrite(os.path.join('static/images', 'result.jpg'), img)

