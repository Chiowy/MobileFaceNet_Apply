
import cv2
from tools import Predictor



face_db_path = 'face_db'
threshold = 0.7
mobilefacenet_model_path = 'save_model/mobilefacenet.pth'
mtcnn_model_path = 'save_model/mtcnn'


if __name__ == '__main__':
    img = cv2.imread('static\images\WeChat_Image_20230506210845.jpg')

    predictor = Predictor(face_db_path=face_db_path, mobilefacenet_path=mobilefacenet_model_path,
                          mtcnn_path=mtcnn_model_path, threshold=threshold)
    print("finished")
    boxes, names = predictor.recognition(img)
    print("2")
    predictor.draw_face(img, boxes, names)