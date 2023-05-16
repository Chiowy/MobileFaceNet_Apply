from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Face

engine = create_engine('sqlite:///database.db')
Session = sessionmaker(bind=engine)
session = Session()

# 删除特定 ID 的数据
id_to_delete = 13169
face = session.query(Face).get(id_to_delete)
print(face)
session.delete(face)

session.commit()
