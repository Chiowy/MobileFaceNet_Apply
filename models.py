from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Face(Base):
    __tablename__ = 'table'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    feature = Column(String)

    def __repr__(self):
        return f"Face(name={self.name}, feature={self.feature})"

