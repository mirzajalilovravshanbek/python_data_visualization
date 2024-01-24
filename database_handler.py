# database_handler.py
from sqlalchemy import MetaData, create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd

Base = declarative_base(metadata=MetaData())



class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
    ideal_func_id = Column(Integer, ForeignKey('ideal_functions.id'))  # Add this line


class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = [Column(Float) for _ in range(50)]


    __table_args__ = {'extend_existing': True}

    def __init__(self, x, y):
        self.x = x
        self.y = y

class TestData(Base):
    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_func_no = Column(Integer)

class TrainingIdealMapping(Base):
    __tablename__ = 'training_ideal_mapping'
    id = Column(Integer, primary_key=True)
    training_data_id = Column(Integer, ForeignKey('training_data.id'))
    ideal_func_id = Column(Integer, ForeignKey('ideal_functions.id'))

class TestIdealMapping(Base):
    __tablename__ = 'test_ideal_mapping'
    id = Column(Integer, primary_key=True)
    test_data_id = Column(Integer, ForeignKey('test_data.id'))
    ideal_func_id = Column(Integer, ForeignKey('ideal_functions.id'))

class DatabaseHandler:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri, echo=True)
        Base.metadata.drop_all(self.engine)  # Drop existing tables
        Base.metadata.create_all(self.engine)  # Create tables
        self.Session = sessionmaker(bind=self.engine)
        print(f"Database setup complete")


    def load_training_data(self, csv_files: str= "csv_files/train.csv"):
        df = pd.read_csv("csv_files/train.csv")
        # Ensure the column names in the DataFrame match your CSV file
        data = [TrainingData(x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4']) for _, row in df.iterrows()]

        with self.Session() as session:
            session.bulk_save_objects(data)
            session.commit()
            print(f"Training data loaded")

    def load_ideal_functions(self, csv_files: str= "csv_files/ideal.csv"):
        df = pd.read_csv("csv_files/ideal.csv")
        data = [IdealFunctions(x=row['x'], y=str([row[f'y{i}'] for i in range(1, 51)])) for _, row in df.iterrows()]

        with self.Session() as session:
            session.bulk_save_objects(data)
            session.commit()
            print(f"Ideal functions loaded")

    def load_test_data(self, csv_path):
        df = pd.read_csv("csv_files/test.csv")

        # Drop any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Check if 'x' column is present in the DataFrame
        if 'x' not in df.columns:
            print("Error: 'x' column not found in the DataFrame.")
            return

        # Print the DataFrame for debugging
        print(df.head())

        # Ensure the column names in the DataFrame match your CSV file
        if 'delta_y' in df.columns:
            data = [
                TestData(x=row['x'], y=row['y'], delta_y=row['delta_y'], ideal_func_no=row['ideal_func_no'])
                for _, row in df.iterrows()
            ]
        else:
            data = [
                TestData(x=row['x'], y=row['y'], ideal_func_no=row['ideal_func_no'])
                for _, row in df.iterrows()
            ]

        with self.Session() as session:
            session.bulk_save_objects(data)
            session.commit()
            print(f"Test data loaded")
    def map_training_data(self):
        with self.Session() as session:
            # Load training data from train.csv
            df_train = pd.read_csv("csv_files/train.csv")
            training_data = [TrainingData(x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4']) for _, row in df_train.iterrows()]

            # Bulk save training data to the database
            session.bulk_save_objects(training_data)
            session.commit()

            # Load ideal functions from ideal.csv
            df_ideal = pd.read_csv("csv_files/ideal.csv")
            ideal_functions = [IdealFunctions(x=row['x'], y=str([row[f'y{i}'] for i in range(1, 51)])) for _, row in df_ideal.iterrows()]

            # Bulk save ideal functions to the database
            session.bulk_save_objects(ideal_functions)
            session.commit()

            # Map training data to ideal functions
            for training_data_row in training_data:
                ideal_func = session.query(IdealFunctions).filter_by(x=training_data_row.x).first()

                # Check if ideal_func is not None before creating the mapping entry
                if ideal_func is not None:
                    # Update the existing training data entry with the ideal_func_id
                    session.query(TrainingData).filter_by(id=training_data_row.id).update({"ideal_func_id": ideal_func.id})
                else:
                    print(f"No matching ideal function found for x={training_data_row.x}")

            session.commit()
            print(f"Training data mapped to ideal functions")





    def print_training_ideal_mapping(self):
        with self.Session() as session:
            training_ideal_mapping = session.query(TrainingIdealMapping).all()

            print("| Training Data ID | Ideal Function ID |")
            print("|-------------------|--------------------|")

            for entry in training_ideal_mapping:
                training_data_id = entry.training_data_id
                ideal_func_id = entry.ideal_func_id

                # Check if ideal_func_id is not None before printing
                if ideal_func_id is not None:
                    print(f"| {training_data_id}  | {ideal_func_id}  |")
                else:
                    print(f"| {training_data_id}  | No matching ideal function found  |")




    def get_training_data(self):
        with self.Session() as session:
            training_data = session.query(TrainingData).all()
            return training_data
