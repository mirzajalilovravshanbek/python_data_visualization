# mapping_visualization.py

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from database_handler import DatabaseHandler, IdealFunctions, TestData, TestIdealMapping

class DataMappingVisualization:
    def __init__(self, db_uri):
        self.db_handler = DatabaseHandler(db_uri)

    def map_test_data(self):
        with self.db_handler.Session() as session:
            test_data = session.query(TestData).all()

            for test_data_row in test_data:
                ideal_func = session.query(IdealFunctions).filter_by(x=test_data_row.x).first()

                if ideal_func is not None:
                    mapping_entry = TestIdealMapping(test_data_id=test_data_row.id, ideal_func_id=ideal_func.id)
                    session.add(mapping_entry)
                else:
                    print(f"No matching ideal function found for x={test_data_row.x}")

            session.commit()
            print(f"Test data mapped to ideal functions")

    def visualize_data(self):
        with self.db_handler.Session() as session:
            result = (
                session.query(TestData, IdealFunctions, TestIdealMapping)
                .join(IdealFunctions, TestIdealMapping.ideal_func_id == IdealFunctions.id)
                .filter(TestIdealMapping.test_data_id == TestData.id)
                .all()
            )

            print("| Test Data ID | X Value | Y Value | Ideal Function ID |")
            print("|--------------|---------|---------|---------------------|")

            for row in result:
                if row.TestData is not None:
                    print(f"| {row.TestData.id}  | {row.TestData.x:.2f}  | {row.TestData.y:.2f}  | "
                          f"{row.IdealFunctions.id}  |")
                else:
                    print("| None         |   None  |   None  |        None         |")

            print("Visualization complete.")

    def run(self):
        self.map_test_data()
        self.visualize_data()

if __name__ == "__main__":
    # Replace 'your_actual_db_uri' with your actual database URI
    db_uri = 'sqlite:///mydb.db'
    data_handler = DataMappingVisualization(db_uri)
    data_handler.run()
