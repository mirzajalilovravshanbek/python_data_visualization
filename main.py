# main.py

import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
import unittest

from database_handler import DatabaseHandler, TrainingData, IdealFunctions, TestData
from regression_handler import RegressionHandler
from mapping_visualization import DataMappingVisualization

# Set up database
db_uri = "sqlite:///mydb.db"
db_handler = DatabaseHandler(db_uri)
db_handler.load_training_data("csv_files/train.csv")
db_handler.load_ideal_functions("csv_files/ideal.csv")
db_handler.load_test_data("csv_files/test.csv")

# Perform least squares fit for training data
regression_handler = RegressionHandler()
training_data = db_handler.get_training_data()
x_values = [data.x for data in training_data]
y_values = [[data.y1, data.y2, data.y3, data.y4] for data in training_data]
params_list = [regression_handler.least_squares_fit(x_values, y_values_column) for y_values_column in zip(*y_values)]

# Save the obtained parameters to the database
with db_handler.Session() as session:
    for params, training_data_row in zip(params_list, training_data):
        training_data_row.params = str(params)
    session.commit()

# Map test data and visualize results
data_handler = DataMappingVisualization(db_uri)
data_handler.map_test_data()
data_handler.visualize_data()
db_handler.map_training_data()
db_handler.print_training_ideal_mapping()

# Import necessary libraries (if not imported already)


class IdealFunction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['x'].tolist()
        self.y_values = [data[f'y{i}'].tolist() for i in range(1, 51)]


class TrainingData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['x'].tolist()
        self.y_values_list = [data[f'y{i}'].tolist() for i in range(1, 5)]


class TestData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['x'].tolist()
        self.y_values = data['y'].tolist()


class Database:
    def __init__(self):
        self.engine = create_engine('sqlite:///mydb.db')
        self.conn = self.engine.connect()

    def load_training_data(self, training_data):
        df = pd.DataFrame({
            'x': training_data.x_values,
            'y1': training_data.y_values_list[0],
            'y2': training_data.y_values_list[1],
            'y3': training_data.y_values_list[2],
            'y4': training_data.y_values_list[3],
        })
        df.to_sql('training_data', con=self.conn, index=False, if_exists='replace')

    def load_ideal_functions(self, ideal_functions):
        df = pd.DataFrame({
            'x': ideal_functions.x_values,
            **{f'y{i}': ideal_functions.y_values[i - 1] for i in range(1, 51)}
        })
        df.to_sql('ideal_functions', con=self.conn, index=False, if_exists='replace')

    def load_test_data(self, test_data):
        df = pd.DataFrame({
            'x': test_data.x_values,
            'y': test_data.y_values,
        })
        df.to_sql('test_data', con=self.conn, index=False, if_exists='replace')

    def map_test_data(self, ideal_functions, test_data):
        results = []
        for _, test_row in pd.read_sql('test_data', con=self.conn).iterrows():
            x_test, y_test = test_row['x'], test_row['y']
            chosen_ideal_func = None
            min_deviation = float('inf')

            for i, ideal_row in pd.read_sql('ideal_functions', con=self.conn).iterrows():
                x_ideal, *y_ideal = ideal_row
                deviation = sum((y_test - y_i) ** 2 for y_i in y_ideal) ** 0.5

                if deviation < min_deviation:
                    min_deviation = deviation
                    chosen_ideal_func = i + 1

            results.append({
                'x': x_test,
                'y': y_test,
                'Delta Y': min_deviation,
                'No. of Ideal Func': chosen_ideal_func,
            })

        results_df = pd.DataFrame(results)
        results_df.to_sql('results', con=self.conn, index=False, if_exists='replace')

    def save_results(self, results):
        results_df = pd.DataFrame(results)
        results_df.to_sql('results', con=self.conn, index=False, if_exists='replace')


class DataVisualization:
    def __init__(self, database):
        self.database = database

    def visualize_data(self):
        results_df = pd.read_sql('results', con=self.database.conn)

        p1 = figure(title='Training Data Visualization', x_axis_label='x', y_axis_label='y')
        p1.circle('x', 'y1', source=ColumnDataSource(results_df), size=10, color='blue', legend_label='Y1')
        p1.circle('x', 'y2', source=ColumnDataSource(results_df), size=10, color='green', legend_label='Y2')
        p1.circle('x', 'y3', source=ColumnDataSource(results_df), size=10, color='red', legend_label='Y3')
        p1.circle('x', 'y4', source=ColumnDataSource(results_df), size=10, color='orange', legend_label='Y4')
        p1.legend.title = 'Training Functions'

        p2 = figure(title='Test Data Visualization', x_axis_label='x', y_axis_label='y')
        p2.circle('x', 'y', source=ColumnDataSource(results_df), size=10, color='purple', legend_label='Test Data')
        p2.line('x', 'y1', source=ColumnDataSource(results_df), line_width=2, line_color='blue',
                legend_label='Chosen Ideal Function Y1')
        p2.line('x', 'y2', source=ColumnDataSource(results_df), line_width=2, line_color='green',
                legend_label='Chosen Ideal Function Y2')
        p2.line('x', 'y3', source=ColumnDataSource(results_df), line_width=2, line_color='red',
                legend_label='Chosen Ideal Function Y3')
        p2.line('x', 'y4', source=ColumnDataSource(results_df), line_width=2, line_color='orange',
                legend_label='Chosen Ideal Function Y4')
        p2.legend.title = 'Test Data and Chosen Ideal Functions'

        p3 = figure(title='Deviation Visualization', x_axis_label='x', y_axis_label='Delta Y')
        p3.line('x', 'Delta Y', source=ColumnDataSource(results_df), line_width=2, line_color='green',
                legend_label='Deviation')
        p3.legend.title = 'Deviation'

        layout = gridplot([[p1, p2], [p3]])
        output_file('visualization.html')
        show(layout)


class TestProject(unittest.TestCase):
    def test_ideal_function(self):
        x_values = [1, 2, 3, 4]
        y_values = [5, 6, 7, 8]
        ideal_function = IdealFunction(x_values, y_values)
        self.assertEqual(ideal_function.x_values, x_values)
        self.assertEqual(ideal_function.y_values, y_values)

    def test_training_data(self):
        x_values = [1, 2, 3, 4]
        y_values_list = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        training_data = TrainingData(x_values, y_values_list)
        self.assertEqual(training_data.x_values, x_values)
        self.assertEqual(training_data.y_values_list, y_values_list)

    def test_test_data(self):
        x_values = [1, 2, 3, 4]
        y_values = [5, 6, 7, 8]
        test_data = TestData(x_values, y_values)
        self.assertEqual(test_data.x_values, x_values)
        self.assertEqual(test_data.y_values, y_values)

    def test_database(self):
        db = Database()

        x_values_train = [1, 2, 3, 4]
        y_values_list_train = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        training_data = TrainingData(x_values_train, y_values_list_train)
        db.load_training_data(training_data)

        x_values_ideal = [1, 2, 3, 4]
        y_values_list_ideal = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        ideal_functions = IdealFunction(x_values_ideal, y_values_list_ideal)
        db.load_ideal_functions(ideal_functions)

        x_values_test = [1, 2, 3, 4]
        y_values_test = [5, 6, 7, 8]
        test_data = TestData(x_values_test, y_values_test)
        db.load_test_data(test_data)

    def test_data_visualization(self):
        db = Database()
        visualization = DataVisualization(db)


if __name__ == "__main__":
    training_data = TrainingData("train.csv")
    ideal_functions = IdealFunction("ideal.csv")
    test_data = TestData("test.csv")

    db = Database()
    db.load_training_data(training_data)
    db.load_ideal_functions(ideal_functions)
    db.load_test_data(test_data)

    db.map_test_data(ideal_functions, test_data)

    results_df = pd.read_sql('results', con=db.conn)
    db.save_results(results_df)
    visualization = DataVisualization(db)
    visualization.visualize_data()

#%%
