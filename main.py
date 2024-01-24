# main.py

from database_handler import DatabaseHandler, TrainingData, IdealFunctions, TestData, TrainingIdealMapping, TestIdealMapping
from regression_handler import RegressionHandler
from mapping_visualization import DataMappingVisualization

# Set up database
db_uri = "sqlite:///mydb.db"
db_handler = DatabaseHandler(db_uri)
# Load training data, ideal functions, and test data into the database
db_handler.load_training_data("csv_files/train.csv")
db_handler.load_ideal_functions("csv_files/ideal.csv")
db_handler.load_test_data("csv_files/test.csv")

# Perform least squares fit for training data
regression_handler = RegressionHandler()

# Fetch training data from the database
training_data = db_handler.get_training_data()

# Extract x and y values for each training function
x_values = [data.x for data in training_data]
y_values = [[data.y1, data.y2, data.y3, data.y4] for data in training_data]

# Perform least squares fit for each training function
params_list = [regression_handler.least_squares_fit(x_values, y_values_column) for y_values_column in zip(*y_values)]

# Save the obtained parameters to the database
with db_handler.Session() as session:
    # Update the 'params' column in the TrainingData table
    for params, training_data_row in zip(params_list, training_data):
        training_data_row.params = str(params)  # Convert params to string or use a suitable serialization method

    session.commit()

# Use training_data as needed
for data_point in training_data:
    print(f"X: {data_point.x}, Y1: {data_point.y1}, Y2: {data_point.y2}, Y3: {data_point.y3}, Y4: {data_point.y4}")

# Map test data and visualize results
data_handler = DataMappingVisualization(db_uri)
data_handler.map_test_data()
data_handler.visualize_data()
db_handler.map_training_data()

# Print training_ideal_mapping table
db_handler.print_training_ideal_mapping()

#%%
