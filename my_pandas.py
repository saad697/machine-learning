import pandas as pd

# Load the dataset
# Specify the correct encoding
file_path = '/Users/saaduddinbaig/Downloads/customer_churn_large_dataset.xlsx'
encoding = 'latin1'  # or 'ISO-8859-1', or other encoding based on your data

# Read the CSV file with the specified encoding
df = pd.read_csv(file_path, encoding=encoding)


# Explore the data
print(df.head())  # Display the first few rows of the dataset
print(df.info())  # Get information about columns and missing values
