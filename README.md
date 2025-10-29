# Workflow
## Step 1: Stack Raw Data 
Stack the CSVs together - this only has to happen once.
```
# stack the CSV files and save them as one big csv file
in_path = "./DIM_JOB_COMPOSITE/Theta/" 
# or use default path: os.getcwd()
df = data_prep.stack_data(prefix="Theta", in_path=in_path)
df.head()
```
```
in_path = "./DIM_JOB_COMPOSITE/Polaris/" 
# or use default path: os.getcwd()
df = data_prep.stack_data(prefix="Polaris", in_path=in_path)
df.head()
```

## Step 2: Feature Engineering
Load the stacked raw data, apply a feature engineering function, and save the "engineered" data (`Theta_full_engineered_data.csv` and `Polaris_full_engineered_data.csv`). To make things clear, we apply the same feature engineering function to both Theta and Polaris datasets. This is currently in a notebook in order to make it easier to explore feature engineering options. However, it could be entirely moved to a Python file. 

Notebooks: `Feature_Engineering_Theta.ipynb` and `Feature_Engineering_Polaris.ipynb`.

## Step 3: Plotting and Exploration
Load the "engineered" data saved in the previous step (`Theta_full_engineered_data.csv` and `Polaris_full_engineered_data.csv`) and explore the data - create plots for the paper. 

Notebooks: `Job_xteristics_Theta.ipynb` and `Job_Xteristics_Polaris.ipynb`

## Step 4: Final Data Prep
Load the "engineered" data again and apply more transformations to make it ready for model training. 
- Filter out jobs that you want to exclude for model training/testing.
- Select only certain columns, because not all are suitable for model training.
- Encode categorical variables.
- Split into training and test sets.
- Remove outliers from the training data.
- Plot a correlation matrix on the training data (without outliers).
- Select features using PCA + clustering.
Then save the final data for training/testing: `Theta_X_train.csv`, `Theta_y_train.csv`, `Theta_X_test.csv`, `Theta_y_test.csv` and the equivalents for Polaris.

Notebooks: `Training_Data_Prep_Theta.ipynb` and `Training_Data_Prep_Polaris.ipynb`.

## Step 5: Train Models
Notebooks: `Queue_wait_time_prediction_Theta.ipynb` and `Queue_wait_time_prediction_polaris.ipynb`




