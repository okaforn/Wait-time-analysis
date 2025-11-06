import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import data_exploration
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import bisect
from scipy.cluster.hierarchy import fcluster,linkage

def stack_data(prefix, in_path=os.getcwd()):
    csv_files = glob.glob(os.path.join(in_path, 'ANL-ALCF-DJC*.csv'))
    print(csv_files)

    # Read and concatenate data files
    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    out_path = os.getcwd()
    df.to_csv(os.path.join(out_path, '%s_full_raw_data.csv' % prefix))
    print(out_path)
    
    return(df)


def load_parse_df(filename):
    df = pd.read_csv(filename, index_col=0)

    #convert timestamp to datetime format
    df['QUEUED_TIMESTAMP']=pd.to_datetime(df['QUEUED_TIMESTAMP'])
    # or should we specify the format, like this, from Queue_wait_time_prediction_polaris.ipynb? 
    #df['QUEUED_TIMESTAMP']=pd.to_datetime(df['QUEUED_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
    df['START_TIMESTAMP']=pd.to_datetime(df['START_TIMESTAMP'])
    df['END_TIMESTAMP']=pd.to_datetime(df['END_TIMESTAMP'])

    # sort data frame in order of time job queued
    df=df.sort_values(by='QUEUED_TIMESTAMP', ascending=True)

    #convert string variables to float 
    float_types = ['USED_CORE_HOURS', 'REQUESTED_CORE_HOURS', 'QUEUED_WAIT_SECONDS', 'ELIGIBLE_WAIT_SECONDS', 'WALLTIME_SECONDS', 'RUNTIME_SECONDS', 'NODES_USED',
                   'NODES_REQUESTED', 'EXIT_CODE', 'CORES_USED', 'CORES_REQUESTED', 'CAPABILITY_USAGE_CORE_HOURS', 'NONCAPABILITY_USAGE_CORE_HOURS']
    for c in float_types:
        df[c] = df[c].astype(float)
    
    return df


def feature_engineering(df, prefix):
    # separate time stamp
    df['YEAR']=df['QUEUED_TIMESTAMP'].dt.year
    df['DAY']=df['QUEUED_TIMESTAMP'].dt.day
    df['MONTH']=df['QUEUED_TIMESTAMP'].dt.month
    df['DAY_NAME']=df['QUEUED_TIMESTAMP'].dt.day_name()
    df['HOUR']=df['QUEUED_TIMESTAMP'].dt.hour

    #convert times in seconds to hours
    df['RUNTIME_HOURS']=df['RUNTIME_SECONDS']/3600.00
    df['WALLTIME_HOURS']=df['WALLTIME_SECONDS']/3600.00
    df['QUEUED_WAIT_HOURS']=df['QUEUED_WAIT_SECONDS']/3600.00
    df['ELIGIBLE_WAIT_HOURS']=df['ELIGIBLE_WAIT_SECONDS']/3600.00

    # 1. How many jobs are currently running at the time the new job was queued
    # 2. How many jobs are queued( but not yet started) at the time this job was queued
   
    queued = df['QUEUED_TIMESTAMP'].values
    start = np.sort(df['START_TIMESTAMP'].values)
    end = np.sort(df['END_TIMESTAMP'].values)
    queued_sorted = np.sort(queued)

    #how many jobs are running at the time each job was queued
    #first counts how many jobs started on or before the new job was queued, then
    #Counts how many jobs already ended on or before that time, then
    #subtract to get the number of currently running jobs at each queued_time
    running = np.searchsorted(start, queued, side='right') - np.searchsorted(end, queued, side='right')
    #How many jobs queued (but not started) at the time each job was queued
    queued_not_started = np.searchsorted(queued_sorted, queued, side='right') - np.searchsorted(start, queued, side='right')

    df['JOBS_RUNNING'] = running
    df['JOBS_QUEUED'] = queued_not_started

    
    df['IS_WEEKEND'] = df['QUEUED_TIMESTAMP'].dt.dayofweek >= 5
    df['IS_NIGHT'] = df['QUEUED_TIMESTAMP'].dt.hour.isin([0,1,2,3,4,5])
    
    print(df.columns)
    print(df.shape)

    out_path = os.getcwd()
    df.to_csv(os.path.join(out_path, '%s_full_engineered_data.csv' % prefix))

    return(df)


def load_parse_engineered_df(prefix):
    out_path = os.getcwd()
    filename = os.path.join(out_path, '%s_full_engineered_data.csv' % prefix)

    # start with list used in load_parse_df. Add engineered features. (and move some to int since we really don't need float for them)
    float_types = ['USED_CORE_HOURS', 'REQUESTED_CORE_HOURS', 'QUEUED_WAIT_SECONDS', 'ELIGIBLE_WAIT_SECONDS', 'WALLTIME_SECONDS', 'RUNTIME_SECONDS', 
                    'CAPABILITY_USAGE_CORE_HOURS', 'NONCAPABILITY_USAGE_CORE_HOURS', 'RUNTIME_HOURS', 'WALLTIME_HOURS', 'QUEUED_WAIT_HOURS',
                    'ELIGIBLE_WAIT_HOURS', 'WALLTIME_RUNTIME_DIFF',  'OVERBURN_CORE_HOURS']
    int_types = ['NODES_USED', 'NODES_REQUESTED', 'EXIT_CODE', 'CORES_USED', 'CORES_REQUESTED', 'HOUR', 'DAY', 'MONTH', 'YEAR']

    # create dictionary specifying dtypes for many columns 
    dtype = dict.fromkeys(float_types, float)
    for col in int_types:
        dtype[col] = int

    # tell pandas which columns should be treated as datetime 
    parse_dates = ['QUEUED_TIMESTAMP', 'START_TIMESTAMP', 'END_TIMESTAMP']

    df = pd.read_csv(filename, index_col=0, parse_dates=parse_dates, dtype=dtype)

    return df


def subset_columns_training(df):

    #select needed columns
    columns=['ELIGIBLE_WAIT_HOURS', 'QUEUED_TIMESTAMP', 'QUEUE_NAME', 'USERNAME_GENID','QUEUED_DATE_ID',  'NODES_REQUESTED',
             'CORES_REQUESTED','WALLTIME_HOURS', 'REQUESTED_CORE_HOURS','HOUR', 'DAY','MONTH','YEAR','CAPABILITY',
             'PROJECT_NAME_GENID', 'MODE','DAY_NAME', 'ALLOCATION_AWARD_CATEGORY',
             'COBALT_NUM_TASKS','IS_SINGLE',  'JOBS_QUEUED', 'JOBS_RUNNING',
             'IS_WEEKEND','IS_NIGHT']
    df=df[columns]
    print(df.shape)

    return(df)


def encode_categorical_variables(df):
    #encode the categorical variables
    categorical_cols = [
        'QUEUE_NAME',
        'CAPABILITY',
        'PROJECT_NAME_GENID',
        'ALLOCATION_AWARD_CATEGORY',
        'DAY_NAME',
        'MODE',
        'USERNAME_GENID'
    ]
    encoders = {}
    # Encode each categorical column
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[f'{col}_ENC'] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    # Drop original categorical columns
    df.drop(columns=categorical_cols, inplace=True)
    print(df.shape)

    return df



def split_dataset(df):
    """
    Splits the input DataFrame into training, testing, and evaluation sets.
    Returns the raw splits and corresponding features (X) and targets (y).
    """

    # First split: 60% train, 40% temp (for test + eval)
    train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)

    # Second split: 50% of temp goes to test, 50% to eval (i.e 20% each overall )
    test_set, eval_set = train_test_split(temp_set, test_size=0.5, random_state=42)

    print(f'Train set: {train_set.shape[0]}')
    print(f'Test set: {test_set.shape[0]}')
    print(f'Eval set: {eval_set.shape[0]}')

    # Prepare feature and target sets
    def split_features_targets(data):
        X = data.drop(['ELIGIBLE_WAIT_HOURS'], axis=1)
        y = data['ELIGIBLE_WAIT_HOURS']
        return X, y

    X_train, y_train = split_features_targets(train_set)
    X_test, y_test = split_features_targets(test_set)
    X_eval, y_eval = split_features_targets(eval_set)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X_eval shape: {X_eval.shape}, y_eval shape: {y_eval.shape}")

    return (train_set, test_set, eval_set,
            X_train, y_train, X_test, y_test, X_eval, y_eval)




def outlier_removal(X_train, y_train, machine_name):
    #outlier detection and removal
    th1=0.05
    th3=0.95
    quartile1 = y_train.quantile(th1)
    quartile3 = y_train.quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr

    X_train_without_outliers=X_train.loc[(y_train > lower_limit) & (y_train<upper_limit)]
    y_train_without_outliers=y_train.loc[(y_train > lower_limit) & (y_train<upper_limit)]
    print(X_train_without_outliers.shape)
    print(y_train_without_outliers.shape)

    data_exploration.timestamp_vs_waittime(X_train, y_train, machine_name)
    data_exploration.timestamp_vs_waittime(X_train_without_outliers, y_train_without_outliers, machine_name)
    print(y_train_without_outliers.shape)

    return X_train_without_outliers, y_train_without_outliers


def remove_timestamp(df):
    return df.drop(['QUEUED_TIMESTAMP'], axis=1)


def rescale_data(X_train, X_test, X_eval):

    #Data Standardization; Standardize features
    sc = StandardScaler()
    X_train_rescaled = sc.fit_transform(X_train) #fit on train features
    # Apply same transformation to test  and eval data
    X_test_rescaled = sc.transform(X_test)
    X_eval_rescaled = sc.transform(X_eval)
    print(f"X_test size:{X_test.shape}")
    print(f"X_test_rescaled size:{X_test_rescaled.shape}")
    print(f"X_eval_rescaled size:{X_eval_rescaled.shape}")

    return  X_train_rescaled, X_test_rescaled, X_eval_rescaled

def compute_PCA(X_train):
    pca = PCA()
    pca.fit(X_train)
    
    return pca

def pca_feature_selection(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_.T, 
        index=feature_names,  # Use saved column names
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )

    #most contributing features in each component
    for row in range(len(loadings)):
        temp=np.argpartition(-(pca.components_[row]),4) #get the indices of the top 4 values in each row
        indices=temp[np.argsort((- pca.components_[row])[temp])][:6] #sort the indices in ascending order, view a portion
        #print(f'Component{row}: {train_set.columns[indices].to_list()}')


    #Perform Hierarchical Clustering on PCA Loadings
    linkage_matrix = linkage(loadings, method='ward')
    num_clusters = 22
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    cluster_labels
    #Select Representative Features (Highest PCA Loadings on PC1)
    cluster_feature_map = {}
    for feature, cluster in zip(loadings.index, cluster_labels):
        cluster_feature_map.setdefault(cluster, []).append(feature)

    selected_features = [max(features, key=lambda f: abs(loadings.loc[f, 'PC1']))
                        for cluster, features in cluster_feature_map.items()]
    
    return selected_features


def feature_selection(X_train, X_test, X_eval, feature_names):
    print(X_train.shape)
    print(X_test.shape)
    pca = compute_PCA(X_train)
    data_exploration.plot_PCA(pca)
    selected_features = pca_feature_selection(pca, feature_names)

    # convert back to Pandas df
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_eval_df = pd.DataFrame(X_eval, columns=feature_names)

    # use only selected features
    X_train_reduced = X_train_df[selected_features]
    X_test_reduced = X_test_df[selected_features]
    X_eval_reduced = X_eval_df[selected_features]

    print(X_train_reduced.shape)
    print(X_test_reduced.shape)
    print(X_eval_reduced.shape)

    return X_train_reduced, X_test_reduced, X_eval_reduced, selected_features


def rescale_target(y_train, y_eval, y_test, machine_name):
    y_train_log= np.log1p(y_train) #np.log(x+1)
    y_eval_log= np.log1p(y_eval)
    y_test_log= np.log1p(y_test)
    return y_train_log, y_eval_log, y_test_log
    

def get_wait_time_category(x):
    if x <= 1:
        return 0
    elif x <= 2:
        return 1
    elif x <= 4:
        return 2
    elif x <= 6:
        return 3
    elif x <= 8:
        return 4
    elif x <= 10:
        return 5
    elif x <= 12:
        return 6
    elif x <= 24:
        return 7
    else:
        return 8

def process_wait_time_range(train_set, test_set, eval_set, selected_features):
    # Apply binning
    train_set['WAIT_TIME_RANGE'] = train_set['ELIGIBLE_WAIT_HOURS'].apply(get_wait_time_category)
    test_set['WAIT_TIME_RANGE'] = test_set['ELIGIBLE_WAIT_HOURS'].apply(get_wait_time_category)
    eval_set['WAIT_TIME_RANGE'] = eval_set['ELIGIBLE_WAIT_HOURS'].apply(get_wait_time_category)
    
    # print grouped value counts
    print(test_set.groupby('WAIT_TIME_RANGE')['WAIT_TIME_RANGE'].value_counts())
    print(train_set.columns)
    
    # Wait time category percentages in test set
    wait_time_percentage = test_set['WAIT_TIME_RANGE'].value_counts(normalize=True).sort_index() * 100
    print(wait_time_percentage)
    
    # Prepare selected features and labels for classification (i.e data with outlier removal and pca applied)
    train_set_reduced = train_set[selected_features + ['WAIT_TIME_RANGE']]
    test_set_reduced = test_set[selected_features + ['WAIT_TIME_RANGE']]
    eval_set_reduced = eval_set[selected_features + ['WAIT_TIME_RANGE']]
    
    X_train_class_reduced = train_set_reduced.drop(['WAIT_TIME_RANGE'], axis=1)
    y_train_class_reduced = train_set_reduced['WAIT_TIME_RANGE']
    
    X_test_class_reduced = test_set_reduced.drop(['WAIT_TIME_RANGE'], axis=1)
    y_test_class_reduced = test_set_reduced['WAIT_TIME_RANGE']

    X_eval_class_reduced = eval_set_reduced.drop(['WAIT_TIME_RANGE'], axis=1)
    y_eval_class_reduced = eval_set_reduced['WAIT_TIME_RANGE']

    # Prepare classification data without outlier removal and pca applied) 
    X_train_class = train_set.drop(['WAIT_TIME_RANGE', 'QUEUED_TIMESTAMP', 'ELIGIBLE_WAIT_HOURS'], axis=1)
    y_train_class = train_set['WAIT_TIME_RANGE']
    
    X_test_class = test_set.drop(['WAIT_TIME_RANGE', 'QUEUED_TIMESTAMP','ELIGIBLE_WAIT_HOURS'], axis=1)
    y_test_class = test_set['WAIT_TIME_RANGE']

    X_eval_class = eval_set.drop(['WAIT_TIME_RANGE', 'QUEUED_TIMESTAMP','ELIGIBLE_WAIT_HOURS'], axis=1)
    y_eval_class = eval_set['WAIT_TIME_RANGE']
    print(X_train_class.columns)
    
    
    return X_train_class_reduced, y_train_class_reduced, X_test_class_reduced, y_test_class_reduced, X_eval_class_reduced, y_eval_class_reduced, X_train_class, y_train_class, X_test_class, y_test_class,X_eval_class, y_eval_class

     

def save_final_data(X_train, y_train, X_test, y_test, X_eval, y_eval, y_train_log, y_eval_log, y_test_log, y_train_without_outliers, X_train_without_outliers, X_train_reduced, X_test_reduced, X_eval_reduced, X_train_class, y_train_class,X_test_class,y_test_class, X_eval_class, y_eval_class, X_train_class_reduced, y_train_class_reduced, X_test_class_reduced, y_test_class_reduced, X_eval_class_reduced, y_eval_class_reduced, machine_name):
    out_path = os.getcwd()
    X_train.to_csv(os.path.join(out_path, '%s_X_train.csv' % machine_name))
    y_train.to_csv(os.path.join(out_path, '%s_y_train.csv' % machine_name))
    X_test.to_csv(os.path.join(out_path, '%s_X_test.csv' % machine_name))
    y_test.to_csv(os.path.join(out_path, '%s_y_test.csv' % machine_name))
    X_eval.to_csv(os.path.join(out_path, '%s_X_eval.csv' % machine_name))
    y_eval.to_csv(os.path.join(out_path, '%s_y_eval.csv' % machine_name))
    #save log-transformed target
    y_train_log.to_csv(os.path.join(out_path, '%s_y_train_log.csv' % machine_name))
    y_eval_log.to_csv(os.path.join(out_path, '%s_y_eval_log.csv' % machine_name))
    y_test_log.to_csv(os.path.join(out_path, '%s_y_test_log.csv' % machine_name))
    #save classification data
    X_train_class.to_csv(os.path.join(out_path, '%s_X_train_class.csv' % machine_name))
    y_train_class.to_csv(os.path.join(out_path, '%s_y_train_class.csv' % machine_name))
    X_test_class.to_csv(os.path.join(out_path, '%s_X_test_class.csv' % machine_name))
    y_test_class.to_csv(os.path.join(out_path, '%s_y_test_class.csv' % machine_name))
    X_eval_class.to_csv(os.path.join(out_path, '%s_X_eval_class.csv' % machine_name))
    y_eval_class.to_csv(os.path.join(out_path, '%s_y_eval_class.csv' % machine_name))

    X_train_class_reduced.to_csv(os.path.join(out_path, '%s_X_train_class_reduced.csv' % machine_name))
    y_train_class_reduced.to_csv(os.path.join(out_path, '%s_y_train_class_reduced.csv' % machine_name))
    X_test_class_reduced.to_csv(os.path.join(out_path, '%s_X_test_class_reduced.csv' % machine_name))
    y_test_class_reduced.to_csv(os.path.join(out_path, '%s_y_test_class_reduced.csv' % machine_name))
    X_eval_class_reduced.to_csv(os.path.join(out_path, '%s_X_eval_class_reduced.csv' % machine_name))
    y_eval_class_reduced.to_csv(os.path.join(out_path, '%s_y_eval_class_reduced.csv' % machine_name))
    
    
    X_train_without_outliers.to_csv(os.path.join(out_path, '%s_X_train_without_outliers.csv' % machine_name))
    y_train_without_outliers.to_csv(os.path.join(out_path, '%s_y_train_without_outliers.csv' % machine_name))
    X_train_reduced.to_csv(os.path.join(out_path, '%s_X_train_reduced.csv' % machine_name))
    X_test_reduced.to_csv(os.path.join(out_path, '%s_X_test_reduced.csv' % machine_name)) 
    X_eval_reduced.to_csv(os.path.join(out_path, '%s_X_eval_reduced.csv' % machine_name)) 
    print(y_train_log.shape)
    

def load_parse_final_data(prefix):
    # now all of the features are float, since they were saved after rescaling

    out_path = os.getcwd()
    X_train = pd.read_csv(os.path.join(out_path, '%s_X_train.csv' % prefix), index_col=0, dtype=float)
    y_train = pd.read_csv(os.path.join(out_path, '%s_y_train.csv' % prefix), index_col=0, dtype=float)
    X_test = pd.read_csv(os.path.join(out_path, '%s_X_test.csv' % prefix), index_col=0, dtype=float)
    y_test = pd.read_csv(os.path.join(out_path, '%s_y_test.csv' % prefix), index_col=0, dtype=float)
    X_eval = pd.read_csv(os.path.join(out_path, '%s_X_eval.csv' % prefix), index_col=0, dtype=float)
    y_eval= pd.read_csv(os.path.join(out_path, '%s_y_eval.csv' % prefix), index_col=0, dtype=float)
    y_train_log = pd.read_csv(os.path.join(out_path, '%s_y_train_log.csv' % prefix), index_col=0, dtype=float)
    y_eval_log = pd.read_csv(os.path.join(out_path, '%s_y_eval_log.csv' % prefix), index_col=0, dtype=float)
    y_test_log = pd.read_csv(os.path.join(out_path, '%s_y_test_log.csv' % prefix), index_col=0, dtype=float)
    
    X_train_class = pd.read_csv(os.path.join(out_path, '%s_X_train_class.csv' % prefix), index_col=0, dtype=float)
    y_train_class = pd.read_csv(os.path.join(out_path, '%s_y_train_class.csv' % prefix), index_col=0, dtype=float)
    X_test_class = pd.read_csv(os.path.join(out_path, '%s_X_test_class.csv' % prefix), index_col=0, dtype=float)
    y_test_class = pd.read_csv(os.path.join(out_path, '%s_y_test_class.csv' % prefix), index_col=0, dtype=float)
    X_eval_class = pd.read_csv(os.path.join(out_path, '%s_X_eval_class.csv' % prefix), index_col=0, dtype=float)
    y_eval_class = pd.read_csv(os.path.join(out_path, '%s_y_eval_class.csv' % prefix), index_col=0, dtype=float)

    X_train_class_reduced = pd.read_csv(os.path.join(out_path, '%s_X_train_class_reduced.csv' % prefix), index_col=0, dtype=float)
    y_train_class_reduced = pd.read_csv(os.path.join(out_path, '%s_y_train_class_reduced.csv' % prefix), index_col=0, dtype=float)
    X_test_class_reduced = pd.read_csv(os.path.join(out_path, '%s_X_test_class_reduced.csv' % prefix), index_col=0, dtype=float)
    y_test_class_reduced = pd.read_csv(os.path.join(out_path, '%s_y_test_class_reduced.csv' % prefix), index_col=0, dtype=float)
    X_eval_class_reduced = pd.read_csv(os.path.join(out_path, '%s_X_eval_class_reduced.csv' % prefix), index_col=0, dtype=float)
    y_eval_class_reduced = pd.read_csv(os.path.join(out_path, '%s_y_eval_class_reduced.csv' % prefix), index_col=0, dtype=float)
    
    
    X_train_without_outliers = pd.read_csv(os.path.join(out_path, '%s_X_train_without_outliers.csv' % prefix), index_col=0, dtype=float)
    y_train_without_outliers = pd.read_csv(os.path.join(out_path, '%s_y_train_without_outliers.csv' % prefix), index_col=0, dtype=float)
    X_train_reduced = pd.read_csv(os.path.join(out_path, '%s_X_train_reduced.csv' % prefix), index_col=0, dtype=float)
    X_test_reduced = pd.read_csv(os.path.join(out_path, '%s_X_test_reduced.csv' % prefix), index_col=0, dtype=float)
    X_eval_reduced = pd.read_csv(os.path.join(out_path, '%s_X_eval_reduced.csv' % prefix), index_col=0, dtype=float)
    print(y_train_log.shape)

    return X_train, y_train, X_test, y_test, X_eval, y_eval, y_train_log, y_eval_log, y_test_log, y_train_without_outliers, X_train_without_outliers, X_train_reduced, X_test_reduced, X_eval_reduced,  X_train_class, y_train_class,X_test_class,y_test_class, X_eval_class, y_eval_class, X_train_class_reduced, y_train_class_reduced, X_test_class_reduced, y_test_class_reduced, X_eval_class_reduced, y_eval_class_reduced
   
