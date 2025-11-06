import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_jobs_per_year(df, title):
     #Get count of jobs per year
    year_counts = df['YEAR'].value_counts().sort_index()
    year_counts_df = year_counts.reset_index()
    year_counts_df.columns = ['year', 'job_count']
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("tab10", len(year_counts_df))  
    sns.barplot(data=year_counts_df, x='year', y='job_count', palette=palette)

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Number of Jobs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def stats_per_year(df):
    # Group by year and calculate summary statistics
    wait_time_summary = df.groupby('YEAR')['ELIGIBLE_WAIT_HOURS'].agg(
        mean='mean',
        median='median',
        std='std',
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75)
    ).reset_index()

    print(wait_time_summary)


def overall_stats(df):
    print(f"Total jobs:{len(df.JOB_NAME.unique())}") #number of jobs executed within the 6.5 year period
    print(f"Users:{df['USERNAME_GENID'].nunique()}") #number of distinct users within the 6.5 year period
    print(f"Projects:{df['PROJECT_NAME_GENID'].nunique()}") #number of distinct projects within the 6.5 year period


# relationships between job size and runtime, job size and wait time
class Job:
    def __init__(self, jobs, thresholds, keys, title):
        self.jobs = jobs
        self.thresholds = thresholds
        self.small_jobs_df = self.jobs[self.jobs['NODES_USED'] < thresholds[0]]
        self.medium_jobs_df = self.jobs[(self.jobs['NODES_USED'] >= thresholds[0]) & (self.jobs['NODES_USED'] < thresholds[1])]
        self.large_jobs_df = self.jobs[self.jobs['NODES_USED'] >= thresholds[1]]
        self.keys = keys
        self.title = title

    def get_job_runtime(self, facecolor, edgecolor, hatch):
        plt.figure(figsize=(8, 6))
        job_size_runtime = {
            self.keys[0]: self.small_jobs_df['RUNTIME_HOURS'].median(),
            self.keys[1]: self.medium_jobs_df['RUNTIME_HOURS'].median(),
            self.keys[2]: self.large_jobs_df['RUNTIME_HOURS'].median()
        }

        plt.bar(job_size_runtime.keys(), job_size_runtime.values(),
                facecolor=facecolor, edgecolor=edgecolor, hatch=hatch)
        plt.ylabel('Median job runtime [hours]')
        plt.xlabel('Job size [number of nodes]')
        plt.title(self.title)
        plt.tight_layout()
        plt.show()

    def get_job_waitime(self, facecolor, edgecolor, hatch):
        plt.figure(figsize=(8, 6))
        job_size_waitime = {
            self.keys[0]: self.small_jobs_df['ELIGIBLE_WAIT_HOURS'].median(),
            self.keys[1]: self.medium_jobs_df['ELIGIBLE_WAIT_HOURS'].median(),
            self.keys[2]: self.large_jobs_df['ELIGIBLE_WAIT_HOURS'].median()
        }
    
        plt.bar(job_size_waitime.keys(), job_size_waitime.values(),
                facecolor=facecolor, edgecolor=edgecolor, hatch=hatch)
        plt.ylabel('Median wait time [hours]')
        plt.xlabel('Job size [number of nodes]')
        # Use log scale for better visibility
        plt.yscale('log')
        plt.title(self.title)
        plt.tight_layout()
        plt.show()

    def get_job_runtime_and_walltime(self):
        plt.figure(figsize=(10, 6))

        # Dictionary to store median runtimes for each job size
        job_size_runtime = {}
        job_size_walltime = {}

        # Calculate median RUNTIME_HOUR and WALLTIME_HOUR for each category
        job_size_runtime[self.keys[0]] = self.small_jobs_df['RUNTIME_HOURS'].median()
        job_size_walltime[self.keys[0]] = self.small_jobs_df['WALLTIME_HOURS'].median()
        job_size_runtime[self.keys[1]] = self.medium_jobs_df['RUNTIME_HOURS'].median()
        job_size_walltime[self.keys[1]] = self.medium_jobs_df['WALLTIME_HOURS'].median()
        job_size_runtime[self.keys[2]] = self.large_jobs_df['RUNTIME_HOURS'].median()
        job_size_walltime[self.keys[2]] = self.large_jobs_df['WALLTIME_HOURS'].median()
        categories = list(job_size_runtime.keys())
        bar_width = 0.4
        x = np.arange(len(categories))

        # Plot the bar chart
        plt.bar(
            x - bar_width / 2, 
            job_size_runtime.values(), 
            width=bar_width, 
            label='Runtime [hours]', 
            facecolor="blue", 
            edgecolor="black", 
            alpha=0.8
        )
        plt.bar(
            x + bar_width / 2, 
            job_size_walltime.values(), 
            width=bar_width, 
            label='Walltime [hours]', 
            facecolor="orange", 
            edgecolor="black", 
            alpha=0.8
        )

        plt.ylabel('Median Time [hours]', fontsize=14)
        plt.xlabel('Job size [number of nodes]', fontsize=14)
        plt.xticks(x, categories, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def plot_cumulative_density(self):
        plt.figure(figsize=(10, 6))

        # Plot cumulative density for RUNTIME_HOUR
        sorted_runtime = np.sort(self.jobs['RUNTIME_HOURS'])
        cumulative_runtime = np.arange(1, len(sorted_runtime) + 1) / len(sorted_runtime)
        plt.plot(
            sorted_runtime, 
            cumulative_runtime, 
            label='Runtime [hours]', 
            color='blue', 
            linewidth=2
        )

        # Plot cumulative density for WALLTIME_HOUR
        sorted_walltime = np.sort(self.jobs['WALLTIME_HOURS'])
        cumulative_walltime = np.arange(1, len(sorted_walltime) + 1) / len(sorted_walltime)
        plt.plot(
            sorted_walltime, 
            cumulative_walltime, 
            label='Walltime [hours]', 
            color='orange', 
            linewidth=2
        )

        plt.xlabel('Time [hours]', fontsize=14)
        plt.ylabel('Cumulative Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.7, linestyle='--')
        plt.tight_layout()
        plt.show()


def plot_by_job_size(df, thresholds, keys, title, 
                     facecolor_runtime, edgecolor_runtime, hatch,
                     facecolor_waittime, edgecolor_waittime):
    job_obj = Job(df, thresholds, keys, title)              
    job_obj.get_job_runtime(facecolor_runtime, edgecolor_runtime, hatch)      
    job_obj.get_job_waitime(facecolor_waittime, edgecolor_waittime, hatch)
    job_obj.get_job_runtime_and_walltime()
    job_obj.plot_cumulative_density()


def walltime_vs_waittime(df, title):
    #Walltime vs. waittime
    plt.figure(figsize=(8, 6))
    plt.scatter(df['WALLTIME_HOURS'], df['ELIGIBLE_WAIT_HOURS'])
    plt.title(title)
    plt.xlabel('Walltime [Hour]')
    plt.ylabel('Wait time')
    plt.show()


# Median walltime, wait time and run time of different queues
def get_queue_time(x, title, list_queues, list_names):
    # Theta: 
    # list_queues = ['default', 'backfill', ['debug-cache-quad', 'debug-flat-quad']]
    # list_names = ['default', 'backfill', 'debug queue', 'others']
    # Polaris: 
    # list_queues = ['small', 'medium', 'large', 'backfill-small', 'backfill-medium', 'backfill-large']
    # list_names = ['small', 'medium', 'large', 
    #           'backfill-small', 'backfill-medium', 'backfill-large', 'others']
    list_dicts = []
    flat_list_queues = []
    for q in list_queues:
        dict = {}
        if type(q) is list:
            queue = x[x['QUEUE_NAME'].isin(q)]
            flat_list_queues = flat_list_queues + q
        else:
            queue = x[x['QUEUE_NAME'] == q]
            flat_list_queues.append(q)
        dict['Wall time'] = queue['WALLTIME_HOURS'].median()
        dict['Run time'] = queue['RUNTIME_HOURS'].median()
        dict['Wait time'] = queue['ELIGIBLE_WAIT_HOURS'].median()

        list_dicts.append(dict)
    other_queue = x[~x['QUEUE_NAME'].isin(flat_list_queues)]
    other_dict = {}
    other_dict['Wall time'] = other_queue['WALLTIME_HOURS'].median()
    other_dict['Run time'] = other_queue['RUNTIME_HOURS'].median()
    other_dict['Wait time'] = other_queue['ELIGIBLE_WAIT_HOURS'].median()
    list_dicts.append(other_dict)

    # Create DataFrame
    queue_df = pd.DataFrame(
        list_dicts, index=list_names
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    queue_df.plot.bar(ax=ax, title=title)
    ax.set_xlabel('Queue')
    ax.set_ylabel('Median Time (hours)')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


def plot_events_per_day(df, title):
    # Group events by date and count the number of events for each day
    daily_count = df.groupby(df['START_TIMESTAMP'].dt.date).size()
    #daily_count = jobs.groupby(jobs.index.date).size()

    # Plotting the daily count of events
    plt.figure(figsize=(8, 6))
    plt.plot(daily_count.index, daily_count.values, linestyle='-')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Jobs', fontsize=14)
    plt.title(title)
    #plt.grid(True)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14) 
    plt.tight_layout()
    plt.show()


def jobs_per_queue(df, title):
    #Check job queue sizes
    plt.figure(figsize=(15,10))
    df['QUEUE_NAME'].value_counts().sort_values(ascending=False).nlargest(10).plot(kind='bar', width=0.7)
    #plt.yscale("log")
    plt.xlabel('Queue Name')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def waittime_per_queue(df, title):
    #Wait time per job queue
    plt.figure(figsize=(15,10))
    df['ELIGIBLE_WAIT_HOURS'].groupby(df['QUEUE_NAME']).median().sort_values(ascending=False).nlargest(15).plot(kind='bar', width=0.7)
    plt.yscale("log")
    plt.ylabel('Job Queue')
    plt.ylabel('Wait time')
    plt.title(title)
    plt.show()


def nodes_vs_waittime(df, title):
    # Plotting the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['NODES_REQUESTED'], df['QUEUED_WAIT_SECONDS'])
    plt.xlabel('Requested nodes')
    plt.ylabel('Wait time (seconds)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def nodes_vs_walltime(df, title):
    # Plotting the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['NODES_REQUESTED'], df['WALLTIME_SECONDS'])
    plt.xlabel('Requested nodes')
    plt.ylabel('Wall time (seconds)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def timestamp_vs_waittime(X, y, title):
    # Scatter plot of ELIGIBLE_WAIT_HOURS over time
    plt.figure(figsize=(10, 5))
    plt.scatter(X['QUEUED_TIMESTAMP'], y, alpha=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('Wait time (hrs)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_correlations(X, machine_name):
    
        df_to_plot=X.drop(['CORES_REQUESTED','YEAR','USERNAME_GENID_ENC','COBALT_NUM_TASKS','IS_SINGLE',
                           'DAY_NAME_ENC','QUEUED_DATE_ID','JOBS_QUEUED','JOBS_RUNNING','MODE_ENC',
                            'IS_WEEKEND','IS_NIGHT'], axis=1)

        columns = {'ELIGIBLE_WAIT_HOURS':'Wait time','QUEUE_NAME_ENC': 'Queue', 'NODES_REQUESTED':'Nodes_requested',
            'WALLTIME_HOURS':'Wall time', 'REQUESTED_CORE_HOURS':'Core_hours','HOUR':'Hour', 'DAY':'Day','MONTH':'Month',
            'YEAR':'Year','EXIT_CODE':'Exit status','PROJECT_NAME_GENID_ENC':'project', 'CAPABILITY_ENC':'capability_job', 
           'COBALT_NUM_TASKS':'cobalt_num_task', 'ALLOCATION_AWARD_CATEGORY_ENC':'Award'}

        df_to_plot=df_to_plot.rename(columns=columns)
        correlation_matrix = df_to_plot.corr()
        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(machine_name, fontsize=16)
        plt.show()

def plot_PCA(pca):
    print(f"Cummulative variance (percent:)/n/t{pca.explained_variance_ratio_.cumsum()*100}") #Percentage of variance explained by each of the PCs
    plt.figure(figsize=(10,8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Principal Components')
    plt.ylabel('cumulative explained variance');
