import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from typing import Optional

class ClusterTimeSeries:
    """
    A class used to perform time series clustering on a pivoted DataFrame.

    Attributes
    ----------
    index : str
        The column name to be used as the index in the pivot table.
    columns : str
        The column name to be used as columns in the pivot table.
    values : str
        The column name to be used as values in the pivot table.
    n_clusters : int
        The number of clusters to form.
    n_init : int
        Number of time the k-means algorithm will be run with different centroid seeds.
    metric : str
        The distance metric to use for clustering. Default is "dtw".
    verbose : bool
        Verbosity mode.
    max_iter_barycenter : int
        Maximum number of iterations for the DBA algorithm.
    random_state : Optional[int]
        Determines random number generation for centroid initialization. Use an int for reproducible results.
    n_jobs : int
        The number of jobs to run in parallel. -1 means using all processors.
    model : Optional[TimeSeriesKMeans]
        The TimeSeriesKMeans model object after fitting.

    Methods
    -------
    fit_predict(df: pd.DataFrame) -> pd.DataFrame
        Fits the model and assigns cluster labels to the input DataFrame.
    """

    def __init__(self, 
                 index: str = 'ds', 
                 columns: str = 'rest_id', 
                 values: str = 'sales',
                 n_clusters: int = 2,
                 n_init: int = 2,
                 metric: str = "dtw",
                 verbose: bool = True,
                 max_iter_barycenter: int = 10,
                 random_state: Optional[int] = 42,
                 n_jobs: int = -1):
        
        # Assertions to ensure proper input types and values
        assert isinstance(index, str), "index must be a string"
        assert isinstance(columns, str), "columns must be a string"
        assert isinstance(values, str), "values must be a string"
        assert isinstance(n_clusters, int) and n_clusters > 0, "n_clusters must be a positive integer"
        assert isinstance(n_init, int) and n_init > 0, "n_init must be a positive integer"
        assert isinstance(metric, str), "metric must be a string"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert isinstance(max_iter_barycenter, int) and max_iter_barycenter > 0, "max_iter_barycenter must be a positive integer"
        assert isinstance(random_state, (int, type(None))), "random_state must be an integer or None"
        assert isinstance(n_jobs, int), "n_jobs must be an integer"
        
        self.index = index
        self.columns = columns
        self.values = values
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.metric = metric
        self.verbose = verbose
        self.max_iter_barycenter = max_iter_barycenter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[TimeSeriesKMeans] = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the TimeSeriesKMeans model to the provided DataFrame and assigns cluster labels.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame that contains the time series data.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with an additional 'cluster' column indicating the assigned cluster for each 'rest_id'.
        """
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        assert self.index in df.columns, f"{self.index} must be a column in the DataFrame"
        assert self.columns in df.columns, f"{self.columns} must be a column in the DataFrame"
        assert self.values in df.columns, f"{self.values} must be a column in the DataFrame"

        # Create pivot table for clustering
        df_pivot = df.pivot(index=self.index, columns=self.columns, values=self.values)

        # Transpose and fill missing values with zeros
        df_train = df_pivot.fillna(0).T.values

        # Clustering
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            metric=self.metric,
            verbose=self.verbose,
            max_iter_barycenter=self.max_iter_barycenter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # Predict clusters
        y_pred = self.model.fit_predict(df_train)

        cluster_df = pd.DataFrame({
            'rest_id': df_pivot.columns,
            'cluster': y_pred
        })

        # Merge cluster information back into the original DataFrame
        df = df.merge(cluster_df, on='rest_id', how='left')

        return df
