import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

class RFMProxyTargetGenerator(BaseEstimator, TransformerMixin):
    """
    Task 4: Calculates RFM metrics, applies K-Means clustering, 
    and appends the 'is_high_risk' proxy target variable.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        df = X.copy()
        # Ensure TransactionStartTime is datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # Aggregate behavioral metrics per customer
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Value': 'Monetary'
        })
        
        # Scale attributes for stable clustering
        scaled_rfm = self.scaler.fit_transform(rfm)
        self.kmeans.fit(scaled_rfm)
        
        # Identify the high-risk cluster (lowest engagement: high recency, low frequency/monetary)
        cluster_centers = self.kmeans.cluster_centers_
        # Recency is col 0, Frequency col 1, Monetary col 2
        # High risk index = high recency score minus low freq/monetary scores
        risk_score = cluster_centers[:, 0] - cluster_centers[:, 1] - cluster_centers[:, 2]
        self.high_risk_cluster_idx_ = np.argmax(risk_score)
        
        return self
        
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Value': 'Monetary'
        })
        
        scaled_rfm = self.scaler.transform(rfm)
        clusters = self.kmeans.predict(scaled_rfm)
        
        # Map back to binary flag
        rfm['is_high_risk'] = (clusters == self.high_risk_cluster_idx_).astype(int)
        
        # Merge target column back into the transaction space
        return df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')

def build_feature_pipeline():
    """
    Task 3: Automated and reproducible pipeline chaining transformations.
    """
    numeric_features = ['Amount', 'Value']
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

if __name__ == "__main__":
    print("Data processing transformations initialized successfully.")