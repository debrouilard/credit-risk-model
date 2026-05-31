import pytest
import pandas as pd
import numpy as np
from src.data_processing import build_feature_pipeline, RFMProxyTargetGenerator

@pytest.fixture
def sample_data():
    """Generates dummy transactional workspace context for verification testing."""
    return pd.DataFrame({
        'TransactionId': [f'T{i}' for i in range(1, 6)],
        'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C2'],
        'Amount': [1000.0, -200.0, 5000.0, 50.0, -100.0],
        'Value': [1000.0, 200.0, 5000.0, 50.0, 100.0],
        'ProductCategory': ['airtime', 'utility', 'financial_services', 'airtime', 'utility'],
        'ChannelId': ['Channel_3', 'Channel_3', 'Channel_1', 'Channel_3', 'Channel_1'],
        'PricingStrategy': [2, 2, 4, 2, 4],
        'TransactionStartTime': [
            '2026-05-28 10:00:00', '2026-05-29 11:00:00',
            '2026-05-27 09:00:00', '2026-05-01 08:00:00', '2026-05-30 14:00:00'
        ]
    })

def test_pipeline_output_shape(sample_data):
    """Test 1: Check that the feature engineering pipeline transforms numerical and categorical columns."""
    pipeline = build_feature_pipeline()
    transformed_matrix = pipeline.fit_transform(sample_data)
    
    # Assert that rows match and columns are expanded via One-Hot encoding
    assert transformed_matrix.shape[0] == sample_data.shape[0]
    assert transformed_matrix.shape[1] > 0

def test_proxy_target_assignment(sample_data):
    """Test 2: Check that the RFM proxy generator appends the target column successfully."""
    generator = RFMProxyTargetGenerator(n_clusters=2)
    processed_df = generator.fit_transform(sample_data)
    
    assert 'is_high_risk' in processed_df.columns
    assert set(processed_df['is_high_risk'].unique()).issubset({0, 1})