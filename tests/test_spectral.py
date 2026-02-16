"""
Test spectral feature extraction.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append('../')

from fraudboost import SpectralFeatureExtractor


class TestSpectralFeatureExtractor:
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for graph construction."""
        # Create synthetic transaction data
        np.random.seed(42)
        
        # 100 transactions with customer, merchant, device relationships
        n_transactions = 100
        n_customers = 30
        n_merchants = 20  
        n_devices = 25
        
        data = {
            'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
            'merchant_id': np.random.randint(1, n_merchants + 1, n_transactions),
            'device_id': np.random.randint(1, n_devices + 1, n_transactions),
            'amount': np.random.lognormal(4, 1, n_transactions),
            'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
        }
        
        return pd.DataFrame(data)
    
    def test_basic_fit_transform(self, sample_transaction_data):
        """Test basic spectral feature extraction."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id', 'device_id']
        
        extractor = SpectralFeatureExtractor(n_components=5, random_state=42)
        features = extractor.fit_transform(df, entity_columns, weight_column='amount')
        
        # Should return features for each transaction
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0  # Should have some features
        
        # Features should be numeric
        assert np.all(np.isfinite(features))
        
        # Should have node mappings
        assert len(extractor.node_mappings_) > 0
        
        # Should have spectral decomposition results
        assert extractor.eigenvalues_ is not None
        assert extractor.eigenvectors_ is not None
        assert extractor.node_features_ is not None
    
    def test_transform_new_data(self, sample_transaction_data):
        """Test transforming new data with fitted extractor."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id', 'device_id']
        
        # Split data for train/test
        train_df = df.iloc[:70]
        test_df = df.iloc[70:]
        
        extractor = SpectralFeatureExtractor(n_components=5, random_state=42)
        train_features = extractor.fit_transform(train_df, entity_columns)
        test_features = extractor.transform(test_df, entity_columns)
        
        # Test features should have same number of columns
        assert train_features.shape[1] == test_features.shape[1]
        assert test_features.shape[0] == len(test_df)
        
        # All features should be finite
        assert np.all(np.isfinite(train_features))
        assert np.all(np.isfinite(test_features))
    
    def test_without_weights(self, sample_transaction_data):
        """Test spectral extraction without edge weights."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id']  # Only 2 entities
        
        extractor = SpectralFeatureExtractor(n_components=3, random_state=42)
        features = extractor.fit_transform(df, entity_columns)  # No weight_column
        
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0
        assert np.all(np.isfinite(features))
    
    def test_feature_names(self, sample_transaction_data):
        """Test feature name generation."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id']
        
        extractor = SpectralFeatureExtractor(n_components=3, random_state=42)
        features = extractor.fit_transform(df, entity_columns)
        
        feature_names = extractor.get_feature_names()
        
        # Should have feature names
        assert len(feature_names) > 0
        # Each name should be a string
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_spectral_summary(self, sample_transaction_data):
        """Test spectral decomposition summary."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id', 'device_id']
        
        extractor = SpectralFeatureExtractor(n_components=5, random_state=42)
        features = extractor.fit_transform(df, entity_columns)
        
        summary = extractor.get_spectral_summary()
        
        # Should have summary statistics
        required_keys = [
            'n_nodes', 'n_components', 'eigenvalue_sum',
            'largest_eigenvalue', 'smallest_eigenvalue'
        ]
        for key in required_keys:
            assert key in summary
        
        # Values should make sense
        assert summary['n_nodes'] > 0
        assert summary['n_components'] == 5
        assert summary['eigenvalue_sum'] >= 0
    
    def test_min_degree_filtering(self, sample_transaction_data):
        """Test minimum degree filtering."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id']
        
        # Use high min_degree to filter out low-degree nodes
        extractor = SpectralFeatureExtractor(
            n_components=3, 
            min_degree=5,  # Require at least 5 connections
            random_state=42
        )
        features = extractor.fit_transform(df, entity_columns)
        
        assert features.shape[0] == len(df)
        # Should still produce features even with filtering
        assert features.shape[1] > 0
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        # Create minimal dataset
        df = pd.DataFrame({
            'customer': [1, 1, 2, 2],
            'merchant': ['A', 'B', 'A', 'C'],
            'amount': [100, 200, 300, 400]
        })
        
        extractor = SpectralFeatureExtractor(n_components=2, random_state=42)
        features = extractor.fit_transform(df, ['customer', 'merchant'])
        
        # Should handle small dataset gracefully
        assert features.shape[0] == 4
        assert features.shape[1] > 0
        assert np.all(np.isfinite(features))
    
    def test_missing_values(self):
        """Test handling of missing values in entity columns."""
        df = pd.DataFrame({
            'customer': [1, 2, np.nan, 3],
            'merchant': ['A', 'B', 'C', np.nan],
            'amount': [100, 200, 300, 400]
        })
        
        extractor = SpectralFeatureExtractor(n_components=2, random_state=42)
        features = extractor.fit_transform(df, ['customer', 'merchant'])
        
        # Should handle missing values without crashing
        assert features.shape[0] == 4
        assert features.shape[1] > 0
        # Features for rows with missing entities might be zeros
        assert np.all(np.isfinite(features))
    
    def test_single_entity_column(self):
        """Test with only one entity column."""
        df = pd.DataFrame({
            'customer': [1, 1, 2, 2, 3, 3],
            'amount': [100, 200, 300, 400, 500, 600]
        })
        
        extractor = SpectralFeatureExtractor(n_components=2, random_state=42)
        features = extractor.fit_transform(df, ['customer'])
        
        # Should work even with single entity type
        assert features.shape[0] == 6
        assert features.shape[1] > 0
    
    def test_component_parameter(self, sample_transaction_data):
        """Test different numbers of spectral components."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id']
        
        # Test with different component counts
        for n_comp in [2, 5, 10]:
            extractor = SpectralFeatureExtractor(n_components=n_comp, random_state=42)
            features = extractor.fit_transform(df, entity_columns)
            
            assert features.shape[0] == len(df)
            assert features.shape[1] > 0
            
            # Number of eigenvalues should match requested components (or less if graph is small)
            assert len(extractor.eigenvalues_) <= n_comp
    
    def test_deterministic_results(self, sample_transaction_data):
        """Test that results are deterministic with fixed random seed."""
        df = sample_transaction_data
        entity_columns = ['customer_id', 'merchant_id']
        
        # Run twice with same seed
        extractor1 = SpectralFeatureExtractor(n_components=3, random_state=42)
        features1 = extractor1.fit_transform(df, entity_columns)
        
        extractor2 = SpectralFeatureExtractor(n_components=3, random_state=42)
        features2 = extractor2.fit_transform(df, entity_columns)
        
        # Results should be identical (up to numerical precision)
        np.testing.assert_array_almost_equal(features1, features2, decimal=10)
    
    def test_error_conditions(self):
        """Test error handling."""
        extractor = SpectralFeatureExtractor()
        
        # Test transform before fit
        df = pd.DataFrame({'customer': [1, 2], 'merchant': ['A', 'B']})
        
        with pytest.raises(ValueError):
            extractor.transform(df, ['customer', 'merchant'])
        
        # Test empty dataframe
        empty_df = pd.DataFrame()
        # Should handle gracefully or raise appropriate error
        try:
            features = extractor.fit_transform(empty_df, [])
            assert features.shape[0] == 0
        except (ValueError, IndexError):
            pass  # Acceptable to raise error for empty data