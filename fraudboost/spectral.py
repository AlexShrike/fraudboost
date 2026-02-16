"""
Spectral feature extraction for fraud detection.

Creates graph-based features by:
1. Building bipartite graphs from transaction data (user-merchant, user-device, etc.)
2. Computing graph Laplacian and extracting eigenvalues/eigenvectors
3. Deriving node features: spectral energy, centrality, heterophily, PageRank

These features capture network patterns that traditional features miss,
like fraud rings, velocity patterns, and behavioral clusters.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings


class SpectralFeatureExtractor:
    """
    Extract spectral graph features from transaction data.
    
    Creates bipartite graphs connecting entities (customers, merchants, devices, etc.)
    and derives features from the graph's spectral properties.
    """
    
    def __init__(self,
                 n_components: int = 10,
                 random_state: Optional[int] = None,
                 min_degree: int = 2,
                 max_nodes: int = 50000):
        """
        Args:
            n_components: Number of eigenvectors to compute
            random_state: Random seed for reproducible results
            min_degree: Minimum node degree to include in graph
            max_nodes: Maximum nodes to prevent memory issues
        """
        self.n_components = n_components
        self.random_state = random_state
        self.min_degree = min_degree
        self.max_nodes = max_nodes
        
        # Fitted attributes
        self.node_mappings_ = {}
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.node_features_ = None
        self.feature_names_ = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_transform(self, df: pd.DataFrame, 
                     entity_columns: List[str],
                     weight_column: Optional[str] = None) -> np.ndarray:
        """
        Extract spectral features from transaction data.
        
        Args:
            df: Transaction dataframe  
            entity_columns: Column names defining graph relationships
                          (e.g., ['customer_id', 'merchant_id', 'device_id'])
            weight_column: Column name for edge weights (e.g., 'amount')
            
        Returns:
            Feature matrix (n_transactions, n_features)
        """
        # Create bipartite graph
        adjacency_matrix, node_mapping = self._build_graph(
            df, entity_columns, weight_column
        )
        
        # Compute spectral decomposition
        eigenvalues, eigenvectors = self._compute_spectral_decomposition(adjacency_matrix)
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.node_mappings_ = node_mapping
        
        # Extract node-level features
        self.node_features_ = self._compute_node_features(
            adjacency_matrix, eigenvectors
        )
        
        # Map transaction-level features
        transaction_features = self._map_transaction_features(
            df, entity_columns
        )
        
        return transaction_features
    
    def transform(self, df: pd.DataFrame, entity_columns: List[str]) -> np.ndarray:
        """Transform new data using fitted spectral features."""
        if self.node_features_ is None:
            raise ValueError("Must call fit_transform first")
        
        return self._map_transaction_features(df, entity_columns)
    
    def _build_graph(self, df: pd.DataFrame, entity_columns: List[str],
                    weight_column: Optional[str] = None) -> Tuple[sparse.csr_matrix, Dict]:
        """Build bipartite graph from transaction data."""
        
        # Create entity type mapping (customer, merchant, device, etc.)
        all_entities = []
        entity_types = []
        
        for col_idx, col in enumerate(entity_columns):
            entities = df[col].dropna().unique()
            # Add type prefix to avoid collisions across columns
            prefixed_entities = [f"{col}:{ent}" for ent in entities]
            all_entities.extend(prefixed_entities)
            entity_types.extend([col_idx] * len(prefixed_entities))
        
        # Limit number of nodes
        if len(all_entities) > self.max_nodes:
            warnings.warn(f"Too many entities ({len(all_entities)}), sampling {self.max_nodes}")
            indices = np.random.choice(len(all_entities), self.max_nodes, replace=False)
            all_entities = [all_entities[i] for i in indices]
            entity_types = [entity_types[i] for i in indices]
        
        # Create node mapping
        node_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
        
        # Build adjacency matrix
        n_nodes = len(all_entities)
        row_indices = []
        col_indices = []
        weights = []
        
        for _, transaction in df.iterrows():
            # Get entities involved in this transaction
            transaction_entities = []
            for col in entity_columns:
                if pd.notna(transaction[col]):
                    entity_key = f"{col}:{transaction[col]}"
                    if entity_key in node_to_idx:
                        transaction_entities.append(node_to_idx[entity_key])
            
            # Connect all entity pairs in this transaction
            weight = 1.0
            if weight_column and pd.notna(transaction[weight_column]):
                weight = float(transaction[weight_column])
            
            for i in range(len(transaction_entities)):
                for j in range(i + 1, len(transaction_entities)):
                    idx_i, idx_j = transaction_entities[i], transaction_entities[j]
                    row_indices.extend([idx_i, idx_j])
                    col_indices.extend([idx_j, idx_i])
                    weights.extend([weight, weight])
        
        # Create sparse adjacency matrix
        adjacency_matrix = sparse.csr_matrix(
            (weights, (row_indices, col_indices)), 
            shape=(n_nodes, n_nodes)
        )
        
        # Remove low-degree nodes
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        high_degree_mask = degrees >= self.min_degree
        
        if np.sum(high_degree_mask) < len(all_entities):
            filtered_entities = [all_entities[i] for i in range(len(all_entities)) 
                               if high_degree_mask[i]]
            node_to_idx = {entity: idx for idx, entity in enumerate(filtered_entities)}
            adjacency_matrix = adjacency_matrix[high_degree_mask][:, high_degree_mask]
        
        return adjacency_matrix, node_to_idx
    
    def _compute_spectral_decomposition(self, adjacency_matrix: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of the graph Laplacian."""
        n_nodes = adjacency_matrix.shape[0]
        
        # Compute degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        
        # Avoid division by zero
        degrees = np.maximum(degrees, 1e-12)
        
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        deg_sqrt_inv = sparse.diags(1.0 / np.sqrt(degrees))
        normalized_adjacency = deg_sqrt_inv @ adjacency_matrix @ deg_sqrt_inv
        
        # Laplacian
        identity = sparse.identity(n_nodes)
        laplacian = identity - normalized_adjacency
        
        # Compute eigendecomposition
        n_eigs = min(self.n_components, n_nodes - 1)
        
        try:
            if laplacian.shape[0] < 100:  # Small matrix, use dense computation
                laplacian_dense = laplacian.toarray()
                eigenvalues, eigenvectors = eigh(laplacian_dense)
                # Sort by eigenvalues
                sort_idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[sort_idx[:n_eigs]]
                eigenvectors = eigenvectors[:, sort_idx[:n_eigs]]
            else:  # Large matrix, use sparse computation
                eigenvalues, eigenvectors = eigsh(
                    laplacian, k=n_eigs, which='SM', tol=1e-6
                )
        except Exception as e:
            warnings.warn(f"Eigendecomposition failed: {e}, using random features")
            eigenvalues = np.random.exponential(1.0, n_eigs)
            eigenvectors = np.random.normal(0, 1, (n_nodes, n_eigs))
        
        return eigenvalues, eigenvectors
    
    def _compute_node_features(self, adjacency_matrix: sparse.csr_matrix, 
                              eigenvectors: np.ndarray) -> np.ndarray:
        """Compute node-level features from spectral decomposition."""
        n_nodes = adjacency_matrix.shape[0]
        features = []
        feature_names = []
        
        # 1. Spectral energy in frequency bands
        # Divide eigenvalue spectrum into 5 bands
        n_bands = 5
        n_eigs = eigenvectors.shape[1]
        band_size = max(1, n_eigs // n_bands)
        
        for band in range(n_bands):
            start_idx = band * band_size
            end_idx = min((band + 1) * band_size, n_eigs)
            
            if start_idx < n_eigs:
                band_energy = np.sum(eigenvectors[:, start_idx:end_idx] ** 2, axis=1)
                features.append(band_energy)
                feature_names.append(f'spectral_energy_band_{band}')
        
        # 2. Degree centrality
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degree_centrality = degrees / np.maximum(degrees.max(), 1)
        features.append(degree_centrality)
        feature_names.append('degree_centrality')
        
        # 3. Local heterophily score
        # Measure how different a node is from its neighbors
        heterophily = self._compute_heterophily(adjacency_matrix, eigenvectors)
        features.append(heterophily)
        feature_names.append('heterophily_score')
        
        # 4. PageRank approximation using power iteration
        pagerank = self._compute_pagerank(adjacency_matrix)
        features.append(pagerank)
        feature_names.append('pagerank')
        
        # 5. Clustering coefficient
        clustering = self._compute_clustering_coefficient(adjacency_matrix)
        features.append(clustering)
        feature_names.append('clustering_coefficient')
        
        self.feature_names_ = feature_names
        return np.column_stack(features)
    
    def _compute_heterophily(self, adjacency_matrix: sparse.csr_matrix, 
                           eigenvectors: np.ndarray) -> np.ndarray:
        """Compute local heterophily score for each node."""
        n_nodes = adjacency_matrix.shape[0]
        heterophily = np.zeros(n_nodes)
        
        # Use first few eigenvectors as node embeddings
        embeddings = eigenvectors[:, :min(5, eigenvectors.shape[1])]
        
        for i in range(n_nodes):
            neighbors = adjacency_matrix[i].nonzero()[1]
            
            if len(neighbors) > 0:
                # Compute distances to neighbors in embedding space
                node_embedding = embeddings[i]
                neighbor_embeddings = embeddings[neighbors]
                
                distances = np.linalg.norm(
                    neighbor_embeddings - node_embedding, axis=1
                )
                heterophily[i] = np.mean(distances)
        
        return heterophily
    
    def _compute_pagerank(self, adjacency_matrix: sparse.csr_matrix, 
                         damping: float = 0.85, max_iter: int = 50) -> np.ndarray:
        """Compute PageRank scores using power iteration."""
        n_nodes = adjacency_matrix.shape[0]
        
        # Normalize adjacency matrix (column-stochastic)
        degrees = np.array(adjacency_matrix.sum(axis=0)).flatten()
        degrees = np.maximum(degrees, 1e-12)  # Avoid division by zero
        
        col_sums = sparse.diags(1.0 / degrees)
        transition_matrix = adjacency_matrix @ col_sums
        
        # Initialize PageRank vector
        pagerank = np.ones(n_nodes) / n_nodes
        
        # Power iteration
        for _ in range(max_iter):
            new_pagerank = (
                damping * transition_matrix @ pagerank + 
                (1 - damping) / n_nodes
            )
            
            # Check convergence
            if np.linalg.norm(new_pagerank - pagerank) < 1e-6:
                break
                
            pagerank = new_pagerank
        
        return pagerank
    
    def _compute_clustering_coefficient(self, adjacency_matrix: sparse.csr_matrix) -> np.ndarray:
        """Compute local clustering coefficient for each node."""
        n_nodes = adjacency_matrix.shape[0]
        clustering = np.zeros(n_nodes)
        
        for i in range(n_nodes):
            neighbors = set(adjacency_matrix[i].nonzero()[1])
            degree = len(neighbors)
            
            if degree < 2:
                clustering[i] = 0.0
                continue
            
            # Count triangles
            triangles = 0
            for j in neighbors:
                j_neighbors = set(adjacency_matrix[j].nonzero()[1])
                triangles += len(neighbors & j_neighbors)
            
            # Clustering coefficient
            max_triangles = degree * (degree - 1)
            clustering[i] = triangles / max_triangles if max_triangles > 0 else 0.0
        
        return clustering
    
    def _map_transaction_features(self, df: pd.DataFrame, 
                                 entity_columns: List[str]) -> np.ndarray:
        """Map node features back to transaction level."""
        n_transactions = len(df)
        n_node_features = self.node_features_.shape[1]
        
        # Initialize transaction features
        transaction_features = np.zeros((n_transactions, n_node_features * len(entity_columns)))
        feature_start = 0
        
        for col_idx, col in enumerate(entity_columns):
            for i, (_, transaction) in enumerate(df.iterrows()):
                entity_value = transaction[col]
                
                if pd.notna(entity_value):
                    entity_key = f"{col}:{entity_value}"
                    
                    if entity_key in self.node_mappings_:
                        node_idx = self.node_mappings_[entity_key]
                        node_feats = self.node_features_[node_idx]
                        
                        start_idx = feature_start
                        end_idx = start_idx + n_node_features
                        transaction_features[i, start_idx:end_idx] = node_feats
            
            feature_start += n_node_features
        
        return transaction_features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for the extracted features."""
        if not self.feature_names_:
            return []
        
        all_names = []
        for entity_col in ['entity_0', 'entity_1', 'entity_2']:  # Generic names
            for base_name in self.feature_names_:
                all_names.append(f"{entity_col}_{base_name}")
        
        return all_names[:self.node_features_.shape[1] if self.node_features_ is not None else 0]
    
    def get_spectral_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the spectral decomposition."""
        if self.eigenvalues_ is None:
            return {}
        
        return {
            'n_nodes': len(self.node_mappings_),
            'n_components': len(self.eigenvalues_),
            'eigenvalue_sum': np.sum(self.eigenvalues_),
            'eigenvalue_variance': np.var(self.eigenvalues_),
            'spectral_gap': self.eigenvalues_[1] - self.eigenvalues_[0] if len(self.eigenvalues_) > 1 else 0,
            'largest_eigenvalue': np.max(self.eigenvalues_),
            'smallest_eigenvalue': np.min(self.eigenvalues_)
        }