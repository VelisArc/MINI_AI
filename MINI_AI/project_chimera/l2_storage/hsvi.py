# project_chimera/l2_storage/hsvi.py
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
    print("[L2 HSVI] Faiss library found. Hyper-Scale Vector Infrastructure is running on a real index.")
except ImportError:
    FAISS_AVAILABLE = False
    print("[L2 HSVI] WARNING: Faiss not installed. HSVI will run in simulation mode (slow).")
    print("[L2 HSVI] Install with: pip install faiss-cpu")

class HNSWIndex:
    """
    A real implementation of the Hyper-Scale Vector Infrastructure (HSVI) using Faiss's HNSW.
    This replaces the slow, simulated version.
    """
    def __init__(self, dim, m=32, ef_construction=40, ef_search=16):
        print(f"[L2 HSVI] Initializing HNSW index for dimension {dim}.")
        self.dim = dim
        self.is_trained = False

        if FAISS_AVAILABLE:
            # The "Flat" in HNSWFlat means vectors are stored as is (no compression).
            # This is the most accurate but uses the most memory.
            self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            # Fallback simulation if faiss is not available
            self.vectors = np.array([], dtype=np.float32).reshape(0, dim)

    def add_vectors(self, data: np.ndarray):
        if not data.ndim == 2 or data.shape[1] != self.dim:
            raise ValueError(f"Input data must be 2D with shape (n_vectors, {self.dim})")
        
        data = data.astype(np.float32)

        if FAISS_AVAILABLE:
            print(f"[L2 HSVI] Adding {data.shape[0]} vectors to the Faiss HNSW index.")
            self.index.add(data)
            self.is_trained = True
        else:
            print(f"[L2 HSVI] (Simulated) Adding {data.shape[0]} vectors.")
            self.vectors = np.vstack([self.vectors, data])

    def search(self, query_vector: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        """
        Searches for the k-nearest neighbors for a query vector.
        Returns: (distances, indices)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dim:
            raise ValueError(f"Query vector must have dimension {self.dim}")
            
        query_vector = query_vector.astype(np.float32)

        if FAISS_AVAILABLE:
            if self.index.ntotal == 0:
                return np.array([]), np.array([])
            print(f"[L2 HSVI->L0] Offloading HNSW graph traversal to optimized Faiss kernels...")
            return self.index.search(query_vector, k)
        else:
            # Fallback simulation
            if self.vectors.shape[0] == 0: return np.array([]), np.array([])
            print(f"[L2 HSVI] (Simulated) Performing slow brute-force search.")
            distances = np.linalg.norm(self.vectors - query_vector, axis=1)
            k = min(k, self.vectors.shape[0])
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            return nearest_distances.reshape(1, -1), nearest_indices.reshape(1, -1)
