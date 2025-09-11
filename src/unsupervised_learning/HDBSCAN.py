from dataclasses import dataclass
import torch
from typing import Optional, Callable, Tuple, List
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from KNN import KNN_Methods
import hdbscan
import time
from tqdm import tqdm


"""Note: need to fix confidence scores"""

def safe_lambda(w):
    if w == 0:
        return float('inf')
    elif math.isinf(w):
        return 0
    else:
        return 1 / w

def excess_of_mass(
    children: List[Tuple[int, int]],
    stabilities: np.ndarray,
    cluster_sizes: np.ndarray,
    min_cluster_size: int,
    n: int,
    verbose: bool = False,
    ) -> List[int]:
        """
        EOM selection on a binary merge tree.
        Leaves: 0..n-1, internals: n..2n-2, root: 2n-2.
        Returns a list of selected internal node IDs (antichain).
        """
        n_clusters = 2 * n - 1
        root = n_clusters - 1
        is_leaf = lambda u: u < n
        is_internal = lambda u: u >= n

        locally_selected = [False] * (n_clusters)
        EOM_values = [None] * n_clusters

        stack = [root]

        while stack:
            curr_cluster_id = stack.pop()
            if is_leaf(curr_cluster_id):
                EOM_values[curr_cluster_id] = 0.0
                continue
            
            left_child_id, right_child_id = children[curr_cluster_id - n]

            if verbose:
                print(f"Compting EOM for cluster {curr_cluster_id} with children {left_child_id}, {right_child_id}")
                print(f"stack after pop: {stack}")

            if EOM_values[left_child_id] is None or EOM_values[right_child_id] is None:
                stack.append(curr_cluster_id)
                if EOM_values[left_child_id] is None:
                    stack.append(left_child_id)
                if EOM_values[right_child_id] is None:
                    stack.append(right_child_id)
                continue
            
            
            child_sum = EOM_values[left_child_id] + EOM_values[right_child_id]
            eligible = (cluster_sizes[curr_cluster_id] >= min_cluster_size) and (curr_cluster_id != root)
            

            if eligible and stabilities[curr_cluster_id-n] >= child_sum:
                locally_selected[curr_cluster_id] = True
                EOM_values[curr_cluster_id] = float(stabilities[curr_cluster_id-n])
            else:
                locally_selected[curr_cluster_id] = False
                EOM_values[curr_cluster_id] = child_sum
            
        
           
          
        if verbose:
            print(f"Locally Selected clusters: {locally_selected}")

        stack = [root]
        selected_clusters = []

        while stack:
            curr_cluster_id = stack.pop()

            if locally_selected[curr_cluster_id]:
                if cluster_sizes[curr_cluster_id] >= min_cluster_size:
                    selected_clusters.append(curr_cluster_id)
            else:
                if is_internal(curr_cluster_id):
                    left_child_id, right_child_id = children[curr_cluster_id - n]
                    stack.append(left_child_id)
                    stack.append(right_child_id)

        if verbose:
            print(f"Selected {selected_clusters} clusters after pruning")
        
        return selected_clusters



@dataclass
class ClusterHierarchy:
    # Tree structure
    parents: np.ndarray            # (num_nodes,) parent pointer; -1 for root(s)
    cluster_sizes: np.ndarray      # (num_nodes,) size of each node's cluster
    children: List[Tuple[int,int]] # for internal nodes: list of (left_child_id, right_child_id) by creation order

    # Lifetimes / weights (HDBSCAN uses 1/lambda convention; store the weights you actually use)
    weight_appearance: np.ndarray  # (num_nodes,) weight at which node is created (merge edge)
    weight_consumed:  np.ndarray   # (num_nodes,) weight at which node is merged into parent; +inf for final root


@dataclass
class HDBSCANResult:
    labels: np.ndarray # Cluster labels for each point
    confidence: Optional[np.ndarray] = None # confidence scores for each point that its correctly clustered

    #Dendogram Metadata
    weight_appearance: Optional[np.ndarray] = None  # edge weight at which the cluster was formed (i.e., 1 / lambda_birth)
    weight_consumed: Optional[np.ndarray] = None    # edge weight at which the cluster was merged into a larger one (i.e., 1 / lambda_death)
    lambda_birth: Optional[np.ndarray] = None  # lambda value at which the cluster was formed
    lambda_points_in_cluster: Optional[np.ndarray] = None # (n-1,2) lambda values for points in each internal cluster


    mst_edges: Optional[np.ndarray] = None # edges in the minimum spanning tree


class HDBSCAN_Reimpl(KNN_Methods):
    def __init__(
        self,
        min_pts: int,
        min_cluster_size: int,
        metric: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        verbose: bool = False,
        chunk_size: int = 512,
        cluster_selection_method: Callable = excess_of_mass
    ):
        """
        Initialize an HDBSCAN clustering object.

        Args:
            min_pts (int): Number of neighbors used to compute core distances.
            min_cluster_size (int): Minimum number of points required for a cluster to be considered valid.
            metric (Callable): Optional custom distance function. Defaults to L2 distance.
            device (str): 'cpu' or 'cuda'. If None, defaults to 'cpu'.
            verbose (bool): If True, enables logging during fit.
            chunk_size (int): Number of points processed at a time when computing pairwise distances.
            use_faiss (bool): If True, uses FAISS for kNN search, otherwise computers kNN exactly.
        """
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.k = min_pts  # Alias for min_points in kNN search
        self.verbose = verbose
        self.chunk_size = chunk_size

        # Outputs after fitting
        self.labels_ = None             # Final cluster labels
        self.confidence_ = None         # Confidence scores in [0, 1] per point)
        self.mst_ = None                # Edges in the mutual reachability MST
        self.result_ = None             # Stores full result (HDBSCANResult object

    

    def fit(self, X: np.ndarray, log_var: Optional[np.ndarray] = None) -> 'HDBSCAN_Reimpl':
        """
        Run the HDBSCAN clustering algorithm on input data X.
        Stores results in self.labels_, self.confidence_, self.mst_ and self.result_.
        
        """
        n = X.shape[0]
        X = np.ascontiguousarray(X, dtype=np.float32)
        if log_var is not None:
            log_var = _as_numpy_f32(log_var)
            if log_var.shape != X.shape:
                raise ValueError(f"log_var shape {log_var.shape} must match X shape {X.shape}")

        # Step 1: Compute k-nearest neighbors
        if self.metric is None:
            knn_dist, knn_idx = self._knn_faiss(X, k=self.k)
        else:
            knn_dist, knn_idx = self._knn(X, log_var=log_var, k=self.k)


        # Step 2: Compute core distances (distance to kth nearest neighbor)
        core_distances = self._core_distances(knn_dist, min_pts=self.min_pts)

            # Step 3: Compute mutual reachability distances
        i_indicies, j_indicies, mreach = self._mutual_reachability(knn_dist, knn_idx, core_distances)

        # Step 4: Build MST from mutual reachability graph
        mst_i_s, mst_j_s, mst_edge_weights = self._mst(mreach, i_indicies, j_indicies, n)
        self.mst_ = (mst_i_s, mst_j_s, mst_edge_weights)

        # Step 5: Build hierarchy of cluster merges from MST
        clustering_hierarchy = self._build_hierarchy(mst_i_s, mst_j_s, mst_edge_weights, n)

        stabilities, lambda_birth_clusters, sum_lambda_ps = self._compute_stabilities_points(cluster_hierarchy, n)
        
        # Step 6: Traverse hierarchy to extract final clusters, stability, confidence
        labels, confidence = self._find_clustering(
            clustering_hierarchy.cluster_sizes,
            clustering_hierarchy.children,
            stabilities,
            n
        )


        # Step 8: Store all results in dataclass
        self.result_ = HDBSCANResult(
            labels=labels,
            #confidence=cluster_hierarchy.confidence,
            weight_appearance=cluster_hierarchy.weight_appearance, 
            weight_consumed=cluster_hierarchy.weight_consumed, 
            mst_edges=(mst_i_s, mst_j_s, mst_edge_weights)
        )
        return self

    
    def _core_distances(self, knn_dist: np.ndarray, min_pts: int) -> np.ndarray:
        """
        Compute core distances for each point in the dataset.

        """
        core_distances = knn_dist[:, min_pts - 1] # distance to the k-th nearest neighbor
        return core_distances
    
    def _mutual_reachability(
        self,
        knn_dist: np.ndarray,
        knn_idx: np.ndarray,
        core_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute the mutual reachability distances for all k-NN edges in the dataset. mreach(i,j) = max(core(i), core(j), dist(i,j)

        """
        if knn_dist.ndim != 2 or knn_idx.ndim != 2:
            raise ValueError("knn_dist and knn_idx must be 2D (n, k)")
        n, k = knn_idx.shape

        if knn_dist.shape != (n, k):
            raise ValueError(f"knn_dist shape {knn_dist.shape} != knn_idx shape {(n, k)}")
        if core_distances.shape != (n,):
            raise ValueError(f"core_distances must be shape ({n},), got {core_distances.shape}")

        # Create source indices for each vertex (k copies needed for each neighbour)
        i_indices = np.repeat(np.arange(n, dtype=np.int64), k)
        # Flatten the neighbor indices
        j_indices = knn_idx.ravel().astype(np.int64) 
        # Flatten the distances to get edge-wise distances
        wij = knn_dist.ravel().astype(np.float32) 

        # Get core distances for each endpoint of each edge
        core_i = core_distances[i_indices].astype(np.float32)
        core_j = core_distances[j_indices].astype(np.float32)

        # Mutual reachability distance = max(core_i, core_j, actual_distance)
        mreach = np.maximum.reduce([core_i, core_j, wij]) .astype(np.float32)
        return i_indices, j_indices, mreach

    def _mst(
        self,
        mreach: np.ndarray,
        i_indices: np.ndarray,
        j_indices: np.ndarray,
        n: int,
        symmetrize = None     # make graph undirected 
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Minimum Spanning Tree (MST) of the mutual reachability graph.


        symmetrize:
            - 'max' (safe for asymmetric distances) : W <- max(W, W^T)
            - 'min' (optimistic, can over-connect)  : W <- min(W, W^T)
            - 'avg'                                 : W <- (W + W^T)/2
             - None                                  : use directed weights as-is (MST then runs on that)

        """
        # Construct sparse mutual reachability graph in COO format
        mreach = np.asarray(mreach, dtype=np.float32)
        i_indices = np.asarray(i_indices, dtype=np.int64)
        j_indices = np.asarray(j_indices, dtype=np.int64)

        G = coo_matrix((mreach, (i_indices, j_indices)), shape=(n, n))


        # symmetrize if requested
        if symmetrize == "max":
            G = G.maximum(G.T)
        elif symmetrize == "min":
            G = G.minimum(G.T)
        elif symmetrize == "avg":
            # (G + G.T)/2 while preserving sparsity pattern
            # sum then scale; coalesce duplicates by converting to CSR/COO
            G = (G + G.T) * 0.5
        elif symmetrize is None:
            pass  # use directed weights as-is
        else:
            raise ValueError(f"symmetrize must be one of 'max','min','avg',None; got {symmetrize!r}")

        # Compute the MST using SciPy's implementation
        mst = minimum_spanning_tree(G).tocoo()

        # Extract edges and weights from the MST
        i_s = mst.row.astype(np.int64)
        j_s = mst.col.astype(np.int64)
        mst_edge_weights = mst.data.astype(np.float32)

        return i_s, j_s, mst_edge_weights

    def _build_hierarchy(
        self,
        i_s: np.ndarray,
        j_s: np.ndarray,
        mst_edge_weights: np.ndarray,
        n: int
    ) -> ClusterHierarchy:
        """
        Construct the dendrogram from the MST.

        This function processes the edges of a Minimum Spanning Tree (MST) in order of increasing weight 
        and builds a binary clustering tree (dendrogram), where each merge corresponds to an edge 
        in the MST. Each merge creates a new internal cluster node, assigning it a unique cluster ID.

        Args:
            i_s (np.ndarray): (num_edges,) source node indices of edges in the MST
            j_s (np.ndarray): (num_edges,) target node indices of edges in the MST
            mst_edge_weights (np.ndarray): (num_edges,) weights of the MST edges
            n (int): number of original data points (leaf nodes)
        """

        # Sort edges by ascending weight
        order = np.argsort(mst_edge_weights, kind="mergesort")
        u_sorted = i_s[order].astype(np.int64,   copy=False)
        v_sorted = j_s[order].astype(np.int64,   copy=False)
        w_sorted = mst_edge_weights[order].astype(np.float32, copy=False)
        if self.verbose:
            print(f"Building hierarchy from MST with {len(u_sorted)} edges...")

        # Initialize information for union-find structure and cluster data
        n_nodes = 2 * n - 1
        parents = np.arange(n_nodes, dtype=np.int64) # Parent pointer for each cluster node
        cluster_sizes = np.zeros(n_nodes, dtype=np.int64) # Size of each cluster
        cluster_sizes[:n] = 1  # Leaf nodes (original points) have size 1

        curr_cluster_id = n  # Next available ID for internal cluster nodes

        children = [(-1, -1) for _ in range(n-1)] # Children list for each internal node in the dendrogram

        weight_appearance = np.full(n_nodes, np.nan, dtype=np.float32) # Weight at which each cluster appears
        weight_appearance[:n] = 0.0  # Leaf nodes appear at weight 0
        weight_consumed = np.full(n_nodes, np.inf, dtype=np.float32) # Weight at which each cluster is merged into parent

        # Union-find with path compression
        def find_parent(u):
            while parents[u] != u:
                parents[u] = parents[parents[u]]
                u = parents[u]
            return u

        # Process each edge in non-decreasing order of weight
        print("Starting to build hierarchy...")
        for u, v, weight in zip(u_sorted, v_sorted, w_sorted):
            if self.verbose:
                print(f"Building {curr_cluster_id}th cluster from edge ({u}, {v}) with weight {weight:.4f}")
            # Find current cluster representatives for endpoints
            cluster_id_u = find_parent(u)
            cluster_id_v = find_parent(v)

             # Skip if already in the same cluster (edge would create a cycle)
            if cluster_id_u == cluster_id_v:
                continue
            
            # Merge clusters: assign a new parent ID
            parents[cluster_id_u] = curr_cluster_id
            parents[cluster_id_v] = curr_cluster_id
            parents[curr_cluster_id] = curr_cluster_id  # parent of new cluster is itself

            # Update new cluster size
            size = cluster_sizes[cluster_id_u] + cluster_sizes[cluster_id_v]
            cluster_sizes[curr_cluster_id] = size
    

            # Record weight of appearance and consumption for each cluster
            weight_appearance[curr_cluster_id] = weight
            weight_consumed[cluster_id_u] = weight
            weight_consumed[cluster_id_v] = weight


             # Register child clusters under current merged cluster
            children[curr_cluster_id-n] = (cluster_id_u, cluster_id_v)


             # Update ID for next internal node
            curr_cluster_id += 1
            if curr_cluster_id >= n_nodes:
                break

        # mark top node(s): convert self-parent to -1
        for node in range(n_nodes):
            if parents[node] == node:
                parents[node] = -1

        return ClusterHierarchy(
            parents=parents,
            cluster_sizes=cluster_sizes,
            children=children,
            weight_appearance=weight_appearance,
            weight_consumed=weight_consumed
            )


    def _compute_stabilities_points(self, hierarchy: ClusterHierarchy, n: int) -> np.ndarray:
        """
        Compute lambda values for each point in each cluster based on the hierarchy.
        """

        children = hierarchy.children
        cluster_sizes = hierarchy.cluster_sizes
        weight_appearance = hierarchy.weight_appearance
        weight_consumed = hierarchy.weight_consumed
        parents = hierarchy.parents

        sum_lambda_ps = np.full(n-1, np.nan, dtype=np.float32)
        lambda_birth_clusters = np.zeros(n-1, dtype=np.float32)
        stabilities = np.zeros(n-1, dtype=np.float32)
        n_clusters = 2 * n - 1

        root = 2*n-2
        is_leaf = lambda u: u < n

        

        stack = [root]

        if self.verbose:
            print("computing lambda_births")
        while stack:
            cluster_id = stack.pop()
            

            if is_leaf(cluster_id):
                continue
            
            left_child_id, right_child_id = children[cluster_id - n]

            left_has_min_size = cluster_sizes[left_child_id] >= self.min_cluster_size
            right_has_min_size = cluster_sizes[right_child_id] >= self.min_cluster_size

            if self.verbose:
                print(f"cluster {cluster_id} with children {left_child_id}, {right_child_id} and sizes {cluster_sizes[left_child_id]}, {cluster_sizes[right_child_id]}")

            if left_has_min_size and right_has_min_size:
                lambda_birth_clusters[left_child_id - n] = safe_lambda(weight_appearance[left_child_id])
                lambda_birth_clusters[right_child_id - n] = safe_lambda(weight_appearance[right_child_id])
            
            elif left_has_min_size:
                lambda_birth_clusters[left_child_id - n] = lambda_birth_clusters[cluster_id - n]
            elif right_has_min_size:
                lambda_birth_clusters[right_child_id - n] = lambda_birth_clusters[cluster_id - n]
            
            stack.append(left_child_id)
            stack.append(right_child_id)
        if self.verbose:
            print("lambda_births: ", lambda_birth_clusters)

        stack = [root]

        if self.verbose:
            print("computing stabilities")
        while stack:
            cluster_id = stack.pop()

            if is_leaf(cluster_id):
                continue
            
            left_child_id, right_child_id = children[cluster_id - n]

            left_has_min_size = cluster_sizes[left_child_id] >= self.min_cluster_size
            right_has_min_size = cluster_sizes[right_child_id] >= self.min_cluster_size

            left_computed = not left_has_min_size or not math.isnan(sum_lambda_ps[left_child_id - n]) 
            right_computed = not right_has_min_size or not math.isnan(sum_lambda_ps[right_child_id - n]) 

            if self.verbose:
                print(f"cluster {cluster_id} with children {left_child_id}, {right_child_id} and sizes {cluster_sizes[left_child_id]}, {cluster_sizes[right_child_id]}")


            if not left_computed or not right_computed:
                stack.append(cluster_id)
                if not left_computed:
                    stack.append(left_child_id)
                if not right_computed:
                    stack.append(right_child_id)
                continue

            

            if (left_has_min_size and right_has_min_size) or (not left_has_min_size and not right_has_min_size):
                sum_lambda_ps[cluster_id - n] = safe_lambda(weight_appearance[cluster_id])*(cluster_sizes[left_child_id] + cluster_sizes[right_child_id])
            elif left_has_min_size:
                sum_lambda_ps[cluster_id - n] = sum_lambda_ps[left_child_id-n]  + \
                                                 safe_lambda(weight_appearance[cluster_id]) * cluster_sizes[right_child_id]
            else:
                sum_lambda_ps[cluster_id - n] = sum_lambda_ps[right_child_id-n]  + \
                                                 safe_lambda(weight_appearance[cluster_id]) * cluster_sizes[left_child_id]
            if cluster_id != root:
                stability = sum_lambda_ps[cluster_id - n] - \
                            cluster_sizes[cluster_id] * lambda_birth_clusters[cluster_id - n]
                stabilities[cluster_id - n] = stability
            
        return stabilities, lambda_birth_clusters, sum_lambda_ps



       

        
        

    def _find_clustering(self, cluster_sizes: list,  children: list, stabilities: np.ndarray, n: int) -> HDBSCANResult:

        """
        Selects clusters from the hierarchy by maximizing stability, following the HDBSCAN approach.

        Node indexing convention:
            - Leaves (points): 0 .. n-1
            - Internal clusters: n .. 2n-2
            z- Root: 2n-2

        """


        root = 2 * n - 2
        is_leaf = lambda u: u < n
        is_internal = lambda u: u >= n



        # Step 1: Extract final clustering based on stability criterion


        #EOM to find locally selected clusters
        clusters = excess_of_mass(children, stabilities, cluster_sizes, self.min_cluster_size, n, verbose=self.verbose)
       
        
        # Step 2: Assign labels and compute confidence for points in selected clusters

        # # Initialize labels and confidence scores

        labels = np.full(n, -1, dtype=np.int64)  # -1 means noise
        confidence = [0.0] * n
        label = 0

        if self.verbose:
            print(f"Assigning labels for {len(clusters)} selected clusters...")

         # For each selected cluster, assign its label to all its leaf points
        for cluster_id in clusters:
            cluster_stack = [cluster_id]
            while cluster_stack:
                curr_cluster_id = cluster_stack.pop()

                # Contains more than one point so we recurse
                if is_internal(curr_cluster_id):
                    left_child_id, right_child_id = children[curr_cluster_id - n]
                    cluster_stack.append(left_child_id)
                    cluster_stack.append(right_child_id)
                # Is a leaf node containing a single point so we assign label and confidence
                else:
                    labels[curr_cluster_id] = label
                    
                    # Compute confidence: lambda_point / lambda_cluster
                    # lp = safe_lambda(weight_consumed[curr_cluster_id])
                    # lc = safe_lambda(weight_appearance[cluster_id])
                    # if lc > 0:
                    #     confidence[curr_cluster_id] = min(lp / lc, 1.0) if lc > 0 else 1.0

            # move to next cluster    
            label += 1

                

        
    
        return labels, confidence



        
                    


