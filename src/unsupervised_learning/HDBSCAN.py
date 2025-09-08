from dataclasses import dataclass
import torch
from typing import Optional, Callable, Tuple
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import math


@dataclass
class HDBSCANResult:
    labels: torch.Tensor # Cluster labels for each point
    confidence: Optional[torch.Tensor] = None # confidence scores for each point that its correctly clustered
    weight_appearance: Optional[torch.Tensor] = None  # edge weight at which the cluster was formed (i.e., 1 / lambda_birth)
    weight_consumed: Optional[torch.Tensor] = None    # edge weight at which the cluster was merged into a larger one (i.e., 1 / lambda_death)

    mst_edges: Optional[torch.Tensor] = None # edges in the minimum spanning tree

class HDBSCAN_Reimpl:
    def __init__(
        self,
        min_pts: int,
        min_cluster_size: int,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        chunk_size: int = 512
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
        self.device = device if device else 'cpu'
        self.verbose = verbose
        self.chunk_size = chunk_size

        # Outputs after fitting
        self.labels_ = None             # Final cluster labels
        self.confidence_ = None         # Confidence scores in [0, 1] per point
        self.result_ = None             # Stores full result (HDBSCANResult object)
        self.mst_ = None                # Edges in the mutual reachability MST

    def fit(self, X: torch.Tensor) -> 'HDBSCAN_Reimpl':
        """
        Run the HDBSCAN clustering algorithm on input data X.
        Stores results in self.labels_, self.confidence_, and self.result_.

        Args:
            X (torch.Tensor): Data matrix of shape (n_samples, n_features)

        Returns:
            self (HDBSCAN_Reimpl): Fitted clustering object
        """
        X = X.to(self.device)
        n = X.shape[0]

        # Step 1: Compute k-nearest neighbors
        if self.metric is None:
            knn_dist, knn_idx = self._knn_faiss(X, k=self.k)
        else:
            knn_dist, knn_idx = self._knn(X, k=self.k)


        # Step 2: Compute core distances (distance to kth nearest neighbor)
        core_distances = self._core_distances(knn_dist, min_pts=self.min_pts)

            # Step 3: Compute mutual reachability distances
        i_indicies, j_indicies, mreach = self._mutual_reachability(knn_dist, knn_idx, core_distances)

        # Step 4: Build MST from mutual reachability graph
        mst_i_s, mst_j_s, mst_edge_weights = self._mst(mreach, i_indicies, j_indicies, n)
        self.mst_ = (mst_i_s, mst_j_s, mst_edge_weights)

        # Step 5: Build hierarchy of cluster merges from MST
        cluster_sizes, weight_appearance, weight_consumed, children = self._build_hierarchy(mst_i_s, mst_j_s, mst_edge_weights, n)
        
        # Step 6: Traverse hierarchy to extract final clusters, stability, confidence
        labels, confidence, stability = self._find_clustering(
            cluster_sizes, weight_appearance, weight_consumed, children, n
        )

        self.labels_ = labels
        self.confidence_ = confidence

        # Step 8: Store all results in dataclass
        self.result_ = HDBSCANResult(
            labels=labels,
            confidence=confidence,
            weight_appearance=weight_appearance, 
            weight_consumed=weight_consumed, 
            mst_edges=torch.stack([mst_i_s, mst_j_s, mst_edge_weights], dim=1)
)

        return self

            

    def _pairwise_distances(
        self, X: torch.Tensor,  Y: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise distances between two datasets X (n samples) and Y (m samples).

        Args:
            X (torch.Tensor): Dataset of shape (n, d)
            Y (torch.Tensor): Dataset of shape (m, d)

        Returns:
            torch.Tensor: A (m, n) distance matrix, where entry (i, j) is the distance from Y[i] to X[j].
                        Uses a custom metric if provided, otherwise defaults to torch.cdist (L2).
        """

        if self.metric is not None:
            return self.metric(X, Y)
        else:
            return torch.cdist(X, Y)
        
    def _knn(self, X: torch.Tensor, k:int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Compute k-nearest neighbors for each point in X using batched distance computation.

        Args:
            X (torch.Tensor): Input data of shape (n, d)
            k (int): Number of neighbors to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - knn_dist: (n, k) distances to k nearest neighbors
                - knn_idx:  (n, k) indices of those neighbors in X
        """

        n = X.shape[0]
        knn_dist = []
        knn_idx = []


        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            X_chunk = X[start:end] # shape (chunk_size, d)
            distances = self._pairwise_distances(X_chunk, X) # shape (chunk_size, n)
            knn_dist_chunk, knn_idx_chunk = torch.topk(distances, k=k, largest=False)
            knn_dist.append(knn_dist_chunk)
            knn_idx.append(knn_idx_chunk)
        return torch.cat(knn_dist, dim=0), torch.cat(knn_idx, dim=0) # shape (n, d)

    def _knn_faiss(self, X: torch.Tensor, k:int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = X.detach().cpu().numpy().astype('float32')
        n, d = X_np.shape

        # Select index type (GPU or CPU)
        if self.device.startswith('cuda'):
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)  # Exact L2
        else:
            index = faiss.IndexFlatL2(d)

        index.add(X_np)  # Add all points
        dist, idx = index.search(X_np, k)  # Search k nearest neighbors

        # Convert back to torch
        knn_dist = torch.from_numpy(dist).to(self.device)
        knn_idx = torch.from_numpy(idx).to(self.device)

        return knn_dist, knn_idx
    
    def _core_distances(self, knn_dist: torch.Tensor, min_pts: int) -> torch.Tensor:
        """
        Compute core distances for each point in the dataset.

        The core distance of a point is defined as the distance to its min_pts-th nearest neighbor.
        This is used later to compute mutual reachability distances.

        Args:
            knn_dist (torch.Tensor): A (n, k) tensor containing distances from each point to its k nearest neighbors.
            min_pts (int): The number of neighbors to consider when computing core distances.

        Returns:
            torch.Tensor: A 1D tensor of shape (n,) where each value is the core distance of a point.
        """
        core_distances = knn_dist[:, min_pts - 1] # distance to the k-th nearest neighbor
        return core_distances
    
    def _mutual_reachability(self, knn_dist: torch.Tensor, knn_idx: torch.Tensor, core_distances: torch.Tensor) -> torch.Tensor:
        """
        Compute the mutual reachability distances for all k-NN edges in the dataset.

        The mutual reachability distance between points i and j is defined as:
            max(core_dist(i), core_dist(j), dist(i, j))
        This forms the edge weights for the mutual reachability graph, which is later used
        to compute the MST for the hierarcical clustering.

        Args:
            knn_dist (torch.Tensor): (n, k) distances from each point to its k nearest neighbors
            knn_idx (torch.Tensor): (n, k) indices of the k nearest neighbors for each point
            core_distances (torch.Tensor): (n,) core distances for each point

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - i_indices: source point indices for each edge
                - j_indices: target point indices for each edge
                - mreach: mutual reachability distances for all edges
        """
        n, k = knn_idx.shape

        # Create source indices for each vertex (k copies needed for each neighbour)
        i_indicies = torch.arange(m, devices=self.device).repeat_interleave(k)
        # Flatten the neighbor indices
        j_indicies = knn_idx.view(-1)
        # Flatten the distances to get edge-wise distances
        wij = knn_dist.reshape(-1)

        # Get core distances for each endpoint of each edge
        core_i = core_distances[i_indicies]
        core_j = core_distances[j_indicies]

        # Mutual reachability distance = max(core_i, core_j, actual_distance)
        mreach = torch.max(torch.stack([core_i, core_j, wij], dim=1), dim=1).values
        return i_indicies, j_indicies, mreach

    def _mst(self, mreach: torch.Tensor, i_indicies: torch.Tensor, j_indicies: torch.tensor, n: int) -> torch.Tensor:
        """
        Compute the Minimum Spanning Tree (MST) of the mutual reachability graph.

        Args:
            mreach (torch.Tensor): (num_edges,) mutual reachability distances for all edges
            i_indicies (torch.Tensor): source node indices for each edge
            j_indicies (torch.Tensor): target node indices for each edge
            n (int): number of nodes (points) in the dataset

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - i_s: source indices of edges in MST
                - j_s: target indices of edges in MST
                - mst_edge_weights: weights of edges in MST
        """
        # Construct sparse mutual reachability graph in COO format
        coo = coo_matrix((mreach.cpu().numpy(), (i_indicies.cpu().numpy(), j_indicies.cpu().numpy())), shape=(n, n))

        # Compute the MST using SciPy's implementation
        mst = minimum_spanning_tree(coo).tocoo()

        # Convert the MST edges and weights to torch tensors on the correct device
        i_s = torch.from_numpy(mst.row).to(self.device)
        j_s = torch.from_numpy(mst.col).to(self.device)
        mst_edge_weights = torch.from_numpy(mst.data).to(self.device)

        return i_s, j_s, mst_edge_weights
    def _build_hierarchy(self, i_s: torch.Tensor, j_s: torch.Tensor, mst_edge_weights: torch.Tensor, n: int
                         ) -> Tuple[List[int], List[int], List[int], List[float], List[float], List[Tuple[int, int, int]]]:
        """
        Construct the dendrogram from the MST.

        This function processes the edges of a Minimum Spanning Tree (MST) in order of increasing weight 
        and builds a binary clustering tree (dendrogram), where each merge corresponds to an edge 
        in the MST. Each merge creates a new internal cluster node, assigning it a unique cluster ID.

        Args:
            i_s (torch.Tensor): (num_edges,) source node indices of edges in the MST
            j_s (torch.Tensor): (num_edges,) target node indices of edges in the MST
            mst_edge_weights (torch.Tensor): (num_edges,) weights of the MST edges
            n (int): number of original data points (leaf nodes)

        Returns:
            Tuple[
                List[int],                  # parents: parent pointer for each node in the tree
                List[int],                  # cluster_id: id assigned to each node in the tree
                List[int],                  # cluster_sizes: size of each cluster in tree
                List[float],                # weight_appearance: edge weight at which each cluster is formed (1/lambda_birth)
                List[float],                # weight_consumed: edge weight at which each cluster is merged  (1/lambda_death)
                List[List[int]]             # children: for each internal node, the two child cluster IDs
            ]
        """

        # Sort edges by ascending weight
        sorted_weights, sorted_idx = torch.sort(mst_edge_weights, descending=False)
        i_s_sorted = i_s[sorted_idx]
        j_s_sorted = j_s[sorted_idx]

        # Initialize information for union-find structure and cluster data
        parents = [None] * (2* n-1) # Parent pointer for each cluster node
        cluster_ids = list(range(n)) # Cluster ID for each point (singleton clusters initially labelled by own index)
        cluster_sizes = [1] * n # Size of each cluster (leaves initialized)
        curr_cluster_id = n  # Next available ID for internal cluster nodes
        children = [[] for _ in range(n - 1)] # Children list for each internal node in the dendrogram

        weight_appearance = [None] * (2*n-1) # Weight at which each cluster appears
        weight_consumed = [None] * (2*n-2) + [0] # Weight at which each cluster is merged into parent

        # Union-find with path compression
        def find_parent(u):
            while parents[u] != u:
                parents[u] = parents[parents[u]]
                u = parents[u]
            return u

        # Process each edge in non-decreasing order of weight
        for u, v, weight in zip(i_s_sorted, j_s_sorted, sorted_weights):
            # Find current cluster representatives for endpoints
            cluster_id_u = find_parent(cluster_ids[u])
            cluster_id_v = find_parent(cluster_ids[v])

             # Skip if already in the same cluster (edge would create a cycle)
            if cluster_id_u == cluster_id_v:
                continue
            
            # Merge clusters: assign a new parent ID
            parents[cluster_id_u] = curr_cluster_id
            parents[cluster_id_v] = curr_cluster_id

            # Update new cluster size
            size = cluster_sizes[cluster_id_u] + cluster_sizes[cluster_id_v]
            cluster_sizes[curr_cluster_id] = size
    

            # Record weight of appearance and consumption for each cluster
            weight_appearance[curr_cluster_id] = weight
            weight_consumed[cluster_id_u] = weight
            weight_consumed[cluster_id_v] = weight

             # Register child clusters under current merged cluster
            children[curr_cluster_id-n] = [cluster_id_u, cluster_id_v]

            
            # Update cluster IDs of points u and v to reflect new cluster
            cluster_ids[u] = curr_cluster_id
            cluster_ids[v] = curr_cluster_id

             # Update ID for next internal node
            curr_cluster_id += 1


        return cluster_sizes, weight_appearance, weight_consumed, children

    def _find_clustering(self, cluster_sizes: list,
        weight_appearance: list, weight_consumed: list,  children: list, n: int) -> HDBSCANResult:

        """
        Selects clusters from the hierarchy by maximizing stability, following the HDBSCAN approach.

        Args:
            cluster_sizes (list): Size of each cluster node.
            weight_appearance (list): edge weight at which each cluster appears
            weight_consumed (list): edge weight at which each cluster is consumed
            children (list): For each internal cluster, the indices of its left and right child.
            n (int): Number of original data points.

        Returns:
            labels (torch.Tensor): Cluster label for each point (-1 means noise).
            confidence (torch.Tensor): Confidence score (lambda_point / lambda_cluster).
            stability (list): Stability score for each node in the hierarchy.
        """

        # Post-order traversal stack for stability computation
        # Start with root of dendrogram (last cluster formed)
        stack = [2*n - 2]
        # Stability score per node (None means not yet computed)
        stability = [None] * (2*n-2) + [0]   # root is dummy with stability 0

        # Convert edge weights to lambda values safely
        def safe_lambda(w):
            if w == 0:
                return float('inf')
            elif math.isinf(w):
                return 0
            else:
                return 1 / w

        # Step 1: Compute cluster stabilities in a bottom-up fashion
        while stack:
            curr_cluster_id = stack.pop()
            left_child_id, right_child_id = children[curr_cluster_id - n]

            # If both child stabilities already known, compute current stability
            if stability[left_child_id] is not None and stability[right_child_id] is not None:
                if curr_cluster_id < 2*n - 1:

                    left_contribution = 0
                    right_contribution = 0

                    # Only include stability from children that are large enough
                    if  cluster_sizes[left_child_id] >= self.min_cluster_size:
                        left_contribution = stability[left_child_id]
                    if cluster_sizes[right_child_id] >= self.min_cluster_size:
                        right_contribution = stability[right_child_id]

                    # Compute lambda 
                    lambda_consumed = safe_lambda(weight_consumed[curr_cluster_id])
                    lambda_appearance = safe_lambda(weight_appearance[curr_cluster_id])

                     # Stability = child stabilities + lifetime of cluster * size
                    stability[curr_cluster_id] = (
                        left_contribution + right_contribution +
                        cluster_sizes[left_child_id] * 
                        (lambda_consumed - lambda_appearance)
                    )

            else:
                # Recurse: push children first
                stack.append(curr_cluster_id)
                if stability[left_child_id] is None:
                    stack.append(left_child_id)
                if stability[right_child_id] is None:
                    stack.append(right_child_id)


        # Step 2: Extract final clustering based on stability criterion

        # Initialize labels and confidence scores
        labels = [-1] * n
        confidence = [0.0] * n

        #List of clusters to return
        clusters= []

        #DFS stack
        search_stack = [2*n - 2]

        # DFS to select clusters with maximum stability
        while search_stack:
            curr_cluster_id = search_stack.pop()
            left_child_id, right_child_id = children[curr_cluster_id - n]

            # If this node is more stable than the sum of its children add it to clusters
            if stability[curr_cluster_id] > stability[left_child_id] + stability[right_child_id]:
                if cluster_sizes[curr_cluster_id] >= self.min_cluster_size:
                    clusters.append(curr_cluster_id)
            
            # Otherwise, recurse on children
            else:
                search_stack.append(left_child_id)
                search_stack.append(right_child_id)
        
        
        # Step 3: Assign labels and compute confidence for points in selected clusters
        label = 0
        for cluster_id in clusters:
            cluster_stack = [cluster_id]
            while cluster_stack:
                curr_cluster_id = cluster_stack.pop()

                # Contains more than one point so we recurse
                if curr_cluster_id > n - 1:
                    left_child_id, right_child_id = children[curr_cluster_id - n]
                    cluster_stack.append(left_child_id)
                    cluster_stack.append(right_child_id)
                # Is a leaf node containing a single point so we assign label and confidence
                else:
                    labels[curr_cluster_id] = label
                    # Compute confidence: lambda_point / lambda_cluster
                    lp = safe_lambda(weight_consumed[curr_cluster_id])
                    lc = safe_lambda(weight_appearance[cluster_id])
                    if lc > 0:
                        confidence[curr_cluster_id] = min(lp / lc, 1.0)

                 # move to next cluster    
                label += 1

                
        labels = torch.tensor(labels, device=self.device)
        confidence = torch.tensor(confidence, device=self.device)
        
         # Convert to tensors
        return labels, confidence, stability



        
                    




        
