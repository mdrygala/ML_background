from dataclasses import dataclass
import torch
from typing import Optional, Callable, Tuple
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


@dataclass
class HDBSCANResult:
    labels: torch.Tensor # Cluster labels for each point
    confidence: Optional[torch.Tensor] = None # confidence scores for each point that its correctly clustered
    lambda_cluster: Optional[torch.Tensor] = None # lambda values for each cluster
    lambdas_point_cluster: Optional[torch.Tensor] = None # lambda values for each point in its cluster
    mst_edges: Optional[torch.Tensor] = None # edges in the minimum spanning tree

class HDBSCAN_Reimpl:
    def __init__(
        self,
        min_pts: int,
        min_cluster_size: int,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], torch.tensor]] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        chunk_size: int = 512,
        use_faiss: bool = False
    ):
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.k = min_pts
        self.device = device if device else 'cpu'
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.use_faiss = use_faiss

        ## Initialize other parameters
        self.labels_ = None
        self.confidence_ = None
        self.result_ = None
        
        self.mst_ = None

        def fit(self, X: torch.Tensor) -> 'HDBSCAN_Reimpl':
            X = X.to(self.device)
            n = X.shape[0]

            if self.use_faiss:
                knn_dist, knn_idx = self._knn_faiss(X, k=self.k)
            else:
                knn_dist, knn_idx = self._knn(X, k=self.k)

            core_distances = self._core_distances(knn_dist, min_pts=self.min_pts)
            i_indicies, j_indicies, mreach = self._mutual_reachability(knn_dist, knn_idx, core_distances)

            mst_i_s, mst_j_s, mst_edge_weights = self._mst(mreach, i_indicies, j_indicies, n)
            self.mst_ = (mst_i_s, mst_j_s, mst_edge_weights)

            parents, cluster_id, cluster_size, weight_appearance, weight_consumed, children = self._build_hierarchy(mst_i_s, mst_j_s, mst_edge_weights, n)
            
            labels, confidence, stability = self._find_clustering(
                parents, cluster_id, cluster_size, weight_appearance, weight_consumed, children, n
            )

            self.labels_ = labels
            self.confidence_ = confidence

            self.result_ = HDBSCANResult(
                labels=labels,
                confidence=confidence,
                lambda_cluster=torch.tensor([...], device=self.device), 
                lambdas_point_cluster=torch.tensor([...], device=self.device), 
                mst_edges=torch.stack([mst_i_s, mst_j_s, mst_edge_weights], dim=1)
)

            return self

            





            return self
        def _pairwise_distances(
            self, X: torch.Tensor,  Y: torch.Tensor) -> torch.Tensor:

            if self.metric is not None:
                return self.metric(X, Y)
            else:
                return torch.cdist(X, Y)
        
        def _knn(self, X: torch.Tensor, k:int) -> Tuple[torch.Tensor, torch.Tensor]:
            
            n = X.shape[0]
            knn_dist = []
            knn_idx = []

            chunk_size = 512

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                X_chunk = X[start:end]
                distances = self._pairwise_distances(X_chunk, X)
                knn_dist_chunk, knn_idx_chunk = torch.topk(distances, k=k, largest=False)
                knn_dist.append(knn_dist_chunk)
                knn_idx.append(knn_idx_chunk)
            return torch.cat(knn_dist, dim=0), torch.cat(knn_idx, dim=0)

        def _knn_faiss(self, X: torch.Tensor, k:int) -> Tuple[torch.Tensor, torch.Tensor]:
            X_np = X.detach().cpu().numpy().astype('float32')  # FAISS needs float32
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
            core_distances = knn_dist[:, min_pts - 1]
            return core_distances
        
        def _mutual_reachability(self, knn_dist: torch.Tensor, knn_idx: torch.Tensor, core_distances: torch.Tensor) -> torch.Tensor:
            n, k = knn_idx.shape

            i_indicies = torch.arange(m, devices=self.device).repeat_interleave(k)
            j_indicies = knn_idx.view(-1)
            wij = knn_dist.reshape(-1)

            core_i = core_distances[i_indicies]
            core_j = core_distances[j_indicies]

            mreach = torch.max(torch.stack([core_i, core_j, wij], dim=1), dim=1).values
            return i_indicies, j_indicies, mreach

        def _mst(self, mreach: torch.Tensor, i_indicies: torch.Tensor, j_indicies: torch.tensor, n: int) -> torch.Tensor:
            coo = coo_matrix((mreach.cpu().numpy(), (i_indices.cpu().numpy(), j_indices.cpu().numpy())), shape=(n, n))
            mst = minimum_spanning_tree(coo).tocoo()

            i_s = torch.from_numpy(mst_sparse.row).to(self.device)
            j_s = torch.from_numpy(mst_sparse.col).to(self.device)
            mst_edge_weights = torch.from_numpy(mst_sparse.data).to(self.device)

            return i_s, j_s, mst_edge_weights
        def _build_hierarchy(self, i_s: torch.Tensor, j_s: torch.Tensor, mst_edge_weights: torch.Tensor, n: int) -> Tuple[List[int], List[int], List[int], List[float], List[float], List[Tuple[int, int, int]]]:
            
            sorted_weights, sorted_idx = torch.sort(mst_edge_weights, descending=False)
            i_s_sorted = i_s[sorted_idx]
            j_s_sorted = j_s[sorted_idx]

            parents = list(range(2 * n-1))
            cluster_id = list(range(n))
            cluster_size = [1] * (2 * n-1)
            curr_cluster_id = n
            children = [[] for _ in range(n - 1)]

            weight_appearance = [None] * (2*n-1)
            weight_consumed = [None] * (2*n-2) + [0]

            def find_parent(u):
                while parents[u] != u:
                    parents[u] = parents[parents[u]]
                    u = parents[u]
                return u

            for u, v, weight in zip(i_s_sorted, j_s_sorted, sorted_weights):
                cluster_id_u = find_parent(cluster_id[u])
                cluster_id_v = find_parent(cluster_id[v])

                if cluster_id_u == cluster_id_v:
                    continue

                parents[cluster_id_u] = curr_cluster_id
                parents[cluster_id_v] = curr_cluster_id

                size = cluster_size[cluster_id_u] + cluster_size[cluster_id_v]
                cluster_size[curr_cluster_id] = size
        

                weight_appearance[curr_cluster_id] = weight
                weight_consumed[cluster_id_u] = weight
                weight_consumed[cluster_id_v] = weight

                children[curr_cluster_id-n] = [cluster_id_u, cluster_id_v]

                

                cluster_id[u] = curr_cluster_id
                cluster_id[v] = curr_cluster_id
                curr_cluster_id += 1


            return parents, cluster_id, cluster_size, weight_appearance, weight_consumed, children

        def _find_clustering(self, parents: list, cluster_id: list , cluster_size: list,
         weight_appearance: list, weight_consumed: list,  children: list, n: int) -> HDBSCANResult:

            stack = []
            stability = [None] * (2*n-2) + [0]

            stack.append(2*n - 2)

            while stack:
                curr_cluster_id = stack.pop()
                left_child_id, right_child_id = children[curr_cluster_id - n]

                if stability[left_child_id] is not None and stability[right_child_id] is not None:
                    if curr_cluster_id < 2*n - 1:

                        left_contribution = 0
                        right_contribution = 0
                        if  cluster_size[left_child_id] >= self.min_cluster_size:
                            left_contribution = stability[left_child_id]
                        if cluster_size[right_child_id] >= self.min_cluster_size:
                            right_contribution = stability[right_child_id]

                        stability[curr_cluster_id] = (
                            left_contribution + right_contribution +
                            cluster_size[left_child_id] * 
                            (1/weight_consumed[curr_cluster_id] - 1/weight_appearance[curr_cluster_id])
                        )

                else:
                    stack.append(curr_cluster_id)
                    if stability[left_child_id] is None:
                        stack.append(left_child_id)
                    if stability[right_child_id] is None:
                        stack.append(right_child_id)

            labels = [-1] * n
            confidence = [0.0] * n

            clusters= []
            search_stack = []
            search_stack.append(2*n - 2)

            while search_stack:
                curr_cluster_id = stack.pop()
                left_child_id, right_child_id = children[curr_cluster_id - n]

                if stability[curr_cluster_id] > stability[left_child_id] + stability[right_child_id]:
                    if clusters_size[curr_cluster_id] >= self.min_cluster_size:
                        clusters_stack.append(curr_cluster_id)
                
                else:
                    search_stack.append(left_child_id)
                    search_stack.append(right_child_id)
            
            label = 0
            for cluster_id in clusters:
                cluster_stack = [cluster_id]
                while cluster_stack:
                    curr_cluster_id = cluster_stack.pop()
                    if curr_cluster_id > n - 1:
                        left_child_id, right_child_id = children[curr_cluster_id - n]
                        cluster_stack.append(left_child_id)
                        cluster_stack.append(right_child_id)
                    else:
                        labels[curr_cluster_id] = label

                        # Compute confidence[i] = lambda_point[i] / lambda_cluster[cid]
                        lp = 1.0 / weight_consumed[curr_cluster_id] if weight_consumed[curr_cluster_id] else 0.0
                        lc = 1.0 / weight_appearance[cid] if weight_appearance[cid] else 0.0
                        if lc > 0:
                            confidence[curr_cluster_id] = min(lp / lc, 1.0)  # clamp to [0,1]
                            label += 1

                    
            labels = torch.tensor(labels, device=self.device)
            confidence = torch.tensor(confidence, device=self.device)
            

            return labels, confidence, stability



        
                    




        
