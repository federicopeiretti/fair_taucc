
import numpy as np
from functools import partial
from time import time

from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

from tauCC.src.fairness_metrics import balance_gen
from tauCC.src.utils import list_to_set
# from tauCC.src.taucc.taucc import CoClust


class FairCoclusRows():
    """
    Fair Associative Co-Clustering

    Fair Tau-CC algorithm is the fair version of Fast Tau-CC algorithm proposed by Battaglia et al., 2023.
    It finds row and column clustering such that row clustering ensures group fairness w.r.t.
    a sensitive attribute associated with row objects.

    Parameters
    ------------

    n_iterations : int, optional, default: 500
        The maximum number of iterations to be performed.
    
    n_iter_per_mode : int, optional, default: 100
        The maximum number of iterations per mode

    init : {'random', 'discrete', 'extract_centroids', 'fair'}, optional, default: 'random'
        The initialization methods.

    k: int, optional (default: 0)
        The initial number of row clusters (0 = discrete partition)
    
    l: int, optional (default: 0)
        The initial number of column clusters (0 = discrete partition)
    
    verbose: bool, optional (default: False)
        The verbosity of the algorithm
    
    random_state: int, opional (default: None)
        The seed for the random numbers generator


    Attributes
    -----------

    row_labels_ : array, length n_rows
        Results of the clustering on rows. `row_labels_[i]` is `c` if
        row `i` is assigned to cluster `c`. Available only after calling ``fit``.

    column_labels : array, length n_columns
        Results of the clustering on columns. `column_labels_[i]` is `c` if
        column `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * Battaglia E., et al., 2023. `Fast parameterless prototype-based co-clustering`
        Machine Learning, 2023

    """

    def __init__(self, n_iterations=1000, n_iter_per_mode=100, initialization='random', k=30, l=30, stop_k=0, stop_l=0, row_clusters=np.zeros(1), col_clusters=np.zeros(1), initial_prototypes=np.zeros(1), verbose=False, random_state=None):

        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of iterations per rows
        :type initialization: string
        :param initialization: the initialization method, default = 'random'
        :type k: int
        :param k: number of initial clusters on rows. 
        :type l: int
        :param l: number of initial clusters on columns. 
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        :type random_state: int | None
        :param random_state: random seed
        
        """

        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = k
        self.l = l
        self.stop_k = stop_k
        self.stop_l = stop_l
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.initial_prototypes = initial_prototypes
        self.verbose = verbose
        self.labelencoder_ = LabelEncoder()
        self.rng = np.random.default_rng(seed = random_state)

        # these fields will be available after calling fit
        self.row_labels_ = None
        self.column_labels_ = None
        self.execution_time_ = None
        
        # co-cluster assignment for each step of the algorithm
        self._row_assignment_steps = []
        self._col_assignment_steps = []
        
        # sensitive attribute about rows
        self.Sx = None
        
        # ratio of groups in the dataset = |group|/|dataset|
        self.ratio_groups_data = None
        
        # ratio of groups = |group|/k    with k: number of clusters
        self.ratio_groups = None

        # number of times the balance check fails
        self.count_fail_fairness = 0

        # KL fairness error for each iteration of both vanilla and fairness
        # self.KL_fairness_errors_fair = []
        # self.KL_fairness_errors_vanilla = []
        self.balance_fair = []
        self.balance_vanilla = []

        np.seterr(all='ignore')

        
    def _init_all(self, V, Sx=None, fair_parameters=None):

        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :param Sx: sensitive attribute about rows
        :param fair_parameters: level of fairness for each protected group with values in [0,1].
                                If value=1 then balance is required (tau=1/k), else the constraint is relaxed for the specified group.
        :return:
        """
        
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None

        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])
        
        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            
        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]

        # the number of row/ column clusters
        self._n_row_clusters = 0
        self._n_col_clusters = 0

        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents)
        self._col_assignment = np.zeros(self._n_features)
        self._tmp_row_assignment = np.zeros(self._n_documents)
        self._tmp_col_assignment = np.zeros(self._n_features)

        self._row_incidence = np.zeros((self._n_documents, self.k))
        self._col_incidence = np.zeros((self._n_features, self.l))
        
        if Sx is not None:
            # sensitive attribute Sx about rows
            if isinstance(Sx, np.ndarray) and Sx.shape[0] == self._n_documents:
                # values of sensitive attribute Sx = {s0,..,sn}
                self.Sx = Sx
                # unique values of groups
                self.Sx_unique_groups, self.Sx_n_users_in_groups = np.unique(self.Sx, return_counts=True)

                if self.k > np.min(self.Sx_n_users_in_groups):
                    raise ValueError("Choose a lower value of k. The k value must be less than or equal to the number of points belonging to the minority group.")

                self.Sx_n_groups = self.Sx_unique_groups.shape[0]
                # length of Sx
                self.Sx_length = self.Sx.shape[0]
                # protected groups = {group: user_list}
                self.Sx_groups = dict()
                
                for group in self.Sx_unique_groups:
                    users_list = np.where(self.Sx == group)[0]
                    self.Sx_groups[group] = users_list

                # ratio of groups in the dataset
                self.ratio_groups_data = self.computeRatioAllGroupsDataset()
                print(f"ratio groups in dataset: {self.ratio_groups_data}")

                # PARAMETERS IN DENOMINATOR OF FAIRNESS TAU FORMULA
                if fair_parameters is None:
                    self.tau_fair_parameters = np.full(self.Sx_n_groups, 1)
                elif fair_parameters.shape[0] == self.Sx_n_groups and np.all((fair_parameters >= 0) & (fair_parameters <= 1)):
                    self.tau_fair_parameters = fair_parameters
                else:
                    raise ValueError("fair_parameters must have the size equals to the number of protected groups.")
                
            else:
                raise Exception("The length of the sensitive attribute Sx must be equal to the number of rows of data matrix." +
                                "Each row must belong to a protected group.")

        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        self.hat_tau_x = []
        self.hat_tau_y = []
        
        if (self.initialization == 'discrete') or (self.initialization == 'random_optimal'):
            self._discrete_initialization()
        elif self.initialization == 'random':
            self._random_initialization()
        elif self.initialization == 'fair':
            self._random_initialization_fair()
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        elif self.initialization == 'vanilla':
            self._vanilla_initialization(V)
        else:
            raise ValueError("The only valid initialization methods are: random, discrete, extract_centroids, fair")
        
        if self.verbose:
            print(f'Initialization step for ({self._n_documents},{self._n_features})-sized input matrix.')
            #print(f'Sensitive attribute about documents: {self.Sx}')
            #print(f'Protected groups: {self.Sx_groups}')
            

    def fit(self, V, Sx=None, fair_parameters=None, y=None):
        """
        Fit Fair TauCC to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)
            
        Sx: array-like;
            shape of the array = (n_documents)

        fair_parameters: array-like;
            shape of the array = (n_documents)

        y : unused parameter

        Returns
        --------

        self

        """
        
        # Initialization phase
        self._init_all(V, Sx, fair_parameters)

        self._T = self._init_contingency_matrix(0)[1]
        tau_x, tau_y, hat_tau_x, hat_tau_y = self.compute_taus()
        
        self.tau_x.append(tau_x)
        self.tau_y.append(tau_y)
        self.hat_tau_x.append(hat_tau_x)
        self.hat_tau_y.append(hat_tau_y)

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0
        actual_n_iterations = 0 
        
        while actual_n_iterations < self.n_iterations:
            
            actual_iteration_x = 0    
            cont = True
            
            while cont and self._actual_n_iterations < self.n_iterations:
                # perform a move within the rows partition
                cont = self._perform_row_move()

                actual_iteration_x += 1
                self._actual_n_iterations +=1 
                
                if actual_iteration_x > self.n_iter_per_mode:
                    cont = False

                if self.verbose:
                    self._T = self._init_contingency_matrix(0)[1]
                    tau_x, tau_y, hat_tau_x, hat_tau_y = self.compute_taus()
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
                    self.hat_tau_x.append(hat_tau_x)
                    self.hat_tau_y.append(hat_tau_y)
                    print(f'Values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T at iteration: {actual_n_iterations} (on rows).')
            
            actual_iteration_y = 0
            cont = True
            
            while cont and self._actual_n_iterations < self.n_iterations:
                # perform a move within the rows partition
                cont = self._perform_col_move()

                actual_iteration_y += 1
                self._actual_n_iterations +=1 

                if actual_iteration_y > self.n_iter_per_mode:
                    cont = False

                if self.verbose:
                    self._T = self._init_contingency_matrix(1)[1]
                    tau_x, tau_y, hat_tau_x, hat_tau_y = self.compute_taus()
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
                    self.hat_tau_x.append(hat_tau_x)
                    self.hat_tau_y.append(hat_tau_y)
                    print(f'Values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T at iteration: {actual_n_iterations} (on columns).')
                
            if (actual_iteration_x == 1) and (actual_iteration_y == 1):
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
            
        end_time = time()

        if not self.verbose:
            self._T = self._init_contingency_matrix(1)[1]
            tau_x, tau_y, hat_tau_x, hat_tau_y = self.compute_taus()
            self.tau_x.append(tau_x)
            self.tau_y.append(tau_y)
            self.hat_tau_x.append(hat_tau_x)
            self.hat_tau_y.append(hat_tau_y)

        execution_time = end_time - start_time
        # clone cluster assignments and transform in lists
        self.row_labels_ = np.copy(self._row_assignment).tolist()
        self.column_labels_ = np.copy(self._col_assignment).tolist()
        self.execution_time_ = execution_time

        if self.verbose:
            print(f'Final values of tau_x: {self.tau_x[-1]:0.4f} and tau_y: {self.tau_y[-1]:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T.')
            print(f'Runtime: {self.execution_time_:0.4f} seconds.')        

        return self


    def _discrete_initialization(self):
        
        # simply assign each row to a row cluster and each column of a view to a column cluster
        self._n_row_clusters = self._n_documents
        self._n_col_clusters = self._n_features

        # assign each row to a row cluster
        self._row_assignment = np.arange(self._n_documents)
        self._row_incidence = np.identity(self._n_documents)

        # assign each column to a cluster
        self._col_assignment = np.arange(self._n_features)
        self._row_incidence = np.identity(self._n_documents)


    def _vanilla_initialization(self, V):
        from tauCC.src.taucc.taucc_vanilla_new import CoClust
        print("***** initialization: vanilla - start *****")
        init_model = CoClust(initialization='random', verbose=True, k=self.k, l=self.l, stop_k=self.stop_k, stop_l=self.stop_l)
        init_model.fit(V, Sx=None)
        self._row_assignment = np.array(init_model.row_labels_)
        self._col_assignment = np.array(init_model.column_labels_)
        self._check_clustering(0)
        self._check_clustering(1)
        print()
        print("row clus:")
        print(self._row_assignment)
        print("col clus:")
        print(self._col_assignment)
        print("***** initialization: vanilla - end *****")
        print()


    def _random_initialization(self):

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = self.rng.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = self.rng.choice(self._n_features)
        else:
            self._n_col_clusters = self.l

        # assign each row to a row cluster
        if self._n_row_clusters < self._n_documents:
            self._row_assignment = self.rng.choice(self._n_row_clusters, size = self._n_documents)
        else:
            self._row_assignment = np.arange(self._n_row_clusters).astype(int)
        
        # assign each column to a cluster
        if self._n_col_clusters < self._n_features:
            self._col_assignment = self.rng.choice(self._n_col_clusters, size = self._n_features)
        else:
            self._col_assignment = np.arange(self._n_col_clusters).astype(int)
  
        self._check_clustering(0)
        self._check_clustering(1)


    def _random_initialization_fair(self):

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = self.rng.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = self.rng.choice(self._n_features)
        else:
            self._n_col_clusters = self.l

        # assign each row to a row cluster fairly    
        if self.Sx is not None:

            if self._n_row_clusters < self._n_documents:
                clusters = np.arange(self._n_row_clusters)
            else:
                minority_group = np.argmin(self.Sx_n_users_in_groups)
                num_users_in_minority = self.Sx_n_users_in_groups[minority_group]
                clusters = np.arange(num_users_in_minority)
                
            num_users_in_clus = np.floor(self.Sx_n_users_in_groups / self._n_row_clusters).astype(int)
            self._row_assignment = np.full(self._n_documents, -1)
            
            remain = list(self.Sx_groups.values())
            Sx_groups_remain = list(map(partial(list_to_set), remain))

            for clus in clusters:

                for group, users in enumerate(Sx_groups_remain):
                    
                    users_g = np.random.choice(
                        list(users),
                        size=num_users_in_clus[group], 
                        replace=False)
                    
                    self._row_assignment[users_g] = clus
                    Sx_groups_remain[group] = Sx_groups_remain[group] - set(users_g)
                    
            not_assigned = np.where(self._row_assignment == -1)[0]
            
            if not_assigned.shape[0] != 0:
                na_clus = np.random.choice(clusters, size=not_assigned.shape)
                self._row_assignment[not_assigned] = na_clus

            print(f"Balance of initial row clustering: {np.min(balance_gen(self.Sx, self._row_assignment))}")
            
        else:
            raise ValueError("Sensitive attribute about rows is not initialized.")


        # assign each column to a cluster
        if self._n_col_clusters < self._n_features:
            self._col_assignment = self.rng.choice(self._n_col_clusters, size = self._n_features)
        else:
            self._col_assignment = np.arange(self._n_col_clusters).astype(int)
  
        self._check_clustering(0)
        self._check_clustering(1)


    def _extract_centroids_initialization(self):
        if (self.k > self._n_documents) or (self.l > self._n_features) or (self.k <= 0) or (self.l <= 0):
            raise ValueError("The number of clusters must be <= the number of objects and greater than 0, on both dimensions")

        self._n_row_clusters = self.k
        self._n_col_clusters = self.l

        if self.k == self._n_documents:
            self._row_assignment = np.arange(self._n_documents)
            self._row_incidence = np.identity(self._n_documents)
        if self.l == self._n_features:
            self._col_assignment = np.arange(self._n_features)
            self._col_incidence = np.identity(self._n_features)
        if (self.k < self._n_documents) or (self.l < self._n_features):
            a = self.rng.choice(self._n_documents, self._n_row_clusters, replace=False)
            T = self._dataset[a]
            S = np.repeat(np.sum(self._dataset, axis = 1).reshape((-1,1)), repeats = self._dataset.shape[1], axis = 1)
            B = np.nan_to_num(self._dataset/np.sum(self._dataset, axis = 0) - S)
            all_tau = np.dot(B,T.T)
            max_tau = np.max(all_tau, axis = 1)
            e_max = np.where(max_tau == all_tau.T)
            self._row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents]
            idx = np.where(max_tau <= 0)[0]
            self._row_assignment[idx] = np.arange(self._n_row_clusters,self._n_row_clusters+len(idx))
            self._check_clustering(0)
            
            dataset, T = self._init_contingency_matrix(1)
            
            b = self.rng.choice(self._n_features, self._n_col_clusters, replace = False)        
            T = dataset[:,b].T
            dataset = dataset.T
            S = np.repeat(np.sum(dataset, axis = 1).reshape((-1,1)), repeats = dataset.shape[1], axis = 1)
            B = np.nan_to_num(dataset/np.sum(dataset, axis = 0) - S)
            all_tau = np.dot(B,T.T)
            max_tau = np.max(all_tau, axis = 1)
            e_max = np.where(max_tau == all_tau.T)
            self._col_assignment[e_max[1][:self._n_features]] = e_max[0][:self._n_features]
            idx = np.where(max_tau <= 0)[0]
            self._col_assignment[idx] = np.arange(self._n_col_clusters,self._n_col_clusters+len(idx))
            self._check_clustering(1)

        self._init_k = self._n_row_clusters
        self._init_l = self._n_col_clusters

    
    def _check_clustering(self, dimension):
        if dimension == 1:
            self._col_assignment = self.labelencoder_.fit_transform(self._col_assignment.astype(int))
            self._n_col_clusters = len(np.unique(self._col_assignment))
            self._col_incidence = np.zeros((self._n_features, self._n_col_clusters))      
            self._col_incidence[np.arange(0,self._n_features,dtype='int'), self._col_assignment.astype(int)] = 1     
        elif dimension == 0:
            self._row_assignment = self.labelencoder_.fit_transform(self._row_assignment.astype(int))
            self._n_row_clusters = len(np.unique(self._row_assignment))
            self._row_incidence = np.zeros((self._n_documents, self._n_row_clusters))
            self._row_incidence[np.arange(0,self._n_documents,dtype='int'), self._row_assignment.astype(int)] = 1 
    
    
    def _init_contingency_matrix(self, dimension):
        dataset = self._update_dataset(dimension)
        #new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:
            new_t = np.dot(self._row_incidence.T, dataset)   
        else:
            new_t = np.dot(dataset, self._col_incidence)   
        return dataset, new_t

    
    def _update_dataset(self, dimension):
        if dimension == 0:
            #new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)
            new_t = np.dot(self._dataset, self._col_incidence)             
        else:
            #new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            new_t = np.dot(self._row_incidence.T, self._dataset)
        return new_t


    def _perform_row_move(self):

        dataset, T = self._init_contingency_matrix(0)
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        self._tmp_row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents]
        
        if self.Sx is not None:
            if self.check_tau_fairness(self._tmp_row_assignment) == False:
                # Compute diff matrix = max_tau - all_tau
                max_tau_matrix = np.tile(max_tau[:,np.newaxis], (1, all_tau.shape[1]))
                diff = (max_tau_matrix - all_tau)
                num_clusters = all_tau.shape[1]

                # Fair Assignment Problem
                fair_row_assignment = self.fair_assignment(diff, num_clusters)

                #print("fair row assignment: ")
                #print(fair_row_assignment)

                if self.check_tau_fairness(fair_row_assignment) == False:
                    raise ValueError("The row assignment found doesn't satisfy tau fairness!")

                # KL_fair = KL_fairness_error(fair_row_assignment, num_clusters, self.Sx)
                # num_clusters_vanilla = np.unique(self._tmp_row_assignment).shape[0]
                # KL_vanilla = KL_fairness_error(self._tmp_row_assignment, num_clusters_vanilla, self.Sx)
                
                balance_fair = np.min(balance_gen(self.Sx, fair_row_assignment))
                balance_vanilla = np.min(balance_gen(self.Sx, self._tmp_row_assignment))

                if balance_fair > balance_vanilla:
                    self._tmp_row_assignment = fair_row_assignment
                else:
                    self.count_fail_fairness += 1
                
                # self.KL_fairness_errors_fair.append(KL_fair)
                # self.KL_fairness_errors_vanilla.append(KL_vanilla)
                self.balance_fair.append(balance_fair)
                self.balance_vanilla.append(balance_vanilla)

        moves = np.sum(self._tmp_row_assignment != self._row_assignment)
        
        if moves > 0:
            self._row_assignment = self._tmp_row_assignment
            self._check_clustering(0)

        self._row_assignment_steps.append(self._row_assignment)
        
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving rows, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        
        if moves:
            return True
        else:
            return False


    def _perform_col_move(self):

        dataset, T = self._init_contingency_matrix(1)
        T = T.T
        dataset = dataset.T
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0

        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        self._tmp_col_assignment[e_max[1][:self._n_features]] = e_max[0][:self._n_features]
        moves = np.sum(self._tmp_col_assignment != self._col_assignment)
        
        if moves > 0:
            self._col_assignment = self._tmp_col_assignment
            self._check_clustering(1)
        
        self._col_assignment_steps.append(self._col_assignment)
        
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving columns, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        
        if moves:
            return True
        else:
            return False

    
    def compute_taus(self):
        tot_per_x = np.sum(self._T, 1)
        tot_per_y = np.sum(self._T, 0)
        t_square = np.power(self._T, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        
        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))

        tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))

        return tau_x, tau_y, (a_x - b_x), (a_y - b_y)

    def saveToNpy(self, path):
        self._row_assignment_steps = np.array(self._row_assignment_steps)
        self._col_assignment_steps = np.array(self._col_assignment_steps)

        with open(str(path+'/row_assignment.npy'), 'wb') as f1:
            np.save(f1, self._row_assignment_steps)

        with open(str(path+'/col_assignment.npy'), 'wb') as f2:
            np.save(f2, self._col_assignment_steps)

        with open(str(path+'/tau_x.npy'), 'wb') as f1:
            np.save(f1, self.tau_x)
        
        with open(str(path+'/tau_y.npy'), 'wb') as f1:
            np.save(f1, self.tau_y)
        return

    
    def computeRatioGroupCluster(self, cluster, group):
        users_r = np.where(self._tmp_row_assignment == cluster)[0]  # list of users in cluster r
        users_groups_r = np.take(self.Sx, users_r)                  # groups of users in cluster r
        users_gr = users_r[np.where(users_groups_r == group)[0]]    # list of users from group g_user in cluster r
        ratio = 0
        if (len(users_r) > 0):
            ratio = len(users_gr)/len(users_r)
        return ratio, users_gr                                      # return 0: ratio of protected group g in cluster r 
                                                                    # return 1: list of users from group g in cluster r
    
    def computeRatioAllGroupsCluster(self, cluster):
        return [self.computeRatioGroupCluster(cluster, group)[0] for group in self.Sx_unique_groups]
    
    # return dictionary: {cluster_id: [ratioGroupCluster(g) for g in groups]}
    def computeRatioAllGroupsAllClusters(self):
        d = {}
        for cluster in np.unique(self._tmp_row_assignment):
            d[cluster] = self.computeRatioAllGroupsCluster(cluster)
        return d
    
    def computeRatioGroupDataset(self, group):
        return self.Sx_n_users_in_groups[group]/self.Sx_length
    
    def computeRatioAllGroupsDataset(self):
        return [self.computeRatioGroupDataset(group) for group in self.Sx_unique_groups]
    
    # ratio_g = |users_g| / #clusters
    def computeRatioGroups(self, num_clusters):
        return [self.Sx_n_users_in_groups[g]/num_clusters for g in self.Sx_groups]
    
    def computeFairnessAlphaBeta(self, delta=0.2):
        alpha, beta = {}, {}
        a_val, b_val = 1 / (1 - delta), 1 - delta
        for group in self.Sx_unique_groups:
            r_i = self.ratio_groups_data[group]
            alpha[group] = a_val * r_i
            beta[group] = b_val * r_i
        return alpha, beta
    
    
    # sort diff columns on basis of group and then value
    # All males are towards start and all females are after males.
    # Among the males, the smallest of diff is at start of list as it has more valuation.
    # Similarly sort the female points among them based on diff value as did with males.
    # Return: sorted values and users
    def sort_diff_by_group_value(self, diff):
        Sx_matrix = np.repeat(self.Sx[:, np.newaxis], diff.shape[1], axis=1)
        indexes = np.lexsort((diff, Sx_matrix), axis=0) #row indices (users)
        for j in range(0,diff.shape[1]):
            diff[:,j] = diff[indexes,j][:,j]
        return diff, indexes #sorted values and users
    

    def sort_diff_by_rows(self, diff):
        diff_clus = np.argsort(diff, axis=1) 
        diff_values = np.take_along_axis(diff, diff_clus, axis=1)
        return diff_values, diff_clus
    

    def check_tau_fairness(self, vanilla_result):
        #vanilla_result : self._tmp_row_assignment (max all_tau)
        clusters = np.unique(vanilla_result)
        num_clusters = len(clusters)
        
        tau_fair = []
        for tau_percent in self.tau_fair_parameters:
            tau_fair.append((1/num_clusters)*tau_percent)
        tau_fair = np.array(tau_fair)

        tot_point_alloc_all_j = [np.floor(tau_fair[g]*self.Sx_n_users_in_groups[g]) for g in self.Sx_unique_groups]
        bool_res = False

        for clus in clusters:
            users_in_clus = np.where(vanilla_result == clus)[0]
            for group in self.Sx_unique_groups:
                users_g_in_clus = users_in_clus[np.in1d(users_in_clus, self.Sx_groups[group])]
                if len(users_g_in_clus) >= tot_point_alloc_all_j[group]:
                    bool_res = True
                else:
                    return False
        return bool_res
    
    def convert_tuple_to_list(self, tupla):
        return list(tupla)
    
    def fair_assignment(self, diff, num_clusters):

        diff_values, diff_clus = self.sort_diff_by_rows(diff)
        Sx_matrix = np.repeat(self.Sx[:, np.newaxis], diff.shape[1], axis=1) #vettore Sx replicato sulle colonne
        sorted_users = np.lexsort((diff_values, Sx_matrix), axis=0) #utenti ordinati per riga in base al gruppo e poi al valore (prima tutti M e poi tutte F)

        # Note:
        # sorted_users[0:num_male] --> male (0 in Sx)
        # sorted_users[num_male:num_female] --> female (1 in Sx)
        # and so on for all other protected groups

        num_col = num_clusters

        sorted_values = np.empty(diff_values.shape)
        sorted_clus = np.empty(diff_clus.shape)

        for j in range(0, num_col):
            users = sorted_users[:,j]                               #order of users to be considered
            sorted_values[:,j] = diff_values[users][:,j]
            sorted_clus[:,j]   = diff_clus[users][:,j].astype(int)
        sorted_clus = sorted_clus.astype(int)

        # for each cluster c = 0...num_clusters
        # a matrix with 0 and 1 values is given:
        #  - if sorted_clus[i,j] = c    then       matrix[i,j] = 1
        #  -                            otherwise  matrix[i,j] = 0

        sorted_users_for_each_cluster = []
        cluster_assignment = np.full(self.Sx_length, -1) # each user i is assigned the cluster it belongs to (-1 unassigned)
        
        tau_fair = np.array([(1/num_clusters)*tau_percent for tau_percent in self.tau_fair_parameters])

        tot_point_alloc_all_j = [np.floor(tau_fair[g]*self.Sx_n_users_in_groups[g]) for g in self.Sx_unique_groups]

        for clus in range(0, num_clusters):

            coordinate_i, coordinate_j = np.where(sorted_clus == clus)
            coordinate_tuple = list(zip(coordinate_i, coordinate_j))
            coordinate_tuple.sort(key=lambda a: a[1])
            coordinate_tuple = np.array(list(map(self.convert_tuple_to_list, coordinate_tuple)))
            users_in_clus = sorted_users[coordinate_tuple[:, 0], coordinate_tuple[:, 1]]

            users_groups_in_clus = []

            for group in self.Sx_unique_groups:

                users_g_in_clus = users_in_clus[np.in1d(users_in_clus, self.Sx_groups[group])]
                users_groups_in_clus.append(users_g_in_clus)
                # number of points to consider 
                min_users = tot_point_alloc_all_j[group]
                assigned_users = np.where(cluster_assignment != -1)
                mask = ~np.isin(users_g_in_clus, assigned_users)
                users_not_assigned = users_g_in_clus[mask]
                num_selected_users = min(min_users, len(users_not_assigned))
                if type(num_selected_users) is int:
                    selected_users = users_not_assigned[:num_selected_users]
                else:
                    selected_users = users_not_assigned[:num_selected_users.astype(int)]
                cluster_assignment[selected_users] = clus

            sorted_users_for_each_cluster.append(users_groups_in_clus)

        users_not_yet_assigned = np.where(cluster_assignment == -1)[0]
        cluster_assignment[users_not_yet_assigned] = self._tmp_row_assignment[users_not_yet_assigned]

        return cluster_assignment
