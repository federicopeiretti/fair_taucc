# UPDATE:
# - Add self.individual_taus_x and self.individual_taus_y


import numpy as np
from functools import partial
from time import time

from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

from tauCC.src.fairness_metrics import balance_gen
from tauCC.src.utils import list_to_set
# from tauCC.src.taucc.taucc import CoClust


class FairCoclus():
    """
    Fair Associative Co-Clustering

    Fair Tau-CC algorithm is the fair version of Fast Tau-CC algorithm proposed by Battaglia et al., 2023.
    It finds row and column clustering such that row clustering ensures group fairness w.r.t.
    a sensitive attribute associated with row objects
    and a sensitive attribute associated with column objects

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

    def __init__(self, n_iterations=1000, n_iter_per_mode=100, initialization='random', k=10, l=10, row_clusters=np.zeros(1), col_clusters=np.zeros(1), initial_prototypes=np.zeros(1), verbose=False, random_state=None):

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
        
        # sensitive attributes
        self.Sx = None
        self.Sy = None
        
        # ratio of groups in the dataset = |group|/|dataset|
        self.Sx_ratio_groups = None
        self.Sy_ratio_groups = None

        # number of times the balance check fails
        self.count_fail_fairness = 0

        # Balance for each iteration of both vanilla and fairness
        self.balance_fair = []
        self.balance_vanilla = []

        # similarity of each item with its final cluster
        self.individual_taus_x = None
        self.individual_taus_y = None

        np.seterr(all='ignore')

        
    def _init_all(self, V, Sx=None, Sy=None, fair_row_parameters=None, fair_col_parameters=None):

        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :param Sx: sensitive attribute about rows
        :param Sy: sensitive attribute about columns
        :param fair_row_parameters: level of fairness for each protected group (associated with row items) with values in [0,1].
                                If value=1 then balance is required (tau=1/k), else the constraint is relaxed for the specified group.
        :param fair_col_parameters: level of fairness for each protected group (associated with column items) with values in [0,1].
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

        # initialization of sensitive attributes Sx and Sy
        self.init_Sx(Sx, fair_row_parameters)
        self.init_Sy(Sy, fair_col_parameters)

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
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        else:
            raise ValueError("The only valid initialization methods are: random, discrete, extract_centroids")
        
        if self.verbose:
            print(f'Initialization step for ({self._n_documents},{self._n_features})-sized input matrix.')
            

    def fit(self, V, Sx=None, Sy=None, fair_row_parameters=None, fair_col_parameters=None, y=None):
        """
        Fit Fair TauCC to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)
            
        Sx: array-like;
            shape of the array = (n_documents)

        Sy: array-like;
            shape of the array = (n_features)

        fair_row_parameters: array-like;
            shape of the array = (n_documents)

        fair_col_parameters: array-like;
            shape of the array = (n_features)

        y : unused parameter

        Returns
        --------

        self

        """
        
        # Initialization phase
        self._init_all(V, Sx, Sy, fair_row_parameters, fair_col_parameters)

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
        self._tmp_row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents].astype(int)

        if self.Sx is not None:
            if self.check_tau_fairness(self._tmp_row_assignment, "rows") == False:
                # Compute diff matrix = max_tau - all_tau
                max_tau_matrix = np.tile(max_tau[:,np.newaxis], (1, all_tau.shape[1]))
                diff = (max_tau_matrix - all_tau)
                num_clusters = all_tau.shape[1]

                # Fair Assignment Problem
                fair_row_labels = self.fair_row_assignments(diff, num_clusters)

                #if self.check_tau_fairness(fair_row_labels, "rows") == False:
                #    raise ValueError("The row assignments found do not satisfy tau fairness.")

                self._tmp_row_assignment = fair_row_labels
                self.individual_taus_x = all_tau[np.arange(len(fair_row_labels)), fair_row_labels]

                #balance_fair = balance_gen(self.Sx, fair_row_labels)
                #balance_vanilla = balance_gen(self.Sx, self._tmp_row_assignment)
                #print("Balance fairness: ", balance_fair)
                #print("Balance vanilla: ", balance_vanilla)
            else:
                self.individual_taus_x = max_tau.copy()

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

        if self.Sy is not None:
            if self.check_tau_fairness(self._tmp_col_assignment, "cols") == False:
                # Compute diff matrix = max_tau - all_tau
                max_tau_matrix = np.tile(max_tau[:, np.newaxis], (1, all_tau.shape[1]))
                diff = (max_tau_matrix - all_tau)
                num_clusters = all_tau.shape[1]

                # Fair Assignment Problem
                fair_col_labels = self.fair_col_assignments(diff, num_clusters)

                #if self.check_tau_fairness(fair_col_labels, "cols") == False:
                #    raise ValueError("The column assignments found do not satisfy tau fairness.")

                # if balance_fair > balance_vanilla:
                self._tmp_col_assignment = fair_col_labels
                self.individual_taus_y = all_tau[np.arange(len(fair_col_labels)), fair_col_labels]

                #balance_fair = balance_gen(self.Sy, fair_col_labels)
                #balance_vanilla = balance_gen(self.Sy, self._tmp_col_assignment)
                #print("Balance fairness: ", balance_fair)
                #print("Balance vanilla: ", balance_vanilla)
            else:
                self.individual_taus_y = max_tau.copy()

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

        with open(str(path + '/row_assignment.npy'), 'wb') as f1:
            np.save(f1, np.array(self.row_labels_))

        with open(str(path + '/col_assignment.npy'), 'wb') as f2:
            np.save(f2, np.array(self.column_labels_))

        with open(str(path+'/tau_x.npy'), 'wb') as f1:
            np.save(f1, self.tau_x)
        
        with open(str(path+'/tau_y.npy'), 'wb') as f1:
            np.save(f1, self.tau_y)
        return


    def compute_ratio_Sx_groups_dataset(self):
        return [self.Sx_n_users_in_groups[group]/self.Sx_length for group in self.Sx_unique_groups]


    def compute_ratio_Sy_groups_dataset(self):
        return [self.Sy_n_users_in_groups[group]/self.Sy_length for group in self.Sy_unique_groups]


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
    

    def check_tau_fairness(self, clustering, dimension):
        #vanilla_result : self._tmp_row_assignment (max all_tau)
        clusters = np.unique(clustering)
        num_clusters = len(clusters)
        
        if dimension == "rows":
            tau_fair = []
            for alpha in self.alpha_rows:
                tau_fair.append((1/num_clusters) * alpha)
            tau_fair = np.array(tau_fair)
    
            tot_point_alloc_all_j = [np.floor(tau_fair[g]*self.Sx_n_users_in_groups[g]) for g in self.Sx_unique_groups]

            bool_res = False

            for clus in clusters:
                clus = int(clus)
                users_in_clus = np.where(clustering == clus)[0]

                for group in self.Sx_unique_groups:
                    users_g_in_clus = users_in_clus[np.in1d(users_in_clus, self.Sx_groups[group])]
                    if len(users_g_in_clus) >= tot_point_alloc_all_j[group]:
                        bool_res = True
                    else:
                        return False
            return bool_res
        
        elif dimension == "cols":
            tau_fair = []
            for alpha in self.alpha_cols:
                tau_fair.append((1 / num_clusters) * alpha)
            tau_fair = np.array(tau_fair)

            tot_point_alloc_all_j = [np.floor(tau_fair[g] * self.Sy_n_users_in_groups[g]) for g in
                                     self.Sy_unique_groups]

            bool_res = False

            for clus in clusters:
                clus = int(clus)
                users_in_clus = np.where(clustering == clus)[0]
                for group in self.Sy_unique_groups:
                    users_g_in_clus = users_in_clus[np.in1d(users_in_clus, self.Sy_groups[group])]
                    if len(users_g_in_clus) >= tot_point_alloc_all_j[group]:
                        bool_res = True
                    else:
                        return False
            return bool_res
            
    
    def convert_tuple_to_list(self, tupla):
        return list(tupla)
    
    def fair_row_assignments(self, diff, num_clusters):

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
        
        tau_fair = np.array([(1/num_clusters) * alpha for alpha in self.alpha_rows])

        tot_point_alloc_all_j = [np.floor(tau_fair[g]*self.Sx_n_users_in_groups[g]) for g in self.Sx_unique_groups]

        if np.all(np.array(tot_point_alloc_all_j) == 0):
            optimal_clusters = np.unique(self._tmp_row_assignment.astype(int))
            if len(optimal_clusters) < np.min(self.Sx_n_users_in_groups):
                return self.fair_row_assignments(diff[:, optimal_clusters], len(optimal_clusters))
            else:
                raise ValueError("Try with a initialization of lower k value.")

        for clus in range(0, num_clusters):
            clus = int(clus)
            coordinate_i, coordinate_j = np.where(sorted_clus == clus)
            coordinate_tuple = list(zip(coordinate_i, coordinate_j))
            coordinate_tuple.sort(key=lambda a: a[1])
            coordinate_tuple = np.array(list(map(self.convert_tuple_to_list, coordinate_tuple)))
            users_in_clus = sorted_users[coordinate_tuple[:, 0], coordinate_tuple[:, 1]]

            users_groups_in_clus = []

            for group in self.Sx_unique_groups:
                group = int(group)
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
        # remaining users assigned to the optimal cluster (argmin diff matrix)
        #cluster_assignment[users_not_yet_assigned] = self._tmp_row_assignment[users_not_yet_assigned]
        cluster_assignment[users_not_yet_assigned] = np.argmin(diff[users_not_yet_assigned, :], axis=1)
        return cluster_assignment


    def fair_col_assignments(self, diff, num_clusters):

        diff_values, diff_clus = self.sort_diff_by_rows(diff)
        Sy_matrix = np.repeat(self.Sy[:, np.newaxis], diff.shape[1], axis=1)
        sorted_users = np.lexsort((diff_values, Sy_matrix), axis=0)

        num_col = num_clusters

        sorted_values = np.empty(diff_values.shape)
        sorted_clus = np.empty(diff_clus.shape)

        for j in range(0, num_col):
            users = sorted_users[:, j]  # order of users to be considered
            sorted_values[:, j] = diff_values[users][:, j]
            sorted_clus[:, j] = diff_clus[users][:, j].astype(int)
        sorted_clus = sorted_clus.astype(int)

        sorted_users_for_each_cluster = []
        cluster_assignment = np.full(self.Sy_length, -1)

        tau_fair = np.array([(1 / num_clusters) * alpha for alpha in self.alpha_cols])

        tot_point_alloc_all_j = [np.floor(tau_fair[g] * self.Sy_n_users_in_groups[g]) for g in self.Sy_unique_groups]

        if np.all(np.array(tot_point_alloc_all_j) == 0):
            optimal_clusters = np.unique(self._tmp_col_assignment.astype(int))
            if len(optimal_clusters) < np.min(self.Sy_n_users_in_groups):
                return self.fair_col_assignments(diff[:, optimal_clusters], len(optimal_clusters))
            else:
                raise ValueError("Try with a initialization of lower l value.")

        for clus in range(0, num_clusters):
            clus = int(clus)
            coordinate_i, coordinate_j = np.where(sorted_clus == clus)
            coordinate_tuple = list(zip(coordinate_i, coordinate_j))
            coordinate_tuple.sort(key=lambda a: a[1])
            coordinate_tuple = np.array(list(map(self.convert_tuple_to_list, coordinate_tuple)))
            users_in_clus = sorted_users[coordinate_tuple[:, 0], coordinate_tuple[:, 1]]

            users_groups_in_clus = []

            for group in self.Sy_unique_groups:
                users_g_in_clus = users_in_clus[np.in1d(users_in_clus, self.Sy_groups[group])]
                users_groups_in_clus.append(users_g_in_clus)
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
        # remaining users assigned to the optimal cluster (argmin diff matrix)
        #cluster_assignment[users_not_yet_assigned] = self._tmp_col_assignment[users_not_yet_assigned]
        cluster_assignment[users_not_yet_assigned] = np.argmin(diff[users_not_yet_assigned,:], axis=1)
        return cluster_assignment


    def init_Sx(self, Sx, fair_row_parameters):
        if Sx is not None:
            if isinstance(Sx, np.ndarray) and Sx.shape[0] == self._n_documents:
                # values of sensitive attribute Sx = {s0,..,sn}
                self.Sx = Sx
                # unique values of groups
                self.Sx_unique_groups, self.Sx_n_users_in_groups = np.unique(self.Sx, return_counts=True)

                if self.k > np.min(self.Sx_n_users_in_groups):
                    raise ValueError(
                        "Choose a lower value of k. The k value must be less than or equal to the number of points belonging to the minority group.")

                self.Sx_n_groups = self.Sx_unique_groups.shape[0]
                # length of Sx
                self.Sx_length = self.Sx.shape[0]
                # protected groups = {group: user_list}
                self.Sx_groups = dict()

                for group in self.Sx_unique_groups:
                    users_list = np.where(self.Sx == group)[0]
                    self.Sx_groups[group] = users_list

                # ratio of groups in the dataset
                self.Sx_ratio_groups = self.compute_ratio_Sx_groups_dataset()
                print(f"Sx ratio groups in dataset: {self.Sx_ratio_groups}")

                # PARAMETERS IN DENOMINATOR OF FAIRNESS TAU FORMULA
                if fair_row_parameters is None:
                    self.alpha_rows = np.full(self.Sx_n_groups, 1)
                elif fair_row_parameters.shape[0] == self.Sx_n_groups and np.all(
                        (fair_row_parameters >= 0) & (fair_row_parameters <= 1)):
                    self.alpha_rows = fair_row_parameters
                else:
                    raise ValueError("fair_row_parameters must have the size equals to the number of protected groups.")

            else:
                raise Exception(
                    "The length of the sensitive attribute Sx must be equal to the number of rows of data matrix." +
                    "Each row must belong to a protected group.")


    def init_Sy(self, Sy, fair_col_parameters):
        if Sy is not None:
            if isinstance(Sy, np.ndarray) and Sy.shape[0] == self._n_documents:
                # values of sensitive attribute Sy = {s0,..,sn}
                self.Sy = Sy
                # unique values of groups
                self.Sy_unique_groups, self.Sy_n_users_in_groups = np.unique(self.Sy, return_counts=True)

                if self.l > np.min(self.Sy_n_users_in_groups):
                    raise ValueError(
                        "Choose a lower value of l. The l value must be less than or equal to the number of points belonging to the minority group.")

                self.Sy_n_groups = self.Sy_unique_groups.shape[0]
                # length of Sy
                self.Sy_length = self.Sy.shape[0]
                # protected groups = {group: user_list}
                self.Sy_groups = dict()

                for group in self.Sy_unique_groups:
                    users_list = np.where(self.Sy == group)[0]
                    self.Sy_groups[group] = users_list

                # ratio of groups in the dataset
                self.Sy_ratio_groups = self.compute_ratio_Sy_groups_dataset()
                print(f"Sy ratio groups in dataset: {self.Sy_ratio_groups}")

                # PARAMETERS IN DENOMINATOR OF FAIRNESS TAU FORMULA
                if fair_col_parameters is None:
                    self.alpha_cols = np.full(self.Sy_n_groups, 1)
                elif fair_col_parameters.shape[0] == self.Sy_n_groups and np.all(
                        (fair_col_parameters >= 0) & (fair_col_parameters <= 1)):
                    self.alpha_cols = fair_col_parameters
                else:
                    raise ValueError("fair_col_parameters must have the size equals to the number of protected groups.")

            else:
                raise Exception(
                    "The length of the sensitive attribute Sy must be equal to the number of rows of data matrix." +
                    "Each row must belong to a protected group.")

    def print_Sy(self):
        print("***** Sy *****")
        print(f"shape: {self.Sy_length}")
        print(self.Sy)

        print(f"num protected groups: {self.Sy_n_groups}")
        print(f"protected groups: {self.Sy_unique_groups}")
        print(f"num users in groups: {self.Sy_n_users_in_groups}")
        print(f"Sy ratio groups in dataset: {self.Sy_ratio_groups}")

        # protected groups = {group: user_list}
        print("users in groups: ")
        print(f"{self.Sy_groups}")

        print("**************")
        print()

    def print_Sx(self):
        print("***** Sx *****")
        print(f"shape: {self.Sx_length}")
        print(self.Sx)

        print(f"num protected groups: {self.Sx_n_groups}")
        print(f"protected groups: {self.Sx_unique_groups}")
        print(f"num users in groups: {self.Sx_n_users_in_groups}")
        print(f"ratio groups in dataset: {self.Sx_ratio_groups}")

        # protected groups = {group: user_list}
        print("users in groups: ")
        print(f"{self.Sx_groups}")

        print("**************")
        print()
