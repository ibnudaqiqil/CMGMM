
import numpy as np
from numpy.linalg import det

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.neighbors import NearestNeighbors
import logging

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max
def _outer(x, y):
    """
    Computes the outer production between 1d-ndarrays x and y.
    """
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res



def normal(x, mu, cov):
    """
    Normal distribution with parameters mu (mean) and cov (covariance matrix)
    """
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * det(cov))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(cov), x-mu)))

class CMGMM(GaussianMixture):
    '''
    classdocs
    '''

    def __init__(self,min_components=3, max_step_components=30, max_components=60,distance_method="Kullback-leiber",merge_type="moment_presaving",pruneComponent=True):

        GaussianMixture.__init__(self, n_components=min_components,covariance_type='full',random_state=121)


        self.min_components = min_components
        self.max_components = max_components
        self.type='CMGMM'
        self.distance_method = distance_method #Kullback-leiber,ISD, Jensen-shannon
        self.merge_type = merge_type #moment_presaving,isomorphic
        self.initialized=False
        self.verbose=False
        self.pruneComponent=pruneComponent

    def fit(self, data):
        if (len(data)<=20):
            return 0
        best_gmm = self.trainBestModel(data)
        if (self.initialized==False):            
            self.weights_ = best_gmm.weights_
            self.covariances_ = best_gmm.covariances_ # self.covariances_ = gmm._get_covars()
            self.means_ = best_gmm.means_
            self.n_components = best_gmm.n_components
            self.precisions_cholesky_ = _compute_precision_cholesky(best_gmm.covariances_, "full")
            self.initialized= True
            logging.debug(f'TRAIN AWAL component : {best_gmm.n_components} \t W: {best_gmm.weights_} ')
        else:
            
            w_all = np.concatenate((self.weights_ , best_gmm.weights_),axis=None)
            mu_all =  np.concatenate((self.means_ , best_gmm.means_ ),axis=0)
            cov_all = np.concatenate((self.covariances_ , best_gmm.covariances_ ),axis=0)
            n_components_range =  range(self.n_components+best_gmm.n_components, self.n_components-1, -1)
            #logging.debug(f'Search  from:{n_components_range}')
            bicreduced=[]
            lowest_bic = np.infty
            jumlahSample = 5*len(data)

            
            currentSample = self.sample(2*jumlahSample)[0]
            dataxx = np.concatenate((currentSample, data), axis=0)
            for n_components in n_components_range:
                #print(n_components)
                w,m,c = self.mixture_reduction(w_all, mu_all, cov_all, n_components, isomorphic=True, verbose=False, optimization=False)
                gmm_p = GaussianMixture(n_components=n_components,covariance_type="full")
                gmm_p.weights_ = w
                gmm_p.covariances_ = c
                gmm_p.means_ = m                
                gmm_p.precisions_cholesky_ = _compute_precision_cholesky(c, "full")
                bic_= gmm_p.bic(dataxx)
                bicreduced.append(bic_)
                
                #print('REDUCD BIC components {0} = {1}'.format(n_components, bic_))  
                if bic_ < lowest_bic:
                    lowest_bic = bic_
                    best_gmm = gmm_p
                
            logging.debug(f'N_component Awal: {self.n_components} \t Drift Comp: {best_gmm.n_components}  ')
            self.weights_ = best_gmm.weights_ /np.sum(best_gmm.weights_ )
            self.means_ = best_gmm.means_
            self.covariances_= best_gmm.covariances_
            self.n_components = best_gmm.n_components
            #print("W awal:", self.weights_)
            #Compute the Cholesky decomposition of the precisions.
            self.prune()
            #print("W Prune:", self.weights_)
            self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

    def prune(self, margin=0.009):
        if(self.pruneComponent):
            
            mask = self.weights_ > margin
            self.weights_ = self.weights_[mask]
            self.means_ = self.means_[mask]
            self.covariances_= self.covariances_[mask]

            self.n_components = len(self.weights_)

    def trainBestModel(self, X):
          #X=np.expand_dims(samples,1)
        bic = []
        #print(X)
        n_components_range = range(self.min_components , self.max_components+1)
        #cv_types = ['spherical', 'tied', 'diag', 'full']
        cv_types = ['full']
        lowest_bic = np.infty
        best_components=0
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,covariance_type="full")
            gmm.fit(X)
            bic_= gmm.bic(X)
            bic.append(bic_)
            if self.verbose: 
                 print('BIC components {0} = {1}'.format(n_components, bic_))  
            if bic_ < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_components = n_components
        if (self.verbose==True): print("Selected Component {0}".format(best_components))
        return best_gmm



    def mixture_reduction(self, w, mu, cov,  target_comp, isomorphic=False, verbose=True, optimization=True):

        # original size of the mixture
        M = len(w) 
        # target size of the mixture
        N = target_comp
        # dimensionality of data
        d = mu.shape[1]

        # we consider neighbors at a radius equivalent to the lenght of 5 pixels
        if cov.ndim==1:
            maxsig = 5*np.max(cov)
            # if cov is 1-dimensional we convert it to its covariance matrix form
            cov = np.asarray( [(val**2)*np.identity(d) for val in cov] )
        else:
            maxsig = 5*max([np.max(np.linalg.eig(_cov)[0])**(1./2) for _cov in cov])

        indexes = np.arange(M, dtype=np.int32)
        nn,nn_indexes = self.compute_neighbors(mu, maxsig)

        # idea: keep track that the k-th component was merged into the l-th positon
        merge_mapping = np.arange(M, dtype=np.int32)

        # max number of neighbors
        max_neigh = nn_indexes.shape[1]
        
        # computing the initial dissimilarity matrix
        diss_matrix = self.build_diss_matrix(w, mu, cov, nn_indexes)  
        
        # main loop
        while M>N:
            i_min, j_min = self.least_dissimilar(diss_matrix, indexes, nn_indexes)
            if self.verbose: 
                 print('Merged components {0} and {1}'.format(i_min, j_min))  
            w_m, mu_m, cov_m = self.merge(w[i_min], mu[i_min], cov[i_min], 
                                     w[j_min], mu[j_min], cov[j_min])
     
            # updating structures
            nindex = min(i_min,j_min) # index of the new component
            dindex = max(i_min,j_min) # index of the del component
            w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
            indexes = np.delete(indexes, self.get_index(indexes,dindex))
            self.update_merge_mapping(merge_mapping, nindex, dindex)
            nn_indexes[nindex] = self.radius_search(nn, mu_m, max_neigh, merge_mapping, nindex, dindex)
            self.update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)
            M -= 1

        # indexes of the "alive" mixture components
        return w[indexes],mu[indexes],cov[indexes]


    def compute_neighbors(self, mu_center, maxsig):
        nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
        nn.fit(mu_center)
        neigh_indexes_arr = nn.radius_neighbors(mu_center, return_distance=False)
        
        # creating the initial array
        maxlen = 0
        for arr in neigh_indexes_arr:
            if len(arr)>maxlen:
                maxlen = len(arr)
        neigh_indexes = MAXINT*np.ones((len(neigh_indexes_arr),maxlen-1), dtype=np.int32)
        
        # filling it with the correct indexes
        for i,arr in enumerate(neigh_indexes_arr):
            ll = arr.tolist(); ll.remove(i); ll.sort()
            for j,index in enumerate(ll):
                neigh_indexes[i,j] = index      
        return nn,neigh_indexes

    def build_diss_matrix(self, w, mu, cov,  nn_indexes):

        M,max_neigh = nn_indexes.shape
        diss_matrix = -1.*np.ones((M,max_neigh))
        for i in range(M):
            for j in range(max_neigh):
                jj = nn_indexes[i,j]
                if jj==MAXINT: break
                diss_matrix[i,j] = self.mixture_distance(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])  
        return diss_matrix

    def get_index(self,array, value):
        n = len(array)
        for i in range(n):
            if array[i]==value: return i
        return -1

    def least_dissimilar(self, diss_matrix, indexes, nn_indexes):
        max_neigh = diss_matrix.shape[1]
        i_min = -1; j_min = -1
        diss_min = np.inf
        for i in indexes:
            for j in range(max_neigh):
                if diss_matrix[i,j]==-1: break
                if diss_matrix[i,j]<diss_min:
                    diss_min = diss_matrix[i,j]
                    i_min = i
                    j_min = nn_indexes[i,j]
        return i_min,j_min

    def merge(self, w1, mu1, cov1, w2, mu2, cov2):
        if (self.merge_type == "moment_presaving"):
            return self.moment_preserving_merge(w1, mu1, cov1, w2, mu2, cov2)
        else:
            return self.isomorphic_merge(w1, mu1, cov1, w2, mu2, cov2)

    def moment_preserving_merge(self, w1, mu1, cov1, w2, mu2, cov2):
        """
        Computes the moment preserving merge of components (w1,mu1,cov1) and
        (w2,mu2,cov2)
        """
        w_m = w1+w2
        mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
        cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2)*_outer(mu1-mu2, mu1-mu2)
        return (w_m, mu_m, cov_m)


    def isomorphic_merge(self, w1, mu1, cov1, w2, mu2, cov2):
        """
        Computes the isomorphic moment preserving merge of components (w1,mu1,cov1) and
        (w2,mu2,cov2)
        """
        d = len(mu1)
        w_m = w1+w2
        mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
        cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2) * np.abs(det(_outer(mu1-mu2, mu1-mu2)))**(1./d) * np.identity(d)
        return (w_m, mu_m, cov_m)



    def mixture_distance(self, w1, mu1, cov1, w2, mu2, cov2):
        if (self.distance_method == "Kullback-leiber"):
            return self.kl_diss( w1, mu1, cov1, w2, mu2, cov2)
        elif(self.distance_method == "ISD") :
            return self.isd_diss(w1, mu1, cov1, w2, mu2, cov2)
        elif(self.distance_method == "Jensen-shannon") :
            return self.isomorphic_merge(w1, mu1, cov1, w2, mu2, cov2)


    def kl_diss(self, w1, mu1, cov1, w2, mu2, cov2):
        """
        Computation of the KL-divergence (dissimilarity) upper bound between components 
        [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
        ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
        """
        w_m, mu_m, cov_m = self.merge(w1, mu1, cov1, w2, mu2, cov2)
        return 0.5*((w1+w2)*np.log(det(cov_m)) - w1*np.log(det(cov1)) - w2*np.log(det(cov2)))




    def isd_diss(self,w1, mu1, cov1, w2, mu2, cov2):
        """
        Computes the ISD (Integral Square Difference between components [(w1,mu1,cov1), (w2,mu2,cov2)])
        and its moment preserving merge. Ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
        """
        w_m, mu_m, cov_m = self.merge(w1, mu1, cov1, w2, mu2, cov2)
        # ISD analytical computation between merged component and the pair of gaussians
        Jhr = w1*w_m * normal(mu1, mu_m, cov1+cov_m) + w2*w_m * normal(mu2, mu_m, cov2+cov_m)
        Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * det(2*cov_m)))
        Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * det(2*cov1))) + \
              (w2**2)*(1./np.sqrt((2*np.pi)**2 * det(2*cov2))) + \
              2*w1*w2*normal(mu1, mu2, cov1+cov2)
        return Jhh - 2*Jhr + Jrr

        #Jensen-Shannon divergence KL(p||(p+q)/2) + KL(q||(p+q)/2)
    def jsd_diss(self,w1, mu1, cov1, w2, mu2, cov2):
        """
        Calculates Jensen-Shannon divergence of two gmm's
        :param gmm_p: mixture.GaussianMixture
        :param gmm_q: mixture.GaussianMixture
        :param sample_count: number of monte carlo samples to use
        :return: Jensen-Shannon divergence
        """
        gmm_p = GaussianMixture(n_components=n_components,covariance_type="full")
        gmm_p.weights_ = w1
        gmm_p.covariances_ = cov1
        gmm_p.means_ = mu1
        gmm_p.n_components = 1
        gmm_p.precisions_cholesky_ = _compute_precision_cholesky(cov1, "full")

        gmm_q = GaussianMixture(n_components=n_components,covariance_type="full")
        gmm_q.weights_ = w2
        gmm_q.covariances_ = cov2 
        gmm_q.means_ = mu2
        gmm_q.n_components = 1
        gmm_q.precisions_cholesky_ = _compute_precision_cholesky(cov2, "full")

        X = gmm_p.sample(sample_count)[0]
        log_p_X = gmm_p.score_samples(X)
        log_q_X = gmm_q.score_samples(X)
        log_mix_X = np.logaddexp(log_p_X, log_q_X)

        Y = gmm_q.sample(sample_count)[0]
        log_p_Y = gmm_p.score_samples(Y)
        log_q_Y = gmm_q.score_samples(Y)
        log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

        # black magic?
        return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
                + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

    def update_merge_mapping(self, merge_mapping, nindex, dindex):
        n = len(merge_mapping)
        for i in range(n):
            if merge_mapping[i]==dindex:
                merge_mapping[i] = nindex

    def radius_search(self, nn, mu, max_neigh, merge_mapping, nindex, dindex):
        neigh_arr = nn.radius_neighbors([mu], return_distance=False)[0]
        for i in range(len(neigh_arr)):
            ii = merge_mapping[neigh_arr[i]]
            # avoiding neighbor of itself
            if ii==nindex or ii==dindex:
                neigh_arr[i] = MAXINT
                continue
            neigh_arr[i] = ii
        neigh_arr = np.unique(neigh_arr)
        if len(neigh_arr)>max_neigh:
            neigh_arr = nn.kneighbors([mu], n_neighbors=max_neigh, return_distance=False)[0]
            for i in range(len(neigh_arr)):
                ii = merge_mapping[neigh_arr[i]]
                # avoiding neighbor of itself
                if ii==nindex or ii==dindex:
                    neigh_arr[i] = MAXINT
                    continue
                neigh_arr[i] = ii
            neigh_arr = np.unique(neigh_arr)
        ret = MAXINT*np.ones(max_neigh, dtype=np.int32)
        ret[0:len(neigh_arr)] = neigh_arr
        return ret

    def update_structs(self,nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex):
        """
        Updates the nn_indexes and diss_matrix structs by removing the items
        corresponding to dindex and updating the ones corresponding to nindex
        """
        max_neigh = nn_indexes.shape[1]
        for i in indexes:
            if i==nindex: continue # this is an special case (see below)
            flag1 = False
            flag2 = False
            for j in range(max_neigh):
                jj = nn_indexes[i,j]
                if jj==MAXINT: break
                if jj==nindex: 
                    diss_matrix[i,j] = self.mixture_distance(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
                    flag1 = True
                elif jj==dindex and flag1:
                    nn_indexes[i,j] = MAXINT
                    diss_matrix[i,j] = -1
                    flag2 = True
                elif jj==dindex and not flag1:
                    nn_indexes[i,j] = nindex
                    diss_matrix[i,j] = self.mixture_distance(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
                    flag2 = True
            if flag2:
                sorted_indexes = np.argsort(nn_indexes[i,:])
                nn_indexes[i,:] = (nn_indexes[i,:])[sorted_indexes]
                diss_matrix[i,:] = (diss_matrix[i,:])[sorted_indexes]

        # the special case...
        for j in range(max_neigh):
            jj = nn_indexes[nindex,j]
            if jj!=MAXINT:
                diss_matrix[nindex,j] = self.mixture_distance(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])
            else:
                diss_matrix[nindex,j] = -1

    def _outer(self,x, y):
        """
        Computes the outer production between 1d-ndarrays x and y.
        """
        m = x.shape[0]
        n = y.shape[0]
        res = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                res[i, j] = x[i]*y[j]
        return res

    def compute_lower_bound(self): 
        # check derivation for details on this
        p = -np.sum((self.m**2 + self.s2) / (2 * self.sigma**2))
        next_term = -0.5 * np.add.outer(self.data**2, self.m**2 + self.s2)
        next_term -= np.outer(self.data, self.m)
        next_term *= self.varphi
        p += np.sum(next_term)
        q = np.sum(np.log(self.varphi)) - 0.5 * np.sum(np.log(self.s2))
        elbo = p + q
        return elbo

    def compare_lower_bound(self):
        e1 = np.outer(self.data, self.m)
        e2 = -0.5 * (self.m**2 + self.s2)
        e = e1 + e2[np.newaxis, :]
        self.varphi = np.exp(e) / np.sum(np.exp(e), axis=1)[:, np.newaxis]
        # cavi m update
        self.m = np.sum(self.data[:, np.newaxis] * self.varphi, axis=0)
        self.m /= (1.0 / self.sigma**2 + np.sum(self.varphi, axis=0))
        # cavi s2 update
        self.s2 = 1.0 / (1.0 / self.sigma**2 + np.sum(self.varphi, axis=0))
