from sklearn.mixture import GaussianMixture as GMM
import itertools


import numpy as np
import copy
from scipy import linalg as LA
from numpy import linalg

from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky



class DynamicParameter(object):
    def __init__(self, *args, **conf):
        self.default_params()
        if len(args) > 1:
            raise ValueError
        elif len(args) == 1:
            if isinstance(args[0], float):
                self.values = args[0]
        elif len(conf.keys())>0:
            self.conf['is_constant'] = False
            for key in conf.keys():
                self.conf[key] = conf[key]
            if self.conf['function'] is 'log':
                self.log_evolution()
            elif self.conf['function'] is 'linear':
                self.linear_evolution()
        else:
            self.default_params()

    def log_evolution(self):
        init_val = self.conf['init']
        end_val = self.conf['end']
        steps = self.conf['steps']
        self.values = np.logspace(np.log(init_val), np.log(end_val), num=steps, base=np.exp(1))
        self.idx = -1
        self.conf['max_idx'] = steps - 1

    def linear_evolution(self):
        init_val = self.conf['init']
        end_val = self.conf['end']
        steps = self.conf['steps']
        self.values = np.linspace(init_val, end_val, num=steps)
        self.idx = -1
        self.conf['max_idx'] = steps - 1

    def default_params(self):
        self.conf = {'is_constant':True}
        self.values = 0.05

    def get_value(self):
        if self.conf['is_constant']:
            return copy.copy(self.values)
        else:
            self.idx += 1
            if self.idx >self.conf['max_idx']:
                self.conf['is_constant'] = True
                self.values = copy.copy(self.values[-1])
                return self.get_value()
            return copy.copy(self.values[self.idx])

    def generate_log(self):
        log = 'type: DynamicParameter,'
        for key in self.conf.keys():
            try:
                attr_log = self.conf[key].generate_log()
                log+=(key + ': {')
                log+=(attr_log)
                log+=('}')
                log = log.replace('\n}', '}')
            except IndexError:
                print("INDEX ERROR in DynamicParamater log generation")
            except AttributeError:
                log+=(key + ': ' + str(self.conf[key]) + ',')
        if log[-1] == ',':
            return log[:-1]
        else:
            return log

    def __print__(self):
        return str(self.value())

class IGMM(GMM):
    '''
    classdocs
    '''

    def __init__(self, min_components=3,
                 max_step_components=30,
                 max_components=60,
                 a_split=0.8,
                 forgetting_factor=DynamicParameter(0.05),
                 x_dims=None,
                 y_dims=None):

        GMM.__init__(self, n_components=min_components,
                     covariance_type='full')

        if isinstance(forgetting_factor, float):
            forgetting_factor = DynamicParameter(forgetting_factor)

        self.params = {'init_components': min_components,
                       'max_step_components': max_step_components,
                       'max_components': max_components,
                       'a_split': a_split,
                       'forgetting_factor': forgetting_factor,
                       'x_dims': x_dims,
                       'y_dims': y_dims,
                       'infer_fixed': False}

        if x_dims is not None and y_dims is not None:
            self.params['infer_fixed'] = True

        self.type='IGMM'
        self.initialized=False


    def fit(self,data):
        return self.train(data)

    def train(self, data):
        if self.initialized:
           

            ff_tmp = self.params['forgetting_factor'].get_value()

            self.short_term_model = IGMM(self.params['init_components'])
            self.short_term_model.get_best_gmm(data,lims=[1, self.params['max_step_components']])
                                               # lims=[self.params['init_components'], self.params['max_step_components']])
            weights_st = self.short_term_model.weights_
            weights_st = ff_tmp * weights_st
            self.short_term_model.weights_ = weights_st


            #print(ff_tmp)
            weights_lt = self.weights_
            weights_lt = (self.weights_.sum() - ff_tmp) * weights_lt  # Regularization to keep sum(w)=1.0

            self.weights_ = weights_lt

            gmm_new = copy.deepcopy(self.short_term_model)

            gmm_new = self.merge_similar_gaussians_in_gmm_minim(gmm_new)
            self.mergeGMM(gmm_new)

            self.weights_=self.weights_/sum(self.weights_) #Regularization


        else:
            #self.get_best_gmm(data, lims=[self.params['init_components'], self.params['max_step_components']])
            self.get_best_gmm(data, lims=[self.params['init_components'],self.params['init_components']])
            self.short_term_model = GMM(self.n_components)
            self.initialized = True

        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, "full")
        if self.params['infer_fixed']:
            y_dims = self.params['y_dims']
            x_dims = self.params['x_dims']
            SIGMA_YY_inv = np.zeros((self.n_components,len(y_dims),len(y_dims)))
            SIGMA_XY = np.zeros((self.n_components,len(x_dims),len(y_dims)))

            for k, (Mu, Sigma) in enumerate(zip(self.means_, self.covariances_)):
                Sigma_yy = Sigma[:, y_dims]
                Sigma_yy = Sigma_yy[y_dims, :]

                Sigma_xy = Sigma[x_dims, :]
                Sigma_xy = Sigma_xy[:, y_dims]
                Sigma_yy_inv = linalg.inv(Sigma_yy)

                SIGMA_YY_inv[k,:,:] = Sigma_yy_inv
                SIGMA_XY[k,:, :] = Sigma_xy

            self.SIGMA_YY_inv = SIGMA_YY_inv
            self.SIGMA_XY = SIGMA_XY


    def get_best_gmm(self, data, lims=[1, 10]):
        lowest_bic = np.infty
        bic = []
        aic = []
        # minim = False
        # minim_flag = 2

        n_components_range = range(lims[0], lims[1] + 1, 1)
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM, beware for cases where te model is not found in any case
            gmm = GMM(n_components=n_components,
                          covariance_type='full')
            gmm.fit(data)
            bic.append(gmm.bic(data))
            aic.append(gmm.aic(data))

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = n_components
            try:
                if (bic[-1] > bic[-2] > bic[-3] and
                                bic[-3] < bic[-4] < bic[-5]):
                    best_gmm = n_components - 2
                    break

            except IndexError:
                pass
        # if best_gmm <= 6: # The derivative does not make sense here  #THS MUST BE CHECKED
        best_gmm = np.array(bic).argmin() + lims[0]

        gmm = GMM(n_components=best_gmm,
                      covariance_type='full')
        gmm.fit(data)

        self.weights_ = gmm.weights_
        self.covariances_ = gmm.covariances_ # self.covariances_ = gmm._get_covars()
        self.means_ = gmm.means_
        self.n_components = gmm.n_components

        self.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, "full")

    def return_copy(self):
        '''If any trouble be sure that assignation of
            means and weights is done copying through assignation
        '''
        copy_tmp = GMM(n_components=self.n_components)

        copy_tmp.covariances_ = self.covariances_ #_get_covars()
        copy_tmp.means_ = self.means_
        copy_tmp.weights_ = self.weights_

        return copy_tmp

    def get_bic(self, data):
        return self.bic(data)

    def infer(self, x_dims, y_dims, y, knn=5):
        """
            This method returns the value of x that maximaze the probability P(x|y)
        """
        if self.params['infer_fixed']:
            y_tmp = np.array(y)
            dist = []
            for mu in self.means_:
                dist += [linalg.norm(y_tmp - mu[y_dims])]
            dist = np.array(dist).flatten()
            voters_idx = dist.argsort()[:knn]
            # print(voters_idx[0])
            gmm = self
            # print(len(gmm.means_))
            Mu_tmp = gmm.means_[voters_idx]
            Sigma_tmp = gmm.covariances_[voters_idx]
            Sigma_yy_inv_tmp = self.SIGMA_YY_inv[voters_idx]  # _get_covars()[voters_idx]
            Sigma_xy_tmp = self.SIGMA_XY[voters_idx]

            y = np.mat(y)
            n_dimensions = np.amax(len(x_dims)) + np.amax(len(y_dims)) #secure (x,y)_dims to avoid errors
            likely_x = np.mat(np.zeros((len(x_dims), knn)))
            sm = np.mat(np.zeros((len(x_dims) + len(y_dims), knn)))
            p_xy = np.mat(np.zeros((knn, 1)))

            for k, (Mu, Sigma_yy_inv, Sigma_xy, Sigma) in enumerate(zip(Mu_tmp, Sigma_yy_inv_tmp, Sigma_xy_tmp, Sigma_tmp)):
                Mu = np.transpose(Mu)
                # ----------------------------------------------- Sigma=np.mat(Sigma)
                tmp1 = Sigma_yy_inv * np.transpose(y - Mu[y_dims])
                tmp2 = np.transpose(Sigma_xy * tmp1)
                likely_x[:, k] = np.transpose(Mu[x_dims] + tmp2)

                sm[x_dims, k] = likely_x[:, k].flatten()
                sm[y_dims, k] = y.flatten()

                tmp4 = 1 / (np.sqrt(((2.0 * np.pi) ** n_dimensions) * np.abs(linalg.det(Sigma)))) #It is possible to predifine
                                                                                                  #the determinant too
                tmp5 = np.transpose(sm[:, k]) - (Mu)
                tmp6 = linalg.inv(Sigma)
                tmp7 = np.exp((-1.0 / 2.0) * (tmp5 * tmp6 * np.transpose(tmp5)))  # Multiply time GMM.Priors????
                p_xy[k, :] = np.reshape(tmp4 * tmp7, (1))

            k_ok = np.argmax(p_xy)
            x = likely_x[:, k_ok]

            return np.array(x.transpose())[0]
        else:
            y_tmp = np.array(y)
            dist = []
            for mu in self.means_:
                dist += [linalg.norm(y_tmp - mu[y_dims])]
            dist = np.array(dist).flatten()
            voters_idx = dist.argsort()[:knn]

            gmm = self
            Mu_tmp = gmm.means_[voters_idx]
            Sigma_tmp = gmm.covariances_[voters_idx] #_get_covars()[voters_idx]

            y = np.mat(y)
            n_dimensions = np.amax(len(x_dims)) + np.amax(len(y_dims))
            likely_x = np.mat(np.zeros((len(x_dims), knn)))
            sm = np.mat(np.zeros((len(x_dims) + len(y_dims), knn)))
            p_xy = np.mat(np.zeros((knn, 1)))

            for k, (Mu, Sigma) in enumerate(zip(Mu_tmp, Sigma_tmp)):
                Mu = np.transpose(Mu)
                # ----------------------------------------------- Sigma=np.mat(Sigma)
                Sigma_yy = Sigma[:, y_dims]
                Sigma_yy = Sigma_yy[y_dims, :]

                Sigma_xy = Sigma[x_dims, :]
                Sigma_xy = Sigma_xy[:, y_dims]
                tmp1 = linalg.inv(Sigma_yy) * np.transpose(y - Mu[y_dims])
                tmp2 = np.transpose(Sigma_xy * tmp1)
                likely_x[:, k] = np.transpose(Mu[x_dims] + tmp2)

                sm[x_dims, k] = likely_x[:, k].flatten()
                sm[y_dims, k] = y.flatten()

                tmp4 = 1 / (np.sqrt(((2.0 * np.pi) ** n_dimensions) * np.abs(linalg.det(Sigma))))
                tmp5 = np.transpose(sm[:, k]) - (Mu)
                tmp6 = linalg.inv(Sigma)
                tmp7 = np.exp((-1.0 / 2.0) * (tmp5 * tmp6 * np.transpose(tmp5)))  # Multiply time GMM.Priors????
                p_xy[k, :] = np.reshape(tmp4 * tmp7, (1))

            k_ok = np.argmax(p_xy)
            x = likely_x[:, k_ok]

            return np.array(x.transpose())[0]

    def merge_similar_gaussians_in_gmm_full(self, gmm2):
        # Selecting high related Gaussians to be mixtured
        gmm1 = self
        similarity = get_similarity_matrix(gmm1, gmm2)

        total_similar = np.sum(similarity.flatten())
        similarity_pct = (1 / total_similar) * similarity

        changed_flag = False
        indices_gmm2 = np.arange(similarity_pct.shape[1] - 1, -1, -1)
        for i in np.arange(similarity_pct.shape[0] - 1, -1, -1):
            j_ = 0
            for j in indices_gmm2:
                if similarity_pct[i, j] <= 0.01:
                    gmm2 = self.mergeGMMComponents(gmm2, i, len(
                        indices_gmm2) - j_ - 1)  # len(indices_gmm2)-j_ to take the correspondent gaussian in gmm2
                    indices_gmm2 = np.delete(indices_gmm2, j_, axis=0)
                    j_ = j_ - 1
                j_ = j_ + 1

        if changed_flag:
            return self.merge_similar_gaussians_in_gmm_full(gmm2)
        else:
            return gmm2

    def merge_similar_gaussians_in_gmm_smart(self, gmm2):
        # Selecting high related Gaussians to be mixtured
        gmm1 = self
        similarity = get_similarity_matrix(gmm1, gmm2)

        total_similar = np.sum(similarity.flatten())
        similarity_pct = (1 / total_similar) * similarity

        indices = np.unravel_index(similarity_pct.argmin(), similarity.shape)

        changed_flag = False
        if similarity_pct[indices[0], indices[1]] <= 0.01:
            gmm2 = self.mergeGMMComponents(gmm2, indices[0], indices[
                1])  # len(indices_gmm2)-j_ to take the correspondent gaussian in gmm2
            changed_flag = True

        if changed_flag:
            return self.merge_similar_gaussians_in_gmm_smart(gmm2)
        else:
            return gmm2

    def merge_similar_gaussians_in_gmm_minim(self, gmm2):
        # Selecting high related Gaussians to be merged
        gmm1 = self
        similarity = get_similarity_matrix(gmm1, gmm2)
        indices = np.unravel_index(similarity.argmin(), similarity.shape)
        gmm2 = self.mergeGMMComponents(gmm2, indices[0],
                                       indices[1])  # len(indices_gmm2)-j_ to take the correspondent gaussian in gmm2

        if self.n_components + gmm2.n_components > self.params['max_components']:
            return self.merge_similar_gaussians_in_gmm_minim(gmm2)
        else:
            return gmm2

    def returnCopy(self):
        copy_tmp = IGMM(min_components=self.params['init_components'],
                         max_step_components=self.params['max_step_components'],
                         max_components=self.params['max_components'],
                         a_split=self.params['a_split'],
                         forgetting_factor=self.params['forgetting_factor'])

        copy_tmp.covariances_ = self.covariances_ #_get_covars()
        copy_tmp.means_ = self.means_
        copy_tmp.weights_ = self.weights_

        copy_tmp.short_term_model = copy.deepcopy(self.short_term_model)
        return copy_tmp

    def mergeGMM(self, gmm2):
        covariances_ = self.covariances_ #_get_covars()
        means_ = self.means_
        weights_ = self.weights_

        covariances_2 = gmm2.covariances_ #_get_covars()
        means_2 = gmm2.means_
        weights_2 = gmm2.weights_

        new_components = weights_2.shape[0]
        for i in range(new_components):
            covariances_ = np.insert(covariances_, [-1], covariances_2[i], axis=0)
            means_ = np.insert(means_, [-1], means_2[i], axis=0)
            weights_ = np.insert(weights_, [-1], weights_2[i], axis=0)

        self.covariances_ = covariances_
        self.means_ = means_
        self.weights_ = weights_
        self.n_components = self.n_components + new_components

    def mergeGMMComponents(self, gmm2, index1, index2):
        gauss1 = {'covariance': self.covariances_[index1], #_get_covars()[index1],
                  'mean': self.means_[index1],
                  'weight': self.weights_[index1]}
        gauss2 = {'covariance': gmm2.covariances_[index2], #_get_covars()[index2],
                  'mean': gmm2.means_[index2],
                  'weight': gmm2.weights_[index2]}
        gauss = merge_gaussians(gauss1, gauss2)

        covariances_1 = self.covariances_ #_get_covars()
        means_1 = self.means_
        weights_1 = self.weights_

        covariances_1[index1] = gauss['covariance']
        means_1[index1] = gauss['mean']
        weights_1[index1] = gauss['weight']

        covariances_2 = gmm2.covariances_ #_get_covars()
        means_2 = gmm2.means_
        weights_2 = gmm2.weights_

        covariances_2 = np.delete(covariances_2, index2, 0)
        means_2 = np.delete(means_2, index2, 0)
        weights_2 = np.delete(weights_2, index2, 0)

        self.covariances_ = covariances_1
        self.means_ = means_1
        self.weights_ = weights_1

        gmm2.covariances_ = covariances_2
        gmm2.means_ = means_2
        gmm2.weights_ = weights_2
        gmm2.n_components = gmm2.n_components - 1

        return gmm2

    def mergeGaussians(self, index1, index2):
        gauss1 = {'covariance': self.covariances_[index1], #_get_covars()[index1],
                  'mean': self.means_[index1],
                  'weight': self.weights_[index1]}
        gauss2 = {'covariance': self.covariances_[index2], #_get_covars()[index2],
                  'mean': self.means_[index2],
                  'weight': self.weights_[index2]}
        gauss = merge_gaussians(gauss1, gauss2)

        covariances_ = self.covariances_ #_get_covars()
        means_ = self.means_
        weights_ = self.weights_

        covariances_[index1] = gauss['covariance']
        means_[index1] = gauss['mean']
        weights_[index1] = gauss['weight']

        covariances_ = np.delete(covariances_, index2, 0)
        means_ = np.delete(means_, index2, 0)
        weights_ = np.delete(weights_, index2, 0)

        self.covariances_ = covariances_
        self.means_ = means_
        self.weights_ = weights_
        self.n_components = self.n_components - 1

    def splitGaussian(self, index):
        gauss = {'covariance': self.covariances_[index], #_get_covars()[index],
                 'mean': self.means_[index],
                 'weight': self.weights_[index]}

        gauss1, gauss2 = split_gaussian(gauss, self.params['a_split'])

        covariances_ = self.covariances_ #_get_covars()
        means_ = self.means_
        weights_ = self.weights_

        covariances_[index] = gauss1['covariance']
        means_[index] = gauss1['mean']
        weights_[index] = gauss1['weight']

        covariances_ = np.insert(covariances_, [-1], gauss2['covariance'], axis=0)
        means_ = np.insert(means_, [-1], gauss2['mean'], axis=0)
        weights_ = np.insert(weights_, [-1], gauss2['weight'], axis=0)

        self.covariances_ = covariances_
        self.means_ = means_
        self.weights_ = weights_
        self.n_components = self.n_components + 1

   
        
    def getGaussians(self):
        return (self.means_, self.covariances_, self.weights_)

    def save(self,file_prefix=''):
        save_gmm(self, file_prefix=file_prefix)

def load_gmm(file_prefix):
    file_weights = file_prefix + 'GMM_weights.txt'
    file_covariances = file_prefix + 'GMM_covariances.txt'
    file_means = file_prefix + 'GMM_means.txt'
    Weights = np.loadtxt(file_weights)
    raw_Means = np.loadtxt(file_means)
    raw_Sigma = np.loadtxt(file_covariances)
    n_components = len(Weights)
    n_dims =  raw_Means.shape[0]
    Means = []
    Sigma = []
    for i in range(n_components):
        Means.append(raw_Means[:, i])
        Sigma.append(raw_Sigma[:, i * n_dims:(i + 1) * n_dims])

    gmm = IGMM()
    gmm.weights_ = Weights
    gmm.means_ = np.array(Means)
    gmm.covariances_ = np.array(Sigma)
    gmm.n_components = n_components

    return gmm

def save_gmm(gmm, file_prefix=''):
    file_weights = file_prefix + 'GMM_weights.txt'
    file_covariances = file_prefix + 'GMM_covariances.txt'
    file_means = file_prefix + 'GMM_means.txt'
    n_components = gmm.n_components
    n_dims = len(gmm.means_[0])

    raw_Means = np.zeros((n_dims, n_components))
    raw_Covariances = np.zeros((n_dims, n_dims * n_components))
    raw_Weights = np.zeros((1, n_components))

    for k, (Weights, Means, Covariances) in enumerate(zip(gmm.weights_, gmm.means_, gmm.covariances_)):
        raw_Means[:, k] = Means
        raw_Covariances[:, k * n_dims:(k + 1) * n_dims] = Covariances
        raw_Weights[0, k] = Weights
    np.savetxt(file_weights, raw_Weights, fmt='%f')
    np.savetxt(file_means, raw_Means, fmt='%f')
    np.savetxt(file_covariances, raw_Covariances, fmt='%f')

def get_KL_divergence(gauss1, gauss2):
    try:
        detC1 = LA.det(gauss1['covariance'])
    except ValueError:
        x = raw_input("Broken")
        pass

    detC2 = LA.det(gauss2['covariance'])
    logC2C1 = np.log(detC2 / detC1)

    invC2 = LA.inv(gauss2['covariance'])
    traceinvC2C1 = np.trace(np.dot(invC2, gauss1['covariance']))

    m2m1 = np.matrix(gauss2['mean'] - gauss1['mean'])
    invC1 = LA.inv(gauss1['covariance'])
    mTC1m = (m2m1) * invC1 * np.transpose(m2m1)

    D = np.shape(gauss1['covariance'])[0]

    return logC2C1 + traceinvC2C1 + mTC1m - D


def get_similarity_estimation(gauss1, gauss2):
    return (1.0 / 2.0) * (get_KL_divergence(gauss1, gauss2) + get_KL_divergence(gauss2, gauss1))


def get_similarity_matrix(gmm1, gmm2):
    n_comp_1 = gmm1.n_components
    n_comp_2 = gmm2.n_components

    similarity_matrix = np.zeros((n_comp_1, n_comp_2))
    for i, (Mu, Sigma) in enumerate(zip(gmm1.means_, gmm1.covariances_)):
        gauss1 = {'covariance': Sigma, 'mean': Mu}
        for j, (Mu2, Sigma2) in enumerate(zip(gmm2.means_, gmm2.covariances_)):
            gauss2 = {'covariance': Sigma2, 'mean': Mu2}
            similarity_matrix[i, j] = get_similarity_estimation(gauss1, gauss2)

    return (similarity_matrix)


def merge_gaussians(gauss1, gauss2):
    weight1 = gauss1['weight']
    covar1 = gauss1['covariance']
    mean1 = gauss1['mean']

    weight2 = gauss2['weight']
    covar2 = gauss2['covariance']
    mean2 = gauss2['mean']

    weight = weight1 + weight2

    f1 = weight1 / weight
    f2 = weight2 / weight

    mean = f1 * mean1 + f2 * mean2

    m1m2 = np.matrix(mean1 - mean2)
    covar = f1 * covar1 + f2 * covar2 + f1 * f2 * np.transpose(m1m2) * m1m2

    return {'covariance': covar, 'mean': mean, 'weight': weight}


def split_gaussian(gauss, a):  # Supervise that all the values here are real
    weight = gauss['weight']
    covar = gauss['covariance']
    mean = gauss['mean']

    w, v_ = LA.eig(covar)
    w = np.real(w) #This was previosly not needed... why???
    max_eig_index = np.argmax(w)

    l = w[max_eig_index]
    v = v_[:, max_eig_index]

    Delta_v = np.matrix(np.sqrt(a * l) * v)
    weight = weight / 2.0
    mean1 = np.matrix(mean + Delta_v)
    mean2 = np.matrix(mean - Delta_v)
    covar -= np.transpose(Delta_v) * Delta_v

    return {'covariance': covar, 'mean': mean1, 'weight': weight}, {'covariance': covar, 'mean': mean2,
                                                                    'weight': weight}
