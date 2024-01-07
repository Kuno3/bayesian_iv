import numpy as np
from tqdm import tqdm
from scipy.stats import bernoulli

class bayesian_iv():
    def __init__(
        self,
        Z,
        W,
        Y,
        X,
        N_a
    ):
        """
        Z: assigned treatment
        W: received treatment
        Y: outcome
        X: confounder
        N_a: parameter of prior distribution
        """
        self.Z = Z
        self.W = W
        self.Y = Y
        self.X = X
        self.N_a = N_a

        self.N = len(Y)
        self.dim = X.shape[1]

    def G_sampler(self, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t):
        """
        G: compliance type
        0: at
        1: nt
        2: co
        """
        G = np.zeros(self.N)
        for i in range(self.N):
            if (self.Z[i] == 0) & (self.W[i]==1):
                G[i] = 0
            elif (self.Z[i] == 1) & (self.W[i]==0):
                G[i] = 1
            elif (self.Z[i] == 0):
                if self.Y[i] == 1:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_nt = np.exp(self.X[i].dot(gamma_nt)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y1_given_co_c = np.exp(self.X[i].dot(beta_co_c)) / (1 + np.exp(self.X[i].dot(beta_co_c)))
                    prob_y1_given_nt = np.exp(self.X[i].dot(beta_nt)) / (1 + np.exp(self.X[i].dot(beta_nt)))
                    p = prob_co * prob_y1_given_co_c
                    p /= prob_co * prob_y1_given_co_c + prob_nt * prob_y1_given_nt
                    if np.random.rand() < p:
                        G[i] = 2
                    else:
                        G[i] = 1
                else:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_nt = np.exp(self.X[i].dot(gamma_nt)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y0_given_co_c = 1 / (1 + np.exp(self.X[i].dot(beta_co_c)))
                    prob_y0_given_nt = 1 / (1 + np.exp(self.X[i].dot(beta_nt)))
                    p = prob_co * prob_y0_given_co_c
                    p /= prob_co * prob_y0_given_co_c + prob_nt * prob_y0_given_nt
                    if np.random.rand() < p:
                        G[i] = 2
                    else:
                        G[i] = 1
            else:
                if self.Y[i] == 1:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_at = np.exp(self.X[i].dot(gamma_at)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y1_given_co_t = np.exp(self.X[i].dot(beta_co_t)) / (1 + np.exp(self.X[i].dot(beta_co_t)))
                    prob_y1_given_at = np.exp(self.X[i].dot(beta_at)) / (1 + np.exp(self.X[i].dot(beta_at)))
                    p = prob_co * prob_y1_given_co_t
                    p /= prob_co * prob_y1_given_co_t + prob_at * prob_y1_given_at
                    if np.random.rand() < p:
                        G[i] = 2
                    else:
                        G[i] = 0
                else:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_at = np.exp(self.X[i].dot(gamma_at)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y0_given_co_t = 1 / (1 + np.exp(self.X[i].dot(beta_co_t)))
                    prob_y0_given_at = 1 / (1 + np.exp(self.X[i].dot(beta_at)))
                    p = prob_co * prob_y0_given_co_t
                    p /= prob_co * prob_y0_given_co_t + prob_at * prob_y0_given_at
                    if np.random.rand() < p:
                        G[i] = 2
                    else:
                        G[i] = 0
        return G

    def gamma_at_sampler(self, G, gamma_at, gamma_nt):
        if sum(G==0) > 0:
            X_at = self.X[G==0]

            gamma_at_new = gamma_at + self.prop_scale['gamma_at'] * np.random.randn(self.dim)

            log_likelihood_old = X_at.dot(gamma_at).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = X_at.dot(gamma_at_new).sum() - np.log(1 + np.exp(self.X.dot(gamma_at_new)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(gamma_at).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum())
            log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(gamma_at_new).sum() - np.log(1 + np.exp(self.X.dot(gamma_at_new)) + np.exp(self.X.dot(gamma_nt))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return gamma_at_new
            else:
                return gamma_at
        else:
            return gamma_at
    
    def gamma_nt_sampler(self, G, gamma_at, gamma_nt):
        if sum(G==1) > 0:
            X_nt = self.X[G==1]

            gamma_nt_new = gamma_nt + self.prop_scale['gamma_at'] * np.random.randn(self.dim)

            log_likelihood_old = X_nt.dot(gamma_nt).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = X_nt.dot(gamma_nt_new).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt_new))).sum()
            log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(gamma_nt).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum())
            log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(gamma_nt_new).sum() - np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt_new))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return gamma_nt_new
            else:
                return gamma_nt
        else:
            return gamma_nt

    def beta_at_sampler(self, G, beta_at):
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            Y_at = self.Y[G==0]

            beta_at_new = beta_at + self.prop_scale['beta_at'] * np.random.randn(self.dim)

            log_likelihood_old = Y_at.dot(X_at.dot(beta_at)).sum() - np.log(1 + np.exp(X_at.dot(beta_at))).sum()
            log_likelihood_new = Y_at.dot(X_at.dot(beta_at_new)).sum() - np.log(1 + np.exp(X_at.dot(beta_at_new))).sum()
            log_prior_old = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_at))).sum())
            log_prior_new = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at_new).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_at_new))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return beta_at_new
            else:
                return beta_at
        else:
            return beta_at
  
    def beta_nt_sampler(self, G, beta_nt):
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            Y_nt = self.Y[G==1]

            beta_nt_new = beta_nt + self.prop_scale['beta_nt'] * np.random.randn(self.dim)

            log_likelihood_old = Y_nt.dot(X_nt.dot(beta_nt)).sum() - np.log(1 + np.exp(X_nt.dot(beta_nt))).sum()
            log_likelihood_new = Y_nt.dot(X_nt.dot(beta_nt_new)).sum() - np.log(1 + np.exp(X_nt.dot(beta_nt_new))).sum()
            log_prior_old = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_nt))).sum())
            log_prior_new = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt_new).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_nt_new))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return beta_nt_new
            else:
                return beta_nt
        else:
            return beta_nt

  
    def beta_co_c_sampler(self, G, beta_co_c):
        if sum((G==2) & (self.Z==0)) > 0:
            X_co_c = self.X[(G==2) & (self.Z==0)]
            Y_co_c = self.Y[(G==2) & (self.Z==0)]

            beta_co_c_new = beta_co_c + self.prop_scale['beta_co_c'] * np.random.randn(self.dim)

            log_likelihood_old = Y_co_c.dot(X_co_c.dot(beta_co_c)).sum() - np.log(1 + np.exp(X_co_c.dot(beta_co_c))).sum()
            log_likelihood_new = Y_co_c.dot(X_co_c.dot(beta_co_c_new)).sum() - np.log(1 + np.exp(X_co_c.dot(beta_co_c_new))).sum()
            log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_c))).sum())
            log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c_new).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_c_new))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return beta_co_c_new
            else:
                return beta_co_c
        else:
            return beta_co_c

    def beta_co_t_sampler(self, G, beta_co_t):
        if sum((G==2) & (self.Z==1)) > 0:
            X_co_t = self.X[(G==2) & (self.Z==1)]
            Y_co_t = self.Y[(G==2) & (self.Z==1)]

            beta_co_t_new = beta_co_t + self.prop_scale['beta_co_t'] * np.random.randn(self.dim)

            log_likelihood_old = Y_co_t.dot(X_co_t.dot(beta_co_t)).sum() - np.log(1 + np.exp(X_co_t.dot(beta_co_t))).sum()
            log_likelihood_new = Y_co_t.dot(X_co_t.dot(beta_co_t_new)).sum() - np.log(1 + np.exp(X_co_t.dot(beta_co_t_new))).sum()
            log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_t))).sum())
            log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t_new).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_t_new))).sum())
            log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
            if log_acceptance_ratio > np.log(np.random.rand()):
                return beta_co_t_new
            else:
                return beta_co_t
        else:
            return beta_co_t

    def log_posterior(self, G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t):
        lp = 0
        for i in range(self.N):
            if (self.Z[i] == 0) & (self.W[i]==1):
                lp += self.Y[i] * (self.X[i].dot(beta_at)) - np.log(1 + np.exp(self.X[i].dot(beta_at)))
            elif (self.Z[i] == 1) & (self.W[i]==0):
                lp += self.Y[i] * (self.X[i].dot(beta_nt)) - np.log(1 + np.exp(self.X[i].dot(beta_nt)))
            elif (self.Z[i] == 0):
                if self.Y[i] == 1:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_nt = np.exp(self.X[i].dot(gamma_nt)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y1_given_co_c = np.exp(self.X[i].dot(beta_co_c)) / (1 + np.exp(self.X[i].dot(beta_co_c)))
                    prob_y1_given_nt = np.exp(self.X[i].dot(beta_nt)) / (1 + np.exp(self.X[i].dot(beta_nt)))
                    p = prob_co * prob_y1_given_co_c
                    p /= prob_co * prob_y1_given_co_c + prob_nt * prob_y1_given_nt
                    if G[i] == 2:
                        lp += np.log(p)
                        lp += self.Y[i] * (self.X[i].dot(beta_co_c)) - np.log(1 + np.exp(self.X[i].dot(beta_co_c)))
                    else:
                        lp += np.log(1-p)
                        lp += self.Y[i] * (self.X[i].dot(beta_nt)) - np.log(1 + np.exp(self.X[i].dot(beta_nt)))
                else:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_nt = np.exp(self.X[i].dot(gamma_nt)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y0_given_co_c = 1 / (1 + np.exp(self.X[i].dot(beta_co_c)))
                    prob_y0_given_nt = 1 / (1 + np.exp(self.X[i].dot(beta_nt)))
                    p = prob_co * prob_y0_given_co_c
                    p /= prob_co * prob_y0_given_co_c + prob_nt * prob_y0_given_nt
                    if G[i] == 2:
                        lp += np.log(p)
                        lp += self.Y[i] * (self.X[i].dot(beta_co_c)) - np.log(1 + np.exp(self.X[i].dot(beta_co_c)))
                    else:
                        lp += np.log(1-p)
                        lp += self.Y[i] * (self.X[i].dot(beta_nt)) - np.log(1 + np.exp(self.X[i].dot(beta_nt)))
            else:
                if self.Y[i] == 1:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_at = np.exp(self.X[i].dot(gamma_at)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y1_given_co_t = np.exp(self.X[i].dot(beta_co_t)) / (1 + np.exp(self.X[i].dot(beta_co_t)))
                    prob_y1_given_at = np.exp(self.X[i].dot(beta_at)) / (1 + np.exp(self.X[i].dot(beta_at)))
                    p = prob_co * prob_y1_given_co_t
                    p /= prob_co * prob_y1_given_co_t + prob_at * prob_y1_given_at
                    if G[i] == 2:
                        lp += np.log(p)
                        lp += self.Y[i] * (self.X[i].dot(beta_co_t)) - np.log(1 + np.exp(self.X[i].dot(beta_co_t)))
                    else:
                        lp += np.log(1-p)
                        lp += self.Y[i] * (self.X[i].dot(beta_at)) - np.log(1 + np.exp(self.X[i].dot(beta_at)))
                else:
                    prob_co = 1 / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_at = np.exp(self.X[i].dot(gamma_at)) / (1 + np.exp(self.X[i].dot(gamma_at)) + np.exp(self.X[i].dot(gamma_nt)))
                    prob_y0_given_co_t = 1 / (1 + np.exp(self.X[i].dot(beta_co_t)))
                    prob_y0_given_at = 1 / (1 + np.exp(self.X[i].dot(beta_at)))
                    p = prob_co * prob_y0_given_co_t
                    p /= prob_co * prob_y0_given_co_t + prob_at * prob_y0_given_at
                    if G[i] == 2:
                        lp += np.log(p)
                        lp += self.Y[i] * (self.X[i].dot(beta_co_t)) - np.log(1 + np.exp(self.X[i].dot(beta_co_t)))
                    else:
                        lp += np.log(1-p)
                        lp += self.Y[i] * (self.X[i].dot(beta_at)) - np.log(1 + np.exp(self.X[i].dot(beta_at)))
        
        lp += (self.N_a / 12 / self.N) * self.X.dot(gamma_at).sum()
        lp += (self.N_a / 12 / self.N) * self.X.dot(gamma_nt).sum()
        lp += - (self.N_a / 12 / self.N) * np.log(1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
        lp += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_at))).sum())
        lp += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_nt))).sum())
        lp += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_c))).sum())
        lp += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t).sum() - 2 * np.log(1 + np.exp(self.X.dot(beta_co_t))).sum())
        
        return lp

    def sampling(
        self,
        prop_scale,
        num_samples,
        thinning=1,
        burn_in=0,
        gamma_at_init=None,
        gamma_nt_init=None,
        beta_at_init=None,
        beta_nt_init=None,
        beta_co_c_init=None,
        beta_co_t_init=None
    ):
        """
        num_samples: number of samples
        prop_scale: dictionary of standard deviation of proposal distribution
        """

        self.prop_scale = prop_scale

        if gamma_at_init is None:
            gamma_at = np.random.randn(self.dim)
        else:
            gamma_at = gamma_at_init

        if gamma_nt_init is None:
            gamma_nt = np.random.randn(self.dim)
        else:
            gamma_nt = gamma_nt_init
  
        if beta_at_init is None:
            beta_at = np.random.randn(self.dim)
        else:
            beta_at = beta_at_init

        if beta_nt_init is None:
            beta_nt = np.random.randn(self.dim)
        else:
            beta_nt = beta_nt_init

        if beta_co_c_init is None:
            beta_co_c = np.random.randn(self.dim)
        else:
            beta_co_c = beta_co_c_init

        if beta_co_t_init is None:
            beta_co_t = np.random.randn(self.dim)
        else:
            beta_co_t = beta_co_t_init

        self.G_samples = np.zeros((num_samples, self.N))
        self.gamma_at_samples = np.zeros((num_samples, self.dim))
        self.gamma_nt_samples = np.zeros((num_samples, self.dim))
        self.beta_at_samples = np.zeros((num_samples, self.dim))
        self.beta_nt_samples = np.zeros((num_samples, self.dim))
        self.beta_co_c_samples = np.zeros((num_samples, self.dim))
        self.beta_co_t_samples = np.zeros((num_samples, self.dim))
        self.lp_list = np.zeros(num_samples)

        for i in tqdm(range(burn_in+num_samples)):
            for _ in range(thinning):
                G = self.G_sampler(gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)
                gamma_at = self.gamma_at_sampler(G, gamma_at, gamma_nt)
                gamma_nt = self.gamma_nt_sampler(G, gamma_at, gamma_nt)
                beta_at = self.beta_at_sampler(G, beta_at)
                beta_nt = self.beta_nt_sampler(G, beta_nt)
                beta_co_c = self.beta_co_c_sampler(G, beta_co_c)
                beta_co_t = self.beta_co_t_sampler(G, beta_co_t)

            if i >= burn_in:
                self.G_samples[i-burn_in, :] = G
                self.gamma_at_samples[i-burn_in, :] = gamma_at
                self.gamma_nt_samples[i-burn_in, :] = gamma_nt
                self.beta_at_samples[i-burn_in, :] = beta_at
                self.beta_nt_samples[i-burn_in, :] = beta_nt
                self.beta_co_c_samples[i-burn_in, :] = beta_co_c
                self.beta_co_t_samples[i-burn_in, :] = beta_co_t
                self.lp_list[i-burn_in] = self.log_posterior(G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)