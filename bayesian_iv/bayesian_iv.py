import numpy as np
from tqdm import tqdm

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
        
        # pattern for (Z_i = 0) & (W_i = 1)
        G[(self.Z == 0) & (self.W == 1)] = 0

        # pattern for (Z_i = 1) & (W_i = 0)
        G[(self.Z == 1) & (self.W == 0)] = 1

        # pattern for (Z_i = 0) & (W_i = 0)
        X_z0_w0 = self.X[(self.Z == 0) & (self.W == 0)]
        Y_z0_w0 = self.Y[(self.Z == 0) & (self.W == 0)]
        N_z0_w0 = len(Y_z0_w0)
        # log p(G_i = co | Y, \theta)
        log_prob_co = - np.log1p(np.exp(X_z0_w0.dot(gamma_at)) + np.exp(X_z0_w0.dot(gamma_nt)))
        # log p(G_i = nt | Y, \theta)
        log_prob_nt = X_z0_w0.dot(gamma_nt) - np.log1p(np.exp(X_z0_w0.dot(gamma_at)) + np.exp(X_z0_w0.dot(gamma_nt)))
        # log p(Y = 1, | G_i = co, \theta)
        log_prob_y1_given_co_c = X_z0_w0.dot(beta_co_c) - np.log1p(np.exp(X_z0_w0.dot(beta_co_c)))
        # log p(Y = 1, | G_i = nt, \theta)
        log_prob_y1_given_nt = X_z0_w0.dot(beta_nt) - np.log1p(np.exp(X_z0_w0.dot(beta_nt)))
        # log p(Y = 0, | G_i = co, \theta)
        log_prob_y0_given_co_c = - np.log1p(np.exp(X_z0_w0.dot(beta_co_c)))
        # log p(Y = 0, | G_i = nt, \theta)
        log_prob_y0_given_nt = - np.log1p(np.exp(X_z0_w0.dot(beta_nt)))
        # log p(G_i = co | Y_i, \theta)
        logp = Y_z0_w0 * (log_prob_co + log_prob_y1_given_co_c - np.log(np.exp(log_prob_co + log_prob_y1_given_co_c) + np.exp(log_prob_nt + log_prob_y1_given_nt)))
        logp += (1-Y_z0_w0) * (log_prob_co + log_prob_y0_given_co_c - np.log(np.exp(log_prob_co + log_prob_y0_given_co_c) + np.exp(log_prob_nt + log_prob_y0_given_nt)))
        # Deciding co or nt
        threshold = np.log(np.random.rand(N_z0_w0))
        G_z0_w0 = np.ones(N_z0_w0)
        G_z0_w0[threshold < logp] = 2
        G[(self.Z == 0) & (self.W == 0)] = G_z0_w0

        # pattern for (Z_i = 1) & (W_i = 1)
        X_z1_w1 = self.X[(self.Z == 1) & (self.W == 1)]
        Y_z1_w1 = self.Y[(self.Z == 1) & (self.W == 1)]
        N_z1_w1 = len(Y_z1_w1)
        # log p(G_i = co | Y, \theta)
        log_prob_co = - np.log1p(np.exp(X_z1_w1.dot(gamma_at)) + np.exp(X_z1_w1.dot(gamma_nt)))
        # log p(G_i = at | Y, \theta)
        log_prob_at = X_z1_w1.dot(gamma_at) - np.log1p(np.exp(X_z1_w1.dot(gamma_at)) + np.exp(X_z1_w1.dot(gamma_nt)))
        # log p(Y = 1, | G_i = co, \theta)
        log_prob_y1_given_co_t = X_z1_w1.dot(beta_co_t) - np.log1p(np.exp(X_z1_w1.dot(beta_co_t)))
        # log p(Y = 1, | G_i = at, \theta)
        log_prob_y1_given_at = X_z1_w1.dot(beta_at) - np.log1p(np.exp(X_z1_w1.dot(beta_at)))
        # log p(Y = 0, | G_i = co, \theta)
        log_prob_y0_given_co_t = - np.log1p(np.exp(X_z1_w1.dot(beta_co_t)))
        # log p(Y = 0, | G_i = at, \theta)
        log_prob_y0_given_at = - np.log1p(np.exp(X_z1_w1.dot(beta_at)))
        # log p(G_i = co | Y_i, \theta)
        logp = Y_z1_w1 * (log_prob_co + log_prob_y1_given_co_t - np.log(np.exp(log_prob_co + log_prob_y1_given_co_t) + np.exp(log_prob_at + log_prob_y1_given_at)))
        logp += (1-Y_z1_w1) * (log_prob_co + log_prob_y0_given_co_t - np.log(np.exp(log_prob_co + log_prob_y0_given_co_t) + np.exp(log_prob_at + log_prob_y0_given_at)))
        # deciding co or at
        threshold = np.log(np.random.rand(N_z1_w1))
        G_z1_w1 = np.zeros(N_z1_w1)
        G_z1_w1[threshold < logp] = 2
        G[(self.Z == 1) & (self.W == 1)] = G_z1_w1

        return G

    def gamma_at_sampler(self, G, gamma_at, gamma_nt):
        # proposing gamma_at_new
        gamma_at_new = gamma_at + self.prop_scale['gamma_at'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            log_likelihood_old = X_at.dot(gamma_at).sum() - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = X_at.dot(gamma_at_new).sum() - np.log1p(np.exp(self.X.dot(gamma_at_new)) + np.exp(self.X.dot(gamma_nt))).sum()
        else:
            log_likelihood_old = - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = - np.log1p(np.exp(self.X.dot(gamma_at_new)) + np.exp(self.X.dot(gamma_nt))).sum()

        # calculating log prior
        log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(gamma_at).sum() - 3 * np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum())
        log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(gamma_at_new).sum() - 3 * np.log1p(np.exp(self.X.dot(gamma_at_new)) + np.exp(self.X.dot(gamma_nt))).sum())

        # deciding accept or not     
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return gamma_at_new
        else:
            return gamma_at
    
    def gamma_nt_sampler(self, G, gamma_at, gamma_nt):
        # proposing gamma_nt_new
        gamma_nt_new = gamma_nt + self.prop_scale['gamma_nt'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            log_likelihood_old = X_nt.dot(gamma_nt).sum() - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = X_nt.dot(gamma_nt_new).sum() - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt_new))).sum()
        else:
            log_likelihood_old = - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
            log_likelihood_new = - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt_new))).sum()

        # calculating log prior
        log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(gamma_nt).sum() - 3 * np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum())
        log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(gamma_nt_new).sum() - 3 * np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt_new))).sum())

        # deciding accept or not
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return gamma_nt_new
        else:
            return gamma_nt

    def beta_at_sampler(self, G, beta_at):
        # proposing beta_at_new
        beta_at_new = beta_at + self.prop_scale['beta_at'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            Y_at = self.Y[G==0]
            log_likelihood_old = Y_at.dot(X_at.dot(beta_at)).sum() - np.log1p(np.exp(X_at.dot(beta_at))).sum()
            log_likelihood_new = Y_at.dot(X_at.dot(beta_at_new)).sum() - np.log1p(np.exp(X_at.dot(beta_at_new))).sum()
        else:
            log_likelihood_old = - np.log1p(np.exp(X_at.dot(beta_at))).sum()
            log_likelihood_new = - np.log1p(np.exp(X_at.dot(beta_at_new))).sum()

        # calculating log prior
        log_prior_old = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_at))).sum())
        log_prior_new = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at_new).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_at_new))).sum())

        # deciding accept or not
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return beta_at_new
        else:
            return beta_at
  
    def beta_nt_sampler(self, G, beta_nt):
        # proposing beta_nt_new
        beta_nt_new = beta_nt + self.prop_scale['beta_nt'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            Y_nt = self.Y[G==1]
            log_likelihood_old = Y_nt.dot(X_nt.dot(beta_nt)).sum() - np.log1p(np.exp(X_nt.dot(beta_nt))).sum()
            log_likelihood_new = Y_nt.dot(X_nt.dot(beta_nt_new)).sum() - np.log1p(np.exp(X_nt.dot(beta_nt_new))).sum()
        else:
            log_likelihood_old = 0
            log_likelihood_new = 0

        # calculating log prior
        log_prior_old = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_nt))).sum())
        log_prior_new = (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt_new).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_nt_new))).sum())
        
        # deciding accept or not
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return beta_nt_new
        else:
            return beta_nt

    def beta_co_c_sampler(self, G, beta_co_c):
        # proposing beta_co_c_new
        beta_co_c_new = beta_co_c + self.prop_scale['beta_co_c'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum((G==2) & (self.Z==0)) > 0:
            X_co_c = self.X[(G==2) & (self.Z==0)]
            Y_co_c = self.Y[(G==2) & (self.Z==0)]
            log_likelihood_old = Y_co_c.dot(X_co_c.dot(beta_co_c)).sum() - np.log1p(np.exp(X_co_c.dot(beta_co_c))).sum()
            log_likelihood_new = Y_co_c.dot(X_co_c.dot(beta_co_c_new)).sum() - np.log1p(np.exp(X_co_c.dot(beta_co_c_new))).sum()
        else:
            log_likelihood_old = 0
            log_likelihood_new = 0

        # calculating log prior
        log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_c))).sum())
        log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c_new).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_c_new))).sum())
        
        # deciding accept or not
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return beta_co_c_new
        else:
            return beta_co_c

    def beta_co_t_sampler(self, G, beta_co_t):
        # proposing beta_co_c_new
        beta_co_t_new = beta_co_t + self.prop_scale['beta_co_t'] * np.random.randn(self.dim)

        # calculating log likelihood
        if sum((G==2) & (self.Z==1)) > 0:
            X_co_t = self.X[(G==2) & (self.Z==1)]
            Y_co_t = self.Y[(G==2) & (self.Z==1)]
            log_likelihood_old = Y_co_t.dot(X_co_t.dot(beta_co_t)).sum() - np.log1p(np.exp(X_co_t.dot(beta_co_t))).sum()
            log_likelihood_new = Y_co_t.dot(X_co_t.dot(beta_co_t_new)).sum() - np.log1p(np.exp(X_co_t.dot(beta_co_t_new))).sum()
        else:
            log_likelihood_old = 0
            log_likelihood_new = 0

        # calculating log prior
        log_prior_old = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_t))).sum())
        log_prior_new = (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t_new).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_t_new))).sum())
        
        # deciding accept or not
        log_acceptance_ratio = log_likelihood_new + log_prior_new - log_likelihood_old - log_prior_old
        if log_acceptance_ratio > np.log(np.random.rand()):
            return beta_co_t_new
        else:
            return beta_co_t

    def log_posterior(self, G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t):
        lp = 0

        # log p(G|\theta)
        prob_co = 1 / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt)))
        prob_nt = np.exp(self.X.dot(gamma_nt)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt)))
        prob_at = np.exp(self.X.dot(gamma_at)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt)))

        lp += np.log(prob_at[G == 0]).sum()
        lp += np.log(prob_nt[G == 1]).sum()
        lp += np.log(prob_co[G == 2]).sum()

        # log p(Y|G, \theta)
        # pattern for at
        X_at = self.X[G == 0]
        Y_at = self.Y[G == 0]
        lp += (Y_at * (X_at.dot(beta_at)) - np.log1p(np.exp(X_at.dot(beta_at)))).sum()

        # pattern for nt
        X_nt = self.X[G == 1]
        Y_nt = self.Y[G == 1]
        lp += (Y_nt * (X_nt.dot(beta_nt)) - np.log1p(np.exp(X_nt.dot(beta_nt)))).sum()

        # pattern for co_c
        X_co_c = self.X[(G == 2) & (self.Z == 0)]
        Y_co_c = self.Y[(G == 2) & (self.Z == 0)]
        lp += (Y_co_c * (X_co_c.dot(beta_co_c)) - np.log1p(np.exp(X_co_c.dot(beta_co_c)))).sum()

        # pattern for co_t
        X_co_t = self.X[(G == 2) & (self.Z == 1)]
        Y_co_t = self.Y[(G == 2) & (self.Z == 1)]
        lp += (Y_co_t * (X_co_t.dot(beta_co_t)) - np.log1p(np.exp(X_co_t.dot(beta_co_t)))).sum()

        # log p(\theta)
        lp += (self.N_a / 12 / self.N) * self.X.dot(gamma_at).sum()
        lp += (self.N_a / 12 / self.N) * self.X.dot(gamma_nt).sum()
        lp += - 3 * (self.N_a / 12 / self.N) * np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
        lp += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_at))).sum())
        lp += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_nt))).sum())
        lp += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_c))).sum())
        lp += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_t))).sum())
        
        return lp

    def W_obs_and_mis_sampler(self, G):
        W_obs_and_mis = np.zeros((self.N, 2))
        
        W_obs_and_mis[G==0, 0] = 1
        W_obs_and_mis[G==0, 1] = 1
        
        W_obs_and_mis[G==1, 0] = 0
        W_obs_and_mis[G==1, 1] = 0

        W_obs_and_mis[G==2, 0] = 0
        W_obs_and_mis[G==2, 1] = 1
        
        return W_obs_and_mis

    def Y_obs_and_mis_sampler(self, G, beta_co_c, beta_co_t):
        Y_obs_and_mis = np.full((self.N, 2), np.nan)
        
        # pattern for at
        Y_obs_and_mis[G==0, 1] = self.Y[G==0]
        # pattern for nt
        Y_obs_and_mis[G==1, 0] = self.Y[G==1]
        
        N_co_c = int(sum((G==2) & (self.Z==0)))
        N_co_t = int(sum((G==2) & (self.Z==1)))

        # pattern for co, c
        if N_co_c > 0:
            X_co_c = self.X[(G==2) & (self.Z==0)]
            log_prob_y1_given_co_c = X_co_c.dot(beta_co_t) - np.log1p(np.exp(X_co_c.dot(beta_co_t)))
            Y_obs_and_mis[(G==2) & (self.Z==0), 0] = self.Y[(G==2) & (self.Z==0)]
            Y_obs_and_mis[(G==2) & (self.Z==0), 1] = (np.log(np.random.rand(N_co_c)) < log_prob_y1_given_co_c).astype(float)

        # pattern for co, t
        if N_co_t > 0:
            X_co_t = self.X[(G==2) & (self.Z==1)]
            log_prob_y1_given_co_t = X_co_t.dot(beta_co_c) - np.log1p(np.exp(X_co_t.dot(beta_co_c)))
            Y_obs_and_mis[(G==2) & (self.Z==1), 0] = (np.log(np.random.rand(N_co_t)) < log_prob_y1_given_co_t).astype(float)
            Y_obs_and_mis[(G==2) & (self.Z==1), 1] = self.Y[(G==2) & (self.Z==1)]

        return Y_obs_and_mis

    def tau_late_sampler(self, Y_mis_and_obs, G):
        N_co = int(sum(G==2))

        if N_co > 0:
            tau_late = (Y_mis_and_obs[(G==2), 1] - Y_mis_and_obs[(G==2), 0]).sum() / N_co
            return tau_late
        else:
            return np.nan

    def sampling(
        self,
        num_samples,
        prop_scale,
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

        self.W_obs_and_mis_samples = np.array(
            [self.W_obs_and_mis_sampler(
                self.G_samples[i, :]
            ) for i in range(num_samples)]
        )

        self.Y_obs_and_mis_samples = np.array(
            [self.Y_obs_and_mis_sampler(
                self.G_samples[i, :],
                self.beta_co_c_samples[i, :],
                self.beta_co_t_samples[i, :]
            ) for i in range(num_samples)]
        )

        self.tau_late_samples = np.array(
            [self.tau_late_sampler(
                self.Y_obs_and_mis_samples[i, :],
                self.G_samples[i, :]
            ) for i in range(num_samples)]
        )

    # Functions for HMC

    def gamma_sampler(self, G, gamma_at, gamma_nt, step_size, num_step):
        r_init = np.random.randn(2 * self.dim)
        r = r_init - step_size / 2 * np.concatenate([
            self.d_nlp_d_gamma_at(G, gamma_at, gamma_nt),
            self.d_nlp_d_gamma_nt(G, gamma_at, gamma_nt)
        ])
        gamma_at_new = gamma_at + step_size * r[:self.dim]
        gamma_nt_new = gamma_nt + step_size * r[self.dim:2*self.dim]
        
        for _ in range(num_step-1):
            r += - step_size * np.concatenate([
                self.d_nlp_d_gamma_at(G, gamma_at_new, gamma_nt_new),
                self.d_nlp_d_gamma_nt(G, gamma_at_new, gamma_nt_new)
            ])
            gamma_at_new += step_size * r[:self.dim]
            gamma_nt_new += step_size * r[self.dim:2*self.dim]

        r += - step_size / 2 * np.concatenate([
            self.d_nlp_d_gamma_at(G, gamma_at_new, gamma_nt_new),
            self.d_nlp_d_gamma_nt(G, gamma_at_new, gamma_nt_new)
        ])

        # deciding accept gamma_at and gamma_nt or not
        lp_old = self.log_posterior_gamma(G, gamma_at, gamma_nt)
        lp_new = self.log_posterior_gamma(G, gamma_at_new, gamma_nt_new)
        threshold = np.log(np.random.rand())
        if threshold < lp_new - sum(r[:2*self.dim]**2) / 2 - lp_old + sum(r_init[:2*self.dim]**2) / 2:
            gamma_at_prop = gamma_at_new
            gamma_nt_prop = gamma_nt_new
        else:
            gamma_at_prop = gamma_at
            gamma_nt_prop = gamma_nt
        
        return gamma_at_prop, gamma_nt_prop

    def beta_sampler(self, G, beta_at, beta_nt, beta_co_c, beta_co_t, step_size, num_step):
        r_init = np.random.randn(4 * self.dim)
        r = r_init - step_size / 2 * np.concatenate([
            self.d_nlp_d_beta_at(G, beta_at),
            self.d_nlp_d_beta_nt(G, beta_nt),
            self.d_nlp_d_beta_co_c(G, beta_co_c),
            self.d_nlp_d_beta_co_t(G, beta_co_t)
        ])
        beta_at_new = beta_at + step_size * r[:self.dim]
        beta_nt_new = beta_nt + step_size * r[self.dim:2*self.dim]
        beta_co_c_new = beta_co_c + step_size * r[2*self.dim:3*self.dim]
        beta_co_t_new = beta_co_t + step_size * r[3*self.dim:]
        
        for _ in range(num_step-1):
            r += - step_size * np.concatenate([
                self.d_nlp_d_beta_at(G, beta_at_new),
                self.d_nlp_d_beta_nt(G, beta_nt_new),
                self.d_nlp_d_beta_co_c(G, beta_co_c_new),
                self.d_nlp_d_beta_co_t(G, beta_co_t_new)
            ])
            beta_at_new += step_size * r[:self.dim]
            beta_nt_new += step_size * r[self.dim:2*self.dim]
            beta_co_c_new += step_size * r[2*self.dim:3*self.dim]
            beta_co_t_new += step_size * r[3*self.dim:]

        r += - step_size / 2 * np.concatenate([
            self.d_nlp_d_beta_at(G, beta_at_new),
            self.d_nlp_d_beta_nt(G, beta_nt_new),
            self.d_nlp_d_beta_co_c(G, beta_co_c_new),
            self.d_nlp_d_beta_co_t(G, beta_co_t_new)
        ])

        lp_old = self.log_posterior_beta_at(G, beta_at)
        lp_new = self.log_posterior_beta_at(G, beta_at_new)
        threshold = np.log(np.random.rand())
        if threshold < lp_new - sum(r[:self.dim]**2) / 2 - lp_old + sum(r_init[:self.dim]**2) / 2:
            beta_at_prop = beta_at_new
        else:
            beta_at_prop = beta_at
            
        lp_old = self.log_posterior_beta_nt(G, beta_nt)
        lp_new = self.log_posterior_beta_nt(G, beta_nt_new)
        threshold = np.log(np.random.rand())
        if threshold < lp_new - sum(r[self.dim:2*self.dim]**2) / 2 - lp_old + sum(r_init[self.dim:2*self.dim]**2) / 2:
            beta_nt_prop = beta_nt_new
        else:
            beta_nt_prop = beta_nt

        lp_old = self.log_posterior_beta_co_c(G, beta_co_c)
        lp_new = self.log_posterior_beta_co_c(G, beta_co_c_new)
        threshold = np.log(np.random.rand())
        if threshold < lp_new - sum(r[2*self.dim:3*self.dim]**2) / 2 - lp_old + sum(r_init[2*self.dim:3*self.dim]**2) / 2:
            beta_co_c_prop = beta_co_c_new
        else:
            beta_co_c_prop = beta_co_c

        lp_old = self.log_posterior_beta_co_t(G, beta_co_t)
        lp_new = self.log_posterior_beta_co_t(G, beta_co_t_new)
        threshold = np.log(np.random.rand())
        if threshold < lp_new - sum(r[3*self.dim:]**2) / 2 - lp_old + sum(r_init[3*self.dim:]**2) / 2:
            beta_co_t_prop = beta_co_t_new
        else:
            beta_co_t_prop = beta_co_t

        return beta_at_prop, beta_nt_prop, beta_co_c_prop, beta_co_t_prop

    def d_nlp_d_gamma_at(self, G, gamma_at, gamma_nt):
        d_log_prior = (self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 3 * (self.X * (np.exp(self.X.dot(gamma_at)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0))
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            d_log_likelihood = X_at.sum(axis=0) - (self.X * (np.exp(self.X.dot(gamma_at)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0)
        else:
            d_log_likelihood = - (self.X * (np.exp(self.X.dot(gamma_at)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0)
        return - d_log_likelihood - d_log_prior
    
    def d_nlp_d_gamma_nt(self, G, gamma_at, gamma_nt):
        d_log_prior = (self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 3 * (self.X * (np.exp(self.X.dot(gamma_nt)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0))
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            d_log_likelihood = X_nt.sum(axis=0) - (self.X * (np.exp(self.X.dot(gamma_nt)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0)
        else:
            d_log_likelihood = - (self.X * (np.exp(self.X.dot(gamma_nt)) / (1 + np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))))[:, np.newaxis]).sum(axis=0)
        return - d_log_likelihood - d_log_prior
    
    def log_posterior_gamma(self, G, gamma_at, gamma_nt):
        log_posterior = 0
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            log_posterior += X_at.dot(gamma_at).sum()
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            log_posterior += X_nt.dot(gamma_nt).sum()
        log_posterior += - np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum()
        log_posterior += (self.N_a / 12 / self.N) * (self.X.dot(gamma_at).sum())
        log_posterior += (self.N_a / 12 / self.N) * (self.X.dot(gamma_nt).sum())
        log_posterior += (self.N_a / 12 / self.N) * (- 3 * np.log1p(np.exp(self.X.dot(gamma_at)) + np.exp(self.X.dot(gamma_nt))).sum())
        return log_posterior

    def d_nlp_d_beta_at(self, G, beta_at):
        d_log_prior = (2 * self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 2 * (self.X * (np.exp(self.X.dot(beta_at)) / (1 + np.exp(self.X.dot(beta_at))))[:, np.newaxis]).sum(axis=0))
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            Y_at = self.Y[G==0]
            d_log_likelihood = (Y_at[:, np.newaxis] * X_at).sum(axis=0) - (X_at * (np.exp(X_at.dot(beta_at)) / (1 + np.exp(X_at.dot(beta_at))))[:, np.newaxis]).sum(axis=0)
            return - d_log_likelihood - d_log_prior
        else:
            return - d_log_prior

    def log_posterior_beta_at(self, G, beta_at):
        log_posterior = 0
        if sum(G==0) > 0:
            X_at = self.X[G==0]
            Y_at = self.Y[G==0]
            log_posterior += Y_at.dot(X_at.dot(beta_at)).sum() - np.log1p(np.exp(X_at.dot(beta_at))).sum()
        log_posterior += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_at).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_at))).sum())
        return log_posterior

    def d_nlp_d_beta_nt(self, G, beta_nt):
        d_log_prior = (2 * self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 2 * (self.X * (np.exp(self.X.dot(beta_nt)) / (1 + np.exp(self.X.dot(beta_nt))))[:, np.newaxis]).sum(axis=0))
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            Y_nt = self.Y[G==1]
            d_log_likelihood = (Y_nt[:, np.newaxis] * X_nt).sum(axis=0) - (X_nt * (np.exp(X_nt.dot(beta_nt)) / (1 + np.exp(X_nt.dot(beta_nt))))[:, np.newaxis]).sum(axis=0)         
            return - d_log_likelihood - d_log_prior
        else:
            return - d_log_prior

    def log_posterior_beta_nt(self, G, beta_nt):
        log_posterior = 0
        if sum(G==1) > 0:
            X_nt = self.X[G==1]
            Y_nt = self.Y[G==1]
            log_posterior += Y_nt.dot(X_nt.dot(beta_nt)).sum() - np.log1p(np.exp(X_nt.dot(beta_nt))).sum()
        log_posterior += (2 * self.N_a / 12 / self.N) * (self.X.dot(beta_nt).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_nt))).sum())
        return log_posterior
  
    def d_nlp_d_beta_co_c(self, G, beta_co_c):
        d_log_prior = (self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 2 * (self.X * (np.exp(self.X.dot(beta_co_c)) / (1 + np.exp(self.X.dot(beta_co_c))))[:, np.newaxis]).sum(axis=0))
        if sum((G==2) & (self.Z==0)) > 0:
            X_co_c = self.X[(G==2) & (self.Z==0)]
            Y_co_c = self.Y[(G==2) & (self.Z==0)]
            d_log_likelihood = (Y_co_c[:, np.newaxis] * X_co_c).sum(axis=0) - (X_co_c * (np.exp(X_co_c.dot(beta_co_c)) / (1 + np.exp(X_co_c.dot(beta_co_c))))[:, np.newaxis]).sum(axis=0)
            return - d_log_likelihood - d_log_prior
        else:
            return - d_log_prior
        
    def log_posterior_beta_co_c(self, G, beta_co_c):
        log_posterior = 0
        if sum((G==2) & (self.Z==0)) > 0:
            X_co_c = self.X[(G==2) & (self.Z==0)]
            Y_co_c = self.Y[(G==2) & (self.Z==0)]
            log_posterior += Y_co_c.dot(X_co_c.dot(beta_co_c)).sum() - np.log1p(np.exp(X_co_c.dot(beta_co_c))).sum()
        log_posterior += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_c).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_c))).sum())
        return log_posterior

    def d_nlp_d_beta_co_t(self, G, beta_co_t):
        d_log_prior = (self.N_a / 12 / self.N) * (self.X.sum(axis=0) - 2 * (self.X * (np.exp(self.X.dot(beta_co_t)) / (1 + np.exp(self.X.dot(beta_co_t))))[:, np.newaxis]).sum(axis=0))
        if sum((G==2) & (self.Z==1)) > 0:
            X_co_t = self.X[(G==2) & (self.Z==1)]
            Y_co_t = self.Y[(G==2) & (self.Z==1)]
            d_log_likelihood = (Y_co_t[:, np.newaxis] * X_co_t).sum(axis=0) - (X_co_t * (np.exp(X_co_t.dot(beta_co_t)) / (1 + np.exp(X_co_t.dot(beta_co_t))))[:, np.newaxis]).sum(axis=0)
            return - d_log_likelihood - d_log_prior
        else:
            return - d_log_prior

    def log_posterior_beta_co_t(self, G, beta_co_t):
        log_posterior = 0
        if sum((G==2) & (self.Z==1)) > 0:
            X_co_t = self.X[(G==2) & (self.Z==1)]
            Y_co_t = self.Y[(G==2) & (self.Z==1)]
            log_posterior += Y_co_t.dot(X_co_t.dot(beta_co_t)).sum() - np.log1p(np.exp(X_co_t.dot(beta_co_t))).sum()
        log_posterior += (self.N_a / 12 / self.N) * (self.X.dot(beta_co_t).sum() - 2 * np.log1p(np.exp(self.X.dot(beta_co_t))).sum())
        return log_posterior

    def sampling_hmc(
        self,
        num_samples,
        thinning=1,
        burn_in=0,
        step_size={"gamma": 1e-3, "beta": 1e-4},
        num_step={"gamma": 1, "beta": 1},
        gamma_at_init=None,
        gamma_nt_init=None,
        beta_at_init=None,
        beta_nt_init=None,
        beta_co_c_init=None,
        beta_co_t_init=None
    ):
        """
        num_samples: number of samples
        step_size: step size of hmc
        num_step: number of hmc step
        """

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
                gamma_at, gamma_nt = self.gamma_sampler(G, gamma_at, gamma_nt, step_size["gamma"], num_step["gamma"])
                beta_at, beta_nt, beta_co_c, beta_co_t = self.beta_sampler(G, beta_at, beta_nt, beta_co_c, beta_co_t, step_size["beta"], num_step["beta"])

            if i >= burn_in:
                self.G_samples[i-burn_in, :] = G
                self.gamma_at_samples[i-burn_in, :] = gamma_at
                self.gamma_nt_samples[i-burn_in, :] = gamma_nt
                self.beta_at_samples[i-burn_in, :] = beta_at
                self.beta_nt_samples[i-burn_in, :] = beta_nt
                self.beta_co_c_samples[i-burn_in, :] = beta_co_c
                self.beta_co_t_samples[i-burn_in, :] = beta_co_t
                self.lp_list[i-burn_in] = self.log_posterior(G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)

        self.W_obs_and_mis_samples = np.array(
            [self.W_obs_and_mis_sampler(
                self.G_samples[i, :]
            ) for i in range(num_samples)]
        )

        self.Y_obs_and_mis_samples = np.array(
            [self.Y_obs_and_mis_sampler(
                self.G_samples[i, :],
                self.beta_co_c_samples[i, :],
                self.beta_co_t_samples[i, :]
            ) for i in range(num_samples)]
        )

        self.tau_late_samples = np.array(
            [self.tau_late_sampler(
                self.Y_obs_and_mis_samples[i, :],
                self.G_samples[i, :]
            ) for i in range(num_samples)]
        )