import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.cluster import KMeans

class SemiSupervisedGMM:
    def __init__(self, n_components, n_classes, max_iter=200, tol=1e-3, device='cuda',
                 reg_covar=1e-6):
        self.n_components = n_components
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.reg_covar = reg_covar  # Ridge regularisation added to every covariance
        self.alpha = None  # Mixing coefficients
        self.mu = None  # Means
        self.sigma = None  # Covariance matrices
        self.beta = None  # Class-component probabilities

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Fits the semi-supervised GMM to the data.

        Args:
            X_labeled: Labeled data (numpy array).
            y_labeled: Labels for labeled data (numpy array).
            X_unlabeled: Unlabeled data (numpy array).
        """
        print("-" * 30)
        print("Starting training of Semi-Supervised GMM...")
        print(f"Labeled samples: {X_labeled.shape[0]}, Unlabeled samples: {X_unlabeled.shape[0]}")
        print(f"Device: {self.device}")

        # Convert data to PyTorch tensors
        X_labeled = torch.from_numpy(X_labeled).to(dtype=torch.float64, device=self.device)
        y_labeled = torch.from_numpy(y_labeled).long().to(self.device)
        X_unlabeled = torch.from_numpy(X_unlabeled).to(dtype=torch.float64, device=self.device)

        X_all = torch.cat([X_labeled, X_unlabeled])

        # Initialize parameters using KMeans
        self._initialize_parameters(X_all)

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            gamma_labeled, gamma_unlabeled = self._e_step(X_labeled, y_labeled, X_unlabeled)

            # M-step: Update parameters
            self._m_step(X_labeled, y_labeled, X_unlabeled, gamma_labeled, gamma_unlabeled)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X_labeled, y_labeled, X_unlabeled)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1} complete, Log-Likelihood: {log_likelihood.item():.4f}")

            # Check for convergence
            if torch.abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"\nModel converged at iteration {iteration+1}")
                break
            log_likelihood_old = log_likelihood

        print("Model training complete!")

    def _initialize_parameters(self, X):
        """
        Initializes the parameters of the GMM.

        Args:
            X: Combined labeled and unlabeled data (PyTorch tensor).
        """
        kmeans = KMeans(
            n_clusters=self.n_components,
            init='k-means++',
            n_init=10,
            random_state=42
        ).fit(X.cpu().numpy())
        assignments = kmeans.labels_
        counts = np.bincount(assignments)

        self.alpha = torch.from_numpy(counts / counts.sum()).to(device=self.device, dtype=torch.float64)
        self.mu = torch.from_numpy(kmeans.cluster_centers_).to(device=self.device, dtype=torch.float64)

        D = X.shape[1]
        reg = self.reg_covar * np.eye(D)
        covariances = []
        for i in range(self.n_components):
            data_i = X.cpu().numpy()[assignments == i]
            if len(data_i) > 1:
                # Empirical covariance can be rank-deficient when n_samples < D.
                # Ridge regularisation ensures positive-definiteness throughout.
                cov_i = np.cov(data_i, rowvar=False) + reg
            else:
                cov_i = np.eye(D)
            covariances.append(cov_i)
        self.sigma = torch.from_numpy(np.stack(covariances, axis=0)).to(device=self.device, dtype=torch.float64)
        self.init_sigma = self.sigma.clone()  # Save initial covariance matrices

        self.beta = torch.rand(self.n_classes, self.n_components, device=self.device, dtype=torch.float64)
        self.beta /= self.beta.sum(dim=0, keepdim=True)  # Normalize

        print(f"  alpha shape: {self.alpha.shape}")
        print(f"  mu shape: {self.mu.shape}")
        print(f"  sigma shape: {self.sigma.shape}")
        print(f"  beta shape: {self.beta.shape}")
        print("Parameter initialization complete!")

    def _e_step(self, X_labeled, y_labeled, X_unlabeled):
        """
        Performs the E-step of the EM algorithm.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).

        Returns:
            gamma_labeled: Responsibilities for labeled data.
            gamma_unlabeled: Responsibilities for unlabeled data.
        """
        # Calculate responsibilities for labeled data
        log_probs_labeled = torch.zeros((X_labeled.shape[0], self.n_components), device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs_labeled[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X_labeled) + torch.log(self.beta[y_labeled, l])

        log_sum_exp_labeled = torch.logsumexp(log_probs_labeled, dim=1, keepdim=True)
        gamma_labeled = torch.exp(log_probs_labeled - log_sum_exp_labeled)

        # Calculate responsibilities for unlabeled data
        log_probs_unlabeled = torch.zeros((X_unlabeled.shape[0], self.n_components), device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs_unlabeled[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X_unlabeled)

        log_sum_exp_unlabeled = torch.logsumexp(log_probs_unlabeled, dim=1, keepdim=True)
        gamma_unlabeled = torch.exp(log_probs_unlabeled - log_sum_exp_unlabeled)

        return gamma_labeled, gamma_unlabeled

    def _m_step(self, X_labeled, y_labeled, X_unlabeled, gamma_labeled, gamma_unlabeled):
        """
        Performs the M-step of the EM algorithm.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).
            gamma_labeled: Responsibilities for labeled data.
            gamma_unlabeled: Responsibilities for unlabeled data.
        """
        _eps = 1e-10  # guard against dead-component (zero-responsibility) NaN
        resp_total = (gamma_labeled.sum(dim=0) + gamma_unlabeled.sum(dim=0)).clamp(min=_eps)

        # Update mu
        for l in range(self.n_components):
            numerator = (gamma_labeled[:, l, None] * X_labeled).sum(dim=0) + (gamma_unlabeled[:, l, None] * X_unlabeled).sum(dim=0)
            denominator = (gamma_labeled[:, l].sum() + gamma_unlabeled[:, l].sum()).clamp(min=_eps)
            self.mu[l] = numerator / denominator

        # Update sigma
        for l in range(self.n_components):
            diff_labeled = X_labeled - self.mu[l]
            sigma_labeled = (gamma_labeled[:, l, None, None] * torch.matmul(diff_labeled[:, :, None], diff_labeled[:, None, :])).sum(dim=0)

            diff_unlabeled = X_unlabeled - self.mu[l]
            sigma_unlabeled = (gamma_unlabeled[:, l, None, None] * torch.matmul(diff_unlabeled[:, :, None], diff_unlabeled[:, None, :])).sum(dim=0)

            denominator = (gamma_labeled[:, l].sum() + gamma_unlabeled[:, l].sum()).clamp(min=_eps)
            reg = self.reg_covar * torch.eye(self.sigma.shape[-1], device=self.device, dtype=torch.float64)
            self.sigma[l] = (sigma_labeled + sigma_unlabeled) / denominator + reg

            # Ensure that no covariance is too small or non-finite.
            # Check isfinite first: near-singular matrices cause CUDA SVD to return
            # NaN singular values, and NaN < eps is always False — so the SVD-only
            # check silently misses the failure.
            _reset_sigma = False
            if not torch.isfinite(self.sigma[l]).all():
                _reset_sigma = True
            else:
                s = torch.linalg.svdvals(self.sigma[l])
                if not torch.isfinite(s).all() or torch.min(s) < np.finfo(float).eps:
                    _reset_sigma = True
            if _reset_sigma:
                self.sigma[l] = self.init_sigma[l].clone()
                print(f"Component {l} covariance matrix reset to initial value.")

        # Update alpha
        self.alpha = (gamma_labeled.sum(dim=0) + gamma_unlabeled.sum(dim=0)) / resp_total

        # Update beta(EM-I)
        for k in range(self.n_classes):
            numerator = gamma_labeled[y_labeled == k].sum(dim=0)
            denominator = gamma_labeled.sum(dim=0).clamp(min=_eps)
            self.beta[k] = numerator / denominator

        '''# Update beta (EM-II)
        for k in range(self.n_classes):
            for l in range(self.n_components):
                # Numerator for labeled data
                numerator_labeled = gamma_labeled[y_labeled == k, l].sum()

                # Numerator for unlabeled data: P(l, c_i=k|x_i, theta^t) = self.beta[k, l] * gamma_unlabeled[:, l]
                numerator_unlabeled = (self.beta[k, l] * gamma_unlabeled[:, l]).sum()

                # Denominator for labeled data
                denominator_labeled = gamma_labeled[:, l].sum()

                # Denominator for unlabeled data
                denominator_unlabeled = gamma_unlabeled[:, l].sum()

                self.beta[k, l] = (numerator_labeled + numerator_unlabeled) / (denominator_labeled + denominator_unlabeled)'''

    def _compute_log_likelihood(self, X_labeled, y_labeled, X_unlabeled):
        """
        Computes the log-likelihood of the data given the model parameters.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).

        Returns:
            The log-likelihood of the data.
        """

        # Log-likelihood for labeled data
        labeled_log_probs = []
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_p = torch.log(self.alpha[l]) + mvn.log_prob(X_labeled) + torch.log(self.beta[y_labeled, l])
            labeled_log_probs.append(log_p.unsqueeze(1))
        labeled_log_probs = torch.cat(labeled_log_probs, dim=1)
        labeled_ll = torch.logsumexp(labeled_log_probs, dim=1).sum()

        # Log-likelihood for unlabeled data
        unlabeled_log_probs = []
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_p = torch.log(self.alpha[l]) + mvn.log_prob(X_unlabeled)
            unlabeled_log_probs.append(log_p.unsqueeze(1))
        unlabeled_log_probs = torch.cat(unlabeled_log_probs, dim=1)
        unlabeled_ll = torch.logsumexp(unlabeled_log_probs, dim=1).sum()

        return labeled_ll + unlabeled_ll

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X: Data to predict (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        print("-" * 30)
        print(f"Predicting {X.shape[0]} samples...")

        X = torch.from_numpy(X).to(dtype=torch.float64, device=self.device)

        # Calculate responsibilities (E-step)
        log_probs = torch.zeros((X.shape[0], self.n_components), device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X)
        log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)
        gamma = torch.exp(log_probs - log_sum_exp)

        # Calculate posterior probabilities
        posteriors = torch.matmul(gamma, self.beta.T)

        # Get predictions
        predictions = torch.argmax(posteriors, dim=1)

        print("Prediction complete!")
        return predictions.cpu().numpy()

    def predict_clusters(self, X):
        """
        Returns Gaussian component (cluster) assignments — argmax of responsibilities.

        Unlike predict(), which returns class labels via the beta matrix, this
        returns the raw component index for each sample, suitable for use as
        cluster IDs in ACC/AMI evaluation.

        Args:
            X: Data to predict (numpy array).

        Returns:
            Component assignments as int64 numpy array, shape (N,).
        """
        X = torch.from_numpy(X).to(dtype=torch.float64, device=self.device)
        log_probs = torch.zeros((X.shape[0], self.n_components),
                                device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X)
        log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)
        gamma = torch.exp(log_probs - log_sum_exp)
        return torch.argmax(gamma, dim=1).cpu().numpy().astype(np.int64)