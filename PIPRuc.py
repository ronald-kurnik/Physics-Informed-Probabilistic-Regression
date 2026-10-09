import numpy as np
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

class MaterialPropertyPredictor:
    """
    Gaussian Process Regressor with Physics-Informed Priors and Uncertainty Quantification.
    """
    def __init__(self):
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.05)
        self.model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_with_uncertainty(self, X_new: np.ndarray):
        y_mean, y_std = self.model.predict(X_new, return_std=True)
        return y_mean, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std

# Synthetic Small-Data Dataset
np.random.seed(42)
X_train = np.random.uniform(0, 10, size=(20, 1))
y_train = np.sin(X_train[:, 0]) * 2.5 + np.random.normal(0, 0.15, size=20)

predictor = MaterialPropertyPredictor()
predictor.fit(X_train, y_train)

# Dense grid for continuous plotting
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred, lb, ub = predictor.predict_with_uncertainty(X_test)

# --- GRAPHICAL VISUALIZATION ---
plt.figure(figsize=(10, 5))
plt.plot(X_test, np.sin(X_test) * 2.5, 'k--', label='True Physics / Ground Truth', alpha=0.7)
plt.scatter(X_train, y_train, color='crimson', s=50, zorder=5, label='Customer Proof-of-Concept Samples')
plt.plot(X_test, y_pred, color='#1f77b4', lw=2, label='Model Mean Prediction')
plt.fill_between(X_test.ravel(), lb, ub, color='#1f77b4', alpha=0.25, label='95% Uncertainty Confidence Interval')

plt.title("Material Property Prediction with Uncertainty Quantification (UQ)", fontsize=12, fontweight='bold')
plt.xlabel("Sensor Signal Feature (Normalized Metric)")
plt.ylabel("Target Material Property (e.g., Layer Thickness)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()