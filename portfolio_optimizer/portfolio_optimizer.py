import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, cut_tree
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) implementation with ML enhancements.
    
    HRP addresses key limitations of mean-variance optimization:
    - More robust to estimation errors
    - No matrix inversion required (handles singular covariance)
    - Better out-of-sample performance
    - Natural interpretation through hierarchical clustering
    
    ML Enhancements:
    - Robust covariance estimation (Ledoit-Wolf shrinkage)
    - Feature engineering for return prediction
    - Dynamic correlation adjustments
    - Regime detection for adaptive allocation
    """
    
    def __init__(self, 
                 covariance_method: str = 'ledoit-wolf',
                 linkage_method: str = 'ward',
                 risk_measure: str = 'variance'):
        """
        Initialize HRP with configuration options.
        
        Args:
            covariance_method: Method for covariance estimation 
                              ('ledoit-wolf', 'oas', 'empirical')
            linkage_method: Hierarchical clustering method
            risk_measure: Risk measure to use ('variance', 'cvar', 'mad')
        """
        self.covariance_method = covariance_method
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure
        self.returns_data = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.clusters = None
        self.linkage_matrix = None
        
    def fit(self, returns: pd.DataFrame):
        """
        Fit the HRP model with returns data.
        
        Args:
            returns: DataFrame of asset returns
        """
        self.returns_data = returns
        self.asset_names = returns.columns.tolist()
        
        # Estimate robust covariance matrix
        self.cov_matrix = self._estimate_covariance(returns)
        
        # Calculate correlation matrix from covariance
        std_dev = np.sqrt(np.diag(self.cov_matrix))
        self.corr_matrix = self.cov_matrix / np.outer(std_dev, std_dev)
        
        # Perform hierarchical clustering
        self._hierarchical_clustering()
        
    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix using specified method with ML enhancements.
        """
        if self.covariance_method == 'ledoit-wolf':
            # Ledoit-Wolf shrinkage estimation
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_
        elif self.covariance_method == 'oas':
            # Oracle Approximating Shrinkage
            oas = OAS()
            cov_matrix = oas.fit(returns).covariance_
        else:
            # Empirical covariance
            cov_matrix = returns.cov().values
            
        return cov_matrix
    
    def _hierarchical_clustering(self):
        """
        Perform hierarchical clustering on the correlation matrix.
        """
        # Convert correlation to distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - self.corr_matrix))
        
        # Ensure distance matrix is valid (no negative values)
        dist_matrix = np.maximum(dist_matrix, 0)
        np.fill_diagonal(dist_matrix, 0)
        
        # Create condensed distance matrix for linkage
        condensed_dist = squareform(dist_matrix)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(condensed_dist, method=self.linkage_method)
        
    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Recover the quasi-diagonal form from hierarchical clustering.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            
            sort_ix[i] = link[j, 0]
            
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()
    
    def _get_cluster_variance(self, cov: np.ndarray, cluster_items: List[int]) -> float:
        """
        Calculate cluster variance using the inverse-variance allocation.
        """
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        cluster_var = np.dot(ivp, np.diag(cov_slice))
        return cluster_var
    
    def _recursive_bisection(self, cov: np.ndarray, sort_ix: List[int]) -> pd.Series:
        """
        Perform recursive bisection for HRP allocation.
        """
        w = pd.Series(1.0, index=sort_ix)
        cluster_items = [sort_ix]
        
        while len(cluster_items) > 0:
            # Pop a cluster for bisection
            cluster = cluster_items.pop()
            if len(cluster) > 1:
                # Get cluster variances
                cluster_0 = cluster[:len(cluster)//2]
                cluster_1 = cluster[len(cluster)//2:]
                
                var_0 = self._get_cluster_variance(cov, cluster_0)
                var_1 = self._get_cluster_variance(cov, cluster_1)
                
                # Allocation based on inverse variance
                alpha = 1 - var_0 / (var_0 + var_1)
                
                # Update weights
                w[cluster_0] *= alpha
                w[cluster_1] *= (1 - alpha)
                
                # Add subclusters to the list
                cluster_items.append(cluster_0)
                cluster_items.append(cluster_1)
                
        return w
    
    def calculate_weights(self) -> Dict[str, float]:
        """
        Calculate HRP portfolio weights.
        
        Returns:
            Dictionary of asset weights
        """
        # Get quasi-diagonal sorted indices
        sort_ix = self._get_quasi_diag(self.linkage_matrix)
        
        # Map sorted indices back to original positions
        sort_ix = [self.asset_names.index(self.asset_names[i]) for i in sort_ix]
        
        # Calculate HRP weights
        weights = self._recursive_bisection(self.cov_matrix, sort_ix)
        
        # Normalize weights
        weights /= weights.sum()
        
        # Return as dictionary
        return dict(zip(self.asset_names, weights))
    
    def ml_enhanced_allocation(self, 
                             returns: pd.DataFrame,
                             features: Optional[pd.DataFrame] = None,
                             lookback_days: int = 252) -> Dict[str, float]:
        """
        ML-enhanced HRP allocation using regime detection and return prediction.
        
        Args:
            returns: Historical returns
            features: Additional features for ML model (e.g., market indicators)
            lookback_days: Days to use for regime detection
            
        Returns:
            Enhanced weight allocation
        """
        # 1. Regime Detection using PCA and clustering
        regime_weights = self._detect_regime(returns, lookback_days)
        
        # 2. Feature engineering for return prediction
        if features is None:
            features = self._engineer_features(returns)
        
        # 3. Predict future correlations using ML
        predicted_corr = self._predict_correlations(returns, features)
        
        # 4. Adjust covariance matrix based on predictions
        adjusted_cov = self._adjust_covariance(predicted_corr)
        
        # 5. Calculate enhanced HRP weights
        self.cov_matrix = adjusted_cov
        self._hierarchical_clustering()
        base_weights = self.calculate_weights()
        
        # 6. Blend with regime-based weights
        final_weights = {}
        for asset in self.asset_names:
            final_weights[asset] = (
                0.7 * base_weights[asset] + 
                0.3 * regime_weights.get(asset, base_weights[asset])
            )
        
        # Normalize
        total = sum(final_weights.values())
        final_weights = {k: v/total for k, v in final_weights.items()}
        
        return final_weights
    
    def _detect_regime(self, returns: pd.DataFrame, lookback_days: int) -> Dict[str, float]:
        """
        Detect market regime using rolling statistics and clustering.
        """
        # Calculate rolling statistics
        rolling_mean = returns.rolling(20).mean()
        rolling_vol = returns.rolling(20).std()
        rolling_corr = returns.rolling(lookback_days).corr().mean()
        
        # Create regime features
        regime_features = pd.DataFrame({
            'trend': rolling_mean.mean(axis=1),
            'volatility': rolling_vol.mean(axis=1),
            'correlation': rolling_corr
        }).dropna()
        
        # Standardize features
        scaler = StandardScaler()
        regime_features_scaled = scaler.fit_transform(regime_features)
        
        # Identify current regime using last observation
        current_regime = regime_features_scaled[-1]
        
        # Adjust weights based on regime
        # High volatility regime: more conservative allocation
        # Low correlation regime: more equal weighting
        vol_factor = 1 / (1 + np.exp(current_regime[1]))  # Sigmoid transformation
        corr_factor = 1 / (1 + np.exp(-current_regime[2]))
        
        # Create regime-adjusted weights
        n_assets = len(self.asset_names)
        base_weight = 1 / n_assets
        
        regime_weights = {}
        for i, asset in enumerate(self.asset_names):
            # Adjust based on asset's recent performance and regime
            asset_vol = rolling_vol[asset].iloc[-1]
            asset_return = rolling_mean[asset].iloc[-1]
            
            # Lower weight for high volatility assets in volatile regime
            vol_adjustment = vol_factor * (1 / (1 + asset_vol))
            
            # Momentum adjustment
            momentum_adjustment = 1 + 0.1 * np.tanh(asset_return * 100)
            
            regime_weights[asset] = base_weight * vol_adjustment * momentum_adjustment
            
        return regime_weights
    
    def _engineer_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models from returns data.
        """
        features = pd.DataFrame(index=returns.index)
        
        # Basic statistics
        features['mean_return'] = returns.rolling(20).mean().mean(axis=1)
        features['volatility'] = returns.rolling(20).std().mean(axis=1)
        features['skewness'] = returns.rolling(60).skew().mean(axis=1)
        features['kurtosis'] = returns.rolling(60).kurt().mean(axis=1)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(returns.mean(axis=1))
        features['momentum'] = returns.mean(axis=1).rolling(10).mean()
        
        # Correlation features
        rolling_corr = returns.rolling(30).corr()
        features['avg_correlation'] = rolling_corr.mean().mean()
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _predict_correlations(self, 
                            returns: pd.DataFrame, 
                            features: pd.DataFrame) -> np.ndarray:
        """
        Use ML to predict future correlations based on features.
        """
        # Prepare training data
        window = 60
        X_train = []
        y_train = []
        
        for i in range(window, len(returns) - 20):
            # Features at time i
            X_train.append(features.iloc[i].values)
            
            # Future correlation (20 days ahead)
            future_corr = returns.iloc[i:i+20].corr().values
            y_train.append(future_corr.flatten())
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )
        
        # Train model for each correlation pair
        n_assets = len(self.asset_names)
        predicted_corr = np.eye(n_assets)
        
        try:
            rf_model.fit(X_train, y_train)
            
            # Predict current correlations
            current_features = features.iloc[-1].values.reshape(1, -1)
            pred_flat = rf_model.predict(current_features)[0]
            
            # Reshape to correlation matrix
            predicted_corr = pred_flat.reshape(n_assets, n_assets)
            
            # Ensure valid correlation matrix
            predicted_corr = (predicted_corr + predicted_corr.T) / 2
            np.fill_diagonal(predicted_corr, 1)
            
            # Clip values to [-1, 1]
            predicted_corr = np.clip(predicted_corr, -1, 1)
            
        except Exception as e:
            # Fallback to empirical correlation
            predicted_corr = self.corr_matrix
            
        return predicted_corr
    
    def _adjust_covariance(self, predicted_corr: np.ndarray) -> np.ndarray:
        """
        Adjust covariance matrix based on predicted correlations.
        """
        # Get standard deviations from original covariance
        std_dev = np.sqrt(np.diag(self.cov_matrix))
        
        # Blend predicted and historical correlations
        blend_factor = 0.3  # 30% predicted, 70% historical
        blended_corr = (
            blend_factor * predicted_corr + 
            (1 - blend_factor) * self.corr_matrix
        )
        
        # Reconstruct covariance matrix
        adjusted_cov = np.outer(std_dev, std_dev) * blended_corr
        
        # Ensure positive definite
        min_eigenvalue = np.min(np.linalg.eigvals(adjusted_cov))
        if min_eigenvalue < 0:
            adjusted_cov -= 1.1 * min_eigenvalue * np.eye(len(adjusted_cov))
            
        return adjusted_cov
    """
    Portfolio optimization tool implementing Modern Portfolio Theory and the Efficient Frontier.
    Based on Markowitz mean-variance optimization as described in "The Mathematics Behind 
    the Efficient Frontier" by Diego Alvarez.
    
    Key concepts:
    - Efficient Frontier: Set of optimal portfolios offering highest expected return for each risk level
    - Two Fund Theorem: Any portfolio on the efficient frontier can be created from two efficient portfolios
    - Minimum Variance Portfolio: Portfolio with the lowest possible risk
    - Maximum Sharpe Portfolio: Best risk-adjusted returns (Mean-Markowitz Portfolio)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer with a risk-free rate for Sharpe ratio calculations.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.cov_matrix = None  # Covariance matrix Ω
        self.mean_returns = None  # Mean returns vector μ
        
        # Efficient frontier parameters (η, ξ, γ notation from the paper)
        self.eta = None  # η = 1ᵀΩ⁻¹1
        self.xi = None   # ξ = 1ᵀΩ⁻¹μ  
        self.gamma = None  # γ = μᵀΩ⁻¹μ
        
    def load_price_data(self, prices: pd.DataFrame):
        """
        Load historical price data and calculate returns.
        Following the mathematical framework where:
        - Returns: Rₙ = (Sₙ(T) - Sₙ(t)) / Sₙ(t)
        - Portfolio return: Rₚ = Σ wₙRₙ
        - Portfolio variance: σₚ² = wᵀΩw
        
        Args:
            prices: DataFrame with dates as index and tickers as columns
        """
        # Calculate daily returns
        self.returns_data = prices.pct_change().dropna()
        
        # Calculate annualized mean returns (μ) and covariance matrix (Ω)
        self.mean_returns = self.returns_data.mean() * 252
        self.cov_matrix = self.returns_data.cov() * 252
        
        # Pre-calculate efficient frontier parameters for optimization
        try:
            cov_inv = np.linalg.inv(self.cov_matrix.values)
            ones = np.ones(len(self.mean_returns))
            mu = self.mean_returns.values
            
            # η = 1ᵀΩ⁻¹1
            self.eta = np.dot(ones.T, np.dot(cov_inv, ones))
            # ξ = 1ᵀΩ⁻¹μ
            self.xi = np.dot(ones.T, np.dot(cov_inv, mu))
            # γ = μᵀΩ⁻¹μ
            self.gamma = np.dot(mu.T, np.dot(cov_inv, mu))
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular. Some optimizations may fail.")
        
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, risk, and Sharpe ratio.
        
        Mathematical formulation:
        - Expected return: μₚ = E[Rₚ] = Σ wₙμₙ
        - Variance: σₚ² = wᵀΩw
        - Sharpe ratio: S = (μₚ - rf) / σₚ
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Portfolio expected return: μₚ = wᵀμ
        portfolio_return = np.dot(weights, self.mean_returns)
        
        # Portfolio variance: σₚ² = wᵀΩw
        portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Sharpe ratio: S = (μₚ - rf) / σₚ
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
def find_minimum_variance_portfolio(self) -> Dict:
    """
    Find the Minimum Variance Portfolio using analytical solution.
    
    From the paper, the minimum variance portfolio has weights:
    w = Ω⁻¹1 / (1ᵀΩ⁻¹1)
    
    This is the leftmost point on the efficient frontier (green star in the paper).
    """
    try:
        cov_inv = np.linalg.inv(self.cov_matrix.values)
        ones = np.ones(len(self.mean_returns))
        
        # w = Ω⁻¹1 / (1ᵀΩ⁻¹1)
        weights = np.dot(cov_inv, ones) / self.eta
        
        # Ensure weights are normalized
        weights = weights / weights.sum()
        
        # Calculate metrics
        ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.returns_data.columns, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    except np.linalg.LinAlgError:
        # Fallback to numerical optimization if matrix is singular
        return self.optimize_portfolio(maximize_sharpe=False)
    def find_maximum_sharpe_portfolio(self) -> Dict:
        """
        Find the Maximum Sharpe Ratio Portfolio (Mean-Markowitz Portfolio).
        
        From the paper, this is the portfolio with the best risk-to-reward profile.
        Uses Lagrangian optimization to find weights that maximize Sharpe ratio.
        
        The analytical solution involves solving:
        w = (γ - ξμₚ)/(ηγ - ξ²) * Ω⁻¹1 + (ημₚ - ξ)/(ηγ - ξ²) * Ω⁻¹μ
        """
        return self.optimize_portfolio(maximize_sharpe=True)
    
    def optimize_portfolio(self, 
                          target_return: Optional[float] = None,
                          maximize_sharpe: bool = True,
                          constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio weights based on given criteria.
        
        Args:
            target_return: Target return for minimum variance portfolio
            maximize_sharpe: If True, maximize Sharpe ratio; else minimize variance
            constraints: Additional constraints (e.g., sector limits)
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(self.mean_returns)
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
        
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.mean_returns) - target_return
            })
        
        # Add custom constraints if provided
        if constraints:
            for constraint in constraints.get('additional', []):
                cons.append(constraint)
        
        # Bounds (0 <= weight <= 1 for each asset)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function
        if maximize_sharpe:
            # Negative Sharpe ratio (minimize negative = maximize positive)
            def objective(weights):
                ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
                return -sharpe
        else:
            # Portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)
        
        # Calculate final metrics
        opt_weights = result.x
        opt_return, opt_vol, opt_sharpe = self.calculate_portfolio_metrics(opt_weights)
        
        return {
            'weights': dict(zip(self.returns_data.columns, opt_weights)),
            'expected_return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe
        }
    
    def two_fund_theorem_analysis(self, asset1_idx: int, asset2_idx: int) -> pd.DataFrame:
        """
        Implement the Two Fund Theorem to understand efficient frontier shape.
        
        From the paper: Any portfolio can be constructed from two assets with weight α:
        - μₚ = (1-α)R₁ + αR₂
        - σₚ² = (1-α)²σ₁² + 2ρα(1-α)σ₁σ₂ + α²σ₂²
        
        Args:
            asset1_idx: Index of first asset
            asset2_idx: Index of second asset
            
        Returns:
            DataFrame with frontier points for the two-asset portfolio
        """
        # Get statistics for the two assets
        mu1 = self.mean_returns.iloc[asset1_idx]
        mu2 = self.mean_returns.iloc[asset2_idx]
        sigma1 = np.sqrt(self.cov_matrix.iloc[asset1_idx, asset1_idx])
        sigma2 = np.sqrt(self.cov_matrix.iloc[asset2_idx, asset2_idx])
        cov12 = self.cov_matrix.iloc[asset1_idx, asset2_idx]
        rho = cov12 / (sigma1 * sigma2)  # Correlation coefficient
        
        # Generate frontier for different alpha values
        alphas = np.linspace(0, 1, 100)
        frontier_data = []
        
        for alpha in alphas:
            # Portfolio return: μₚ = (1-α)R₁ + αR₂
            portfolio_return = (1 - alpha) * mu1 + alpha * mu2
            
            # Portfolio variance based on correlation
            if abs(rho - 1) < 1e-10:  # Perfect positive correlation (ρ = 1)
                # σₚ = |(1-α)σ₁ + ασ₂|
                portfolio_vol = abs((1 - alpha) * sigma1 + alpha * sigma2)
            elif abs(rho + 1) < 1e-10:  # Perfect negative correlation (ρ = -1)
                # σₚ = |(1-α)σ₁ - ασ₂|
                portfolio_vol = abs((1 - alpha) * sigma1 - alpha * sigma2)
            else:  # General case
                # σₚ² = (1-α)²σ₁² + 2ρα(1-α)σ₁σ₂ + α²σ₂²
                var = ((1 - alpha)**2 * sigma1**2 + 
                       2 * rho * alpha * (1 - alpha) * sigma1 * sigma2 + 
                       alpha**2 * sigma2**2)
                portfolio_vol = np.sqrt(var)
            
            frontier_data.append({
                'alpha': alpha,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'asset1_weight': 1 - alpha,
                'asset2_weight': alpha
            })
        
        return pd.DataFrame(frontier_data)
    def generate_efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """
        Generate the efficient frontier by calculating minimum variance
        portfolios for different target returns.
        
        The efficient frontier represents the set of portfolios that offer:
        - Highest expected return for a given level of risk
        - Lowest risk for a given level of expected return
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with frontier points
        """
        # Get range of possible returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        
        # Find the minimum variance portfolio first
        min_var_portfolio = self.find_minimum_variance_portfolio()
        min_var_return = min_var_portfolio['expected_return']
        
        # Generate target returns from minimum variance return to maximum return
        # (only the upper half of the parabola is efficient)
        target_returns = np.linspace(min_var_return, max_return, n_points)
        
        frontier_points = []
        
        for target_return in target_returns:
            try:
                result = self.optimize_portfolio(
                    target_return=target_return,
                    maximize_sharpe=False
                )
                frontier_points.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            except:
                continue
        
        return pd.DataFrame(frontier_points)
    
    def analyze_current_portfolio(self, current_weights: Dict[str, float]) -> Dict:
        """
        Analyze the current portfolio's position relative to the efficient frontier.
        
        Args:
            current_weights: Dictionary of ticker: weight
            
        Returns:
            Analysis results including distance from frontier
        """
        # Convert to array
        weights_array = np.array([current_weights.get(ticker, 0) 
                                 for ticker in self.returns_data.columns])
        
        # Normalize weights
        weights_array = weights_array / weights_array.sum()
        
        # Calculate current metrics
        current_return, current_vol, current_sharpe = self.calculate_portfolio_metrics(weights_array)
        
        # Find optimal portfolio with same return
        optimal_same_return = self.optimize_portfolio(
            target_return=current_return,
            maximize_sharpe=False
        )
        
        # Find maximum Sharpe ratio portfolio
        optimal_sharpe = self.optimize_portfolio(maximize_sharpe=True)
        
        # Calculate inefficiency
        inefficiency = (current_vol - optimal_same_return['volatility']) / current_vol
        
        return {
            'current_metrics': {
                'return': current_return,
                'volatility': current_vol,
                'sharpe_ratio': current_sharpe,
                'weights': current_weights
            },
            'optimal_same_return': optimal_same_return,
            'optimal_sharpe': optimal_sharpe,
            'inefficiency_percentage': inefficiency * 100,
            'improvement_potential': {
                'volatility_reduction': current_vol - optimal_same_return['volatility'],
                'sharpe_improvement': optimal_sharpe['sharpe_ratio'] - current_sharpe
            }
        }
    
    def recommend_rebalancing(self, 
                            current_weights: Dict[str, float],
                            max_trade_size: float = 0.1,
                            target_type: str = 'sharpe') -> Dict:
        """
        Recommend rebalancing actions based on risk metrics.
        
        Args:
            current_weights: Current portfolio weights
            max_trade_size: Maximum position change allowed (default 10%)
            target_type: 'sharpe' for max Sharpe, 'minvar' for min variance
            
        Returns:
            Rebalancing recommendations
        """
        # Get optimal portfolio
        if target_type == 'sharpe':
            optimal = self.optimize_portfolio(maximize_sharpe=True)
        else:
            current_return = np.dot(
                [current_weights.get(ticker, 0) for ticker in self.returns_data.columns],
                self.mean_returns
            )
            optimal = self.optimize_portfolio(target_return=current_return, maximize_sharpe=False)
        
        # Calculate recommended changes
        recommendations = []
        
        for ticker in self.returns_data.columns:
            current_weight = current_weights.get(ticker, 0)
            optimal_weight = optimal['weights'][ticker]
            change = optimal_weight - current_weight
            
            # Apply max trade size constraint
            if abs(change) > max_trade_size:
                change = max_trade_size if change > 0 else -max_trade_size
            
            if abs(change) > 0.001:  # Only recommend meaningful changes
                recommendations.append({
                    'ticker': ticker,
                    'current_weight': current_weight,
                    'recommended_weight': current_weight + change,
                    'change': change,
                    'action': 'BUY' if change > 0 else 'SELL'
                })
        
        # Sort by absolute change
        recommendations.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return {
            'recommendations': recommendations,
            'target_portfolio': optimal,
            'expected_improvement': {
                'return': optimal['expected_return'] - self.calculate_portfolio_metrics(
                    np.array([current_weights.get(t, 0) for t in self.returns_data.columns])
                )[0],
                'sharpe_ratio': optimal['sharpe_ratio'] - self.calculate_portfolio_metrics(
                    np.array([current_weights.get(t, 0) for t in self.returns_data.columns])
                )[2]
            }
        }
    
    def optimize_with_hrp(self, use_ml_enhancement: bool = True) -> Dict:
        """
        Optimize portfolio using Hierarchical Risk Parity.
        
        Args:
            use_ml_enhancement: Whether to use ML-enhanced allocation
            
        Returns:
            Dictionary with HRP weights and metrics
        """
        # Fit HRP model
        self.hrp_model.fit(self.returns_data)
        
        # Get weights
        if use_ml_enhancement:
            weights_dict = self.hrp_model.ml_enhanced_allocation(self.returns_data)
        else:
            weights_dict = self.hrp_model.calculate_weights()
        
        # Convert to array for metrics calculation
        weights = np.array([weights_dict[ticker] for ticker in self.returns_data.columns])
        
        # Calculate metrics
        ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
        
        result = {
            'method': 'HRP' + (' ML-Enhanced' if use_ml_enhancement else ''),
            'weights': weights_dict,
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
        
        # Store result
        self.optimization_results['hrp'] = result
        
        return result
    
    def compare_optimization_methods(self) -> pd.DataFrame:
        """
        Compare different portfolio optimization methods:
        - Mean-Variance (Max Sharpe)
        - Minimum Variance
        - HRP
        - HRP ML-Enhanced
        - Equal Weight (benchmark)
        
        Returns:
            DataFrame comparing methods
        """
        results = []
        
        # 1. Maximum Sharpe Ratio (Mean-Variance)
        try:
            max_sharpe = self.optimize_portfolio(maximize_sharpe=True)
            results.append({
                'Method': 'Maximum Sharpe Ratio',
                'Expected Return': max_sharpe['expected_return'],
                'Volatility': max_sharpe['volatility'],
                'Sharpe Ratio': max_sharpe['sharpe_ratio'],
                'Max Weight': max(max_sharpe['weights'].values()),
                'Effective Assets': 1 / sum(w**2 for w in max_sharpe['weights'].values())
            })
        except:
            pass
        
        # 2. Minimum Variance
        try:
            min_var = self.find_minimum_variance_portfolio()
            results.append({
                'Method': 'Minimum Variance',
                'Expected Return': min_var['expected_return'],
                'Volatility': min_var['volatility'],
                'Sharpe Ratio': min_var['sharpe_ratio'],
                'Max Weight': max(min_var['weights'].values()),
                'Effective Assets': 1 / sum(w**2 for w in min_var['weights'].values())
            })
        except:
            pass
        
        # 3. HRP
        try:
            hrp = self.optimize_with_hrp(use_ml_enhancement=False)
            results.append({
                'Method': 'HRP',
                'Expected Return': hrp['expected_return'],
                'Volatility': hrp['volatility'],
                'Sharpe Ratio': hrp['sharpe_ratio'],
                'Max Weight': max(hrp['weights'].values()),
                'Effective Assets': 1 / sum(w**2 for w in hrp['weights'].values())
            })
        except:
            pass
        
        # 4. HRP ML-Enhanced
        try:
            hrp_ml = self.optimize_with_hrp(use_ml_enhancement=True)
            results.append({
                'Method': 'HRP ML-Enhanced',
                'Expected Return': hrp_ml['expected_return'],
                'Volatility': hrp_ml['volatility'],
                'Sharpe Ratio': hrp_ml['sharpe_ratio'],
                'Max Weight': max(hrp_ml['weights'].values()),
                'Effective Assets': 1 / sum(w**2 for w in hrp_ml['weights'].values())
            })
        except:
            pass
        
        # 5. Equal Weight (benchmark)
        n_assets = len(self.mean_returns)
        equal_weights = np.ones(n_assets) / n_assets
        eq_ret, eq_vol, eq_sharpe = self.calculate_portfolio_metrics(equal_weights)
        results.append({
            'Method': 'Equal Weight',
            'Expected Return': eq_ret,
            'Volatility': eq_vol,
            'Sharpe Ratio': eq_sharpe,
            'Max Weight': 1/n_assets,
            'Effective Assets': n_assets
        })
        
        return pd.DataFrame(results)
    
    def create_hrp_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Create an efficient frontier that combines HRP with mean-variance optimization.
        
        This creates a frontier where each point blends HRP weights with
        mean-variance weights to achieve target returns while maintaining
        HRP's robustness benefits.
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with blended frontier points
        """
        # Get HRP weights
        hrp_result = self.optimize_with_hrp(use_ml_enhancement=True)
        hrp_weights = np.array([hrp_result['weights'][ticker] 
                               for ticker in self.returns_data.columns])
        hrp_return = hrp_result['expected_return']
        
        # Get min variance and max return portfolios
        min_var = self.find_minimum_variance_portfolio()
        max_sharpe = self.optimize_portfolio(maximize_sharpe=True)
        
        # Create frontier points
        frontier_points = []
        
        # Generate target returns
        min_return = min(hrp_return, min_var['expected_return'])
        max_return = max_sharpe['expected_return']
        target_returns = np.linspace(min_return, max_return, n_points)
        
        for target_return in target_returns:
            try:
                # Get mean-variance weights for this target return
                mv_result = self.optimize_portfolio(
                    target_return=target_return,
                    maximize_sharpe=False
                )
                mv_weights = np.array([mv_result['weights'][ticker] 
                                     for ticker in self.returns_data.columns])
                
                # Blend HRP and MV weights
                # More HRP weight for lower returns (more conservative)
                # More MV weight for higher returns (more aggressive)
                blend_factor = (target_return - min_return) / (max_return - min_return)
                blended_weights = (1 - blend_factor) * hrp_weights + blend_factor * mv_weights
                
                # Normalize
                blended_weights = blended_weights / blended_weights.sum()
                
                # Calculate metrics
                ret, vol, sharpe = self.calculate_portfolio_metrics(blended_weights)
                
                frontier_points.append({
                    'target_return': target_return,
                    'actual_return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'hrp_weight': 1 - blend_factor,
                    'mv_weight': blend_factor,
                    'method': 'HRP-MV Blend'
                })
                
            except:
                continue
        
        # Add pure HRP and pure MV points
        frontier_points.append({
            'target_return': hrp_return,
            'actual_return': hrp_return,
            'volatility': hrp_result['volatility'],
            'sharpe_ratio': hrp_result['sharpe_ratio'],
            'hrp_weight': 1.0,
            'mv_weight': 0.0,
            'method': 'Pure HRP'
        })
        
        frontier_points.append({
            'target_return': max_sharpe['expected_return'],
            'actual_return': max_sharpe['expected_return'],
            'volatility': max_sharpe['volatility'],
            'sharpe_ratio': max_sharpe['sharpe_ratio'],
            'hrp_weight': 0.0,
            'mv_weight': 1.0,
            'method': 'Pure MV'
        })
        
        return pd.DataFrame(frontier_points).sort_values('volatility')
    
    def recommend_strategy(self, 
                          risk_tolerance: str = 'moderate',
                          market_regime: str = 'normal') -> Dict:
        """
        Recommend portfolio strategy based on risk tolerance and market regime.
        
        Args:
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            market_regime: 'normal', 'volatile', 'trending'
            
        Returns:
            Recommended portfolio with explanation
        """
        recommendations = {
            'conservative': {
                'normal': {'hrp_weight': 0.7, 'mv_weight': 0.3},
                'volatile': {'hrp_weight': 0.9, 'mv_weight': 0.1},
                'trending': {'hrp_weight': 0.6, 'mv_weight': 0.4}
            },
            'moderate': {
                'normal': {'hrp_weight': 0.5, 'mv_weight': 0.5},
                'volatile': {'hrp_weight': 0.7, 'mv_weight': 0.3},
                'trending': {'hrp_weight': 0.4, 'mv_weight': 0.6}
            },
            'aggressive': {
                'normal': {'hrp_weight': 0.3, 'mv_weight': 0.7},
                'volatile': {'hrp_weight': 0.5, 'mv_weight': 0.5},
                'trending': {'hrp_weight': 0.2, 'mv_weight': 0.8}
            }
        }
        
        # Get blend weights
        blend = recommendations[risk_tolerance][market_regime]
        
        # Get HRP and MV portfolios
        hrp = self.optimize_with_hrp(use_ml_enhancement=True)
        mv = self.optimize_portfolio(maximize_sharpe=True)
        
        # Blend weights
        final_weights = {}
        for ticker in self.returns_data.columns:
            final_weights[ticker] = (
                blend['hrp_weight'] * hrp['weights'][ticker] +
                blend['mv_weight'] * mv['weights'][ticker]
            )
        
        # Calculate metrics
        weights_array = np.array([final_weights[ticker] 
                                for ticker in self.returns_data.columns])
        ret, vol, sharpe = self.calculate_portfolio_metrics(weights_array)
        
        return {
            'strategy': f"{risk_tolerance.capitalize()} - {market_regime.capitalize()} Market",
            'weights': final_weights,
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'hrp_contribution': blend['hrp_weight'],
            'mv_contribution': blend['mv_weight'],
            'explanation': self._generate_strategy_explanation(
                risk_tolerance, market_regime, blend, ret, vol
            )
        }
    
    def _generate_strategy_explanation(self, 
                                     risk_tolerance: str,
                                     market_regime: str,
                                     blend: Dict,
                                     ret: float,
                                     vol: float) -> str:
        """Generate explanation for recommended strategy."""
        explanation = f"""
        Portfolio Strategy: {risk_tolerance.capitalize()} investor in {market_regime} market conditions
        
        Allocation Method:
        - {blend['hrp_weight']*100:.0f}% Hierarchical Risk Parity (HRP)
        - {blend['mv_weight']*100:.0f}% Mean-Variance Optimization
        
        Expected Performance:
        - Annual Return: {ret*100:.1f}%
        - Annual Volatility: {vol*100:.1f}%
        - Risk-Adjusted Return: {ret/vol:.2f}
        
        Rationale:
        """
        
        if market_regime == 'volatile':
            explanation += """
        In volatile markets, we emphasize HRP for its robustness to estimation errors
        and better downside protection. HRP doesn't rely on expected returns, making
        it more stable when correlations are unstable."""
        elif market_regime == 'trending':
            explanation += """
        In trending markets, mean-variance optimization can better capture momentum
        and optimize for higher returns. We increase MV allocation while maintaining
        some HRP for diversification."""
        else:
            explanation += """
        In normal market conditions, we balance both approaches to capture the benefits
        of each: HRP's robustness and MV's return optimization."""
        
        if risk_tolerance == 'conservative':
            explanation += """
        
        As a conservative investor, the portfolio prioritizes capital preservation
        and stable returns over maximum growth."""
        elif risk_tolerance == 'aggressive':
            explanation += """
        
        As an aggressive investor, the portfolio targets higher returns while
        accepting increased volatility."""
        
        return explanation
        """
        Provide mathematical insights about the portfolio based on efficient frontier theory.
        
        Returns insights including:
        - Position relative to efficient frontier
        - Correlation analysis between assets
        - Diversification metrics
        - Mathematical properties (η, ξ, γ parameters)
        """
        # Convert weights to array
        weights_array = np.array([current_weights.get(ticker, 0) 
                                 for ticker in self.returns_data.columns])
        weights_array = weights_array / weights_array.sum()
        
        # Calculate current metrics
        current_return, current_vol, current_sharpe = self.calculate_portfolio_metrics(weights_array)
        
        # Find efficient portfolios
        min_var = self.find_minimum_variance_portfolio()
        max_sharpe = self.find_maximum_sharpe_portfolio()
        
        # Calculate correlation matrix
        corr_matrix = self.returns_data.corr()
        
        # Average correlation (diversification measure)
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        avg_correlation = corr_matrix.values[mask].mean()
        
        # Distance from efficient frontier
        # Find the efficient portfolio with same return
        efficient_same_return = self.optimize_portfolio(
            target_return=current_return,
            maximize_sharpe=False
        )
        
        # Inefficiency ratio
        inefficiency_ratio = current_vol / efficient_same_return['volatility'] - 1
        
        insights = {
            'mathematical_parameters': {
                'eta': self.eta,
                'xi': self.xi,
                'gamma': self.gamma,
                'determinant': np.linalg.det(self.cov_matrix)
            },
            'portfolio_characteristics': {
                'current_return': current_return,
                'current_volatility': current_vol,
                'current_sharpe': current_sharpe,
                'min_variance_return': min_var['expected_return'],
                'min_variance_vol': min_var['volatility'],
                'max_sharpe_return': max_sharpe['expected_return'],
                'max_sharpe_vol': max_sharpe['volatility'],
                'max_sharpe_ratio': max_sharpe['sharpe_ratio']
            },
            'efficiency_analysis': {
                'inefficiency_ratio': inefficiency_ratio,
                'excess_volatility': current_vol - efficient_same_return['volatility'],
                'sharpe_gap': max_sharpe['sharpe_ratio'] - current_sharpe,
                'is_efficient': inefficiency_ratio < 0.01  # Within 1% of frontier
            },
            'diversification_metrics': {
                'average_correlation': avg_correlation,
                'effective_assets': 1 / np.sum(weights_array**2),  # Herfindahl index
                'max_weight': np.max(weights_array),
                'concentration_risk': np.sum(weights_array**2)
            },
            'recommendations': {
                'primary': 'Move to max Sharpe portfolio' if current_sharpe < max_sharpe['sharpe_ratio'] * 0.9 
                          else 'Portfolio is near optimal',
                'risk_reduction': f"Can reduce volatility by {inefficiency_ratio:.1%} at same return level",
                'correlation_warning': 'High correlation between assets' if avg_correlation > 0.7 else 'Good diversification'
            }
        }
        
        return insights
        """
        Provide sector-based allocation recommendations.
        
        Args:
            sector_mapping: Dictionary mapping tickers to sectors
            current_weights: Current portfolio weights
            
        Returns:
            Sector allocation recommendations
        """
        # Get optimal portfolio
        optimal = self.optimize_portfolio(maximize_sharpe=True)
        
        # Calculate sector allocations
        current_sectors = {}
        optimal_sectors = {}
        
        for ticker, sector in sector_mapping.items():
            if ticker in current_weights:
                current_sectors[sector] = current_sectors.get(sector, 0) + current_weights[ticker]
            if ticker in optimal['weights']:
                optimal_sectors[sector] = optimal_sectors.get(sector, 0) + optimal['weights'][ticker]
        
        # Generate recommendations
        sector_recommendations = []
        
        for sector in set(list(current_sectors.keys()) + list(optimal_sectors.keys())):
            current = current_sectors.get(sector, 0)
            optimal = optimal_sectors.get(sector, 0)
            change = optimal - current
            
            if abs(change) > 0.01:  # 1% threshold
                sector_recommendations.append({
                    'sector': sector,
                    'current_allocation': current,
                    'recommended_allocation': optimal,
                    'change': change,
                    'action': 'INCREASE' if change > 0 else 'DECREASE'
                })
        
        return {
            'sector_recommendations': sorted(
                sector_recommendations, 
                key=lambda x: abs(x['change']), 
                reverse=True
            ),
            'current_sector_allocation': current_sectors,
            'optimal_sector_allocation': optimal_sectors
        }


# Example usage and API interface
class ElektraPortfolioAPI:
    """
    API wrapper for easy integration with Elektra app.
    Supports both traditional optimization and HRP with ML enhancements.
    """
    
    def __init__(self):
        from portfolio_optimizer import PortfolioOptimizer  # Import the class if defined elsewhere
        self.optimizer = PortfolioOptimizer()
        
    def analyze_portfolio(self, 
                         price_history: pd.DataFrame,
                         current_holdings: Dict[str, float],
                         sector_map: Optional[Dict[str, str]] = None,
                         optimization_method: str = 'auto',
                         risk_tolerance: str = 'moderate') -> Dict:
        """
        Main API endpoint for portfolio analysis with multiple optimization methods.
        
        Args:
            price_history: Historical prices DataFrame
            current_holdings: Dict of ticker: weight
            sector_map: Optional sector mapping
            optimization_method: 'mv', 'hrp', 'hrp_ml', 'auto'
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Complete analysis with recommendations
        """
        # Load data
        self.optimizer.load_price_data(price_history)
        
        # Detect market regime
        market_regime = self._detect_market_regime(price_history)
        
        # Choose optimization method
        if optimization_method == 'auto':
            # Auto-select based on market conditions
            if market_regime == 'volatile':
                optimization_method = 'hrp_ml'
            else:
                optimization_method = 'blend'
        
        # Get optimized portfolio
        if optimization_method == 'mv':
            optimal = self.optimizer.optimize_portfolio(maximize_sharpe=True)
        elif optimization_method == 'hrp':
            optimal = self.optimizer.optimize_with_hrp(use_ml_enhancement=False)
        elif optimization_method == 'hrp_ml':
            optimal = self.optimizer.optimize_with_hrp(use_ml_enhancement=True)
        else:  # blend
            optimal = self.optimizer.recommend_strategy(risk_tolerance, market_regime)
        
        # Analyze current portfolio
        analysis = self.optimizer.analyze_current_portfolio(current_holdings)
        
        # Get comparison of methods
        comparison = self.optimizer.compare_optimization_methods()
        
        # Get rebalancing recommendations
        rebalancing = self.optimizer.recommend_rebalancing(
            current_holdings, 
            target_weights=optimal.get('weights', optimal['weights'])
        )
        
        # Get efficient frontier (including HRP blend)
        frontier = self.optimizer.create_hrp_efficient_frontier(50)
        
        # Prepare response
        response = {
            'current_portfolio': analysis['current_metrics'],
            'optimal_portfolio': {
                'method': optimal.get('method', optimal.get('strategy', 'Optimized')),
                'weights': optimal['weights'],
                'expected_return': optimal['expected_return'],
                'volatility': optimal['volatility'],
                'sharpe_ratio': optimal['sharpe_ratio']
            },
            'efficiency_score': 100 - analysis['inefficiency_percentage'],
            'rebalancing_recommendations': rebalancing['recommendations'][:5],
            'efficient_frontier': frontier.to_dict('records'),
            'method_comparison': comparison.to_dict('records'),
            'improvement_potential': analysis['improvement_potential'],
            'market_regime': market_regime,
            'risk_profile': risk_tolerance
        }
        
        # Add sector analysis if mapping provided
        if sector_map:
            sector_analysis = self.optimizer.sector_allocation_recommendation(
                sector_map, current_holdings
            )
            response['sector_recommendations'] = sector_analysis['sector_recommendations']
        
        # Add explanation if using blended strategy
        if 'explanation' in optimal:
            response['strategy_explanation'] = optimal['explanation']
        
        return response
    
    def _detect_market_regime(self, prices: pd.DataFrame) -> str:
        """
        Detect current market regime based on recent price behavior.
        
        Returns: 'normal', 'volatile', or 'trending'
        """
        returns = prices.pct_change().dropna()
        
        # Calculate recent metrics (last 30 days)
        recent_vol = returns.tail(30).std().mean() * np.sqrt(252)
        recent_trend = returns.tail(30).mean().mean() * 252
        
        # Calculate historical metrics (last year)
        hist_vol = returns.std().mean() * np.sqrt(252)
        
        # Detect regime
        if recent_vol > hist_vol * 1.5:
            return 'volatile'
        elif abs(recent_trend) > 0.20:  # Strong trend (>20% annualized)
            return 'trending'
        else:
            return 'normal'


# Example implementation with HRP and ML enhancements
if __name__ == "__main__":
    # Example usage demonstrating HRP with ML enhancements
    api = ElektraPortfolioAPI()
    
    # Generate realistic sample data
    dates = pd.date_range('2021-01-01', '2024-01-01', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'XOM']
    
    # Simulate realistic price data with different characteristics
    np.random.seed(42)
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Asset characteristics
    mean_returns = np.array([0.15, 0.12, 0.14, 0.10, 0.20, 0.08, 0.06, 0.11])
    volatilities = np.array([0.20, 0.25, 0.18, 0.22, 0.35, 0.15, 0.12, 0.28])
    
    # Correlation structure with clusters (tech stocks correlated, etc.)
    correlation = np.array([
        [1.00, 0.65, 0.70, 0.60, 0.45, 0.20, 0.15, 0.10],  # AAPL
        [0.65, 1.00, 0.68, 0.62, 0.40, 0.18, 0.12, 0.08],  # GOOGL
        [0.70, 0.68, 1.00, 0.58, 0.42, 0.22, 0.18, 0.12],  # MSFT
        [0.60, 0.62, 0.58, 1.00, 0.38, 0.15, 0.10, 0.05],  # AMZN
        [0.45, 0.40, 0.42, 0.38, 1.00, 0.10, 0.08, 0.15],  # TSLA
        [0.20, 0.18, 0.22, 0.15, 0.10, 1.00, 0.40, 0.25],  # JPM
        [0.15, 0.12, 0.18, 0.10, 0.08, 0.40, 1.00, 0.20],  # JNJ
        [0.10, 0.08, 0.12, 0.05, 0.15, 0.25, 0.20, 1.00]   # XOM
    ])
    
    # Generate covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Generate returns with regime changes
    returns = []
    regime_changes = [0, n_days//3, 2*n_days//3, n_days]
    regimes = ['normal', 'volatile', 'trending']
    
    for i in range(len(regimes)):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i+1]
        n_regime_days = end_idx - start_idx
        
        # Adjust covariance for regime
        if regimes[i] == 'volatile':
            regime_cov = cov_matrix * 1.5
        elif regimes[i] == 'trending':
            regime_cov = cov_matrix * 0.8
        else:
            regime_cov = cov_matrix
        
        # Generate returns for this regime
        regime_returns = np.random.multivariate_normal(
            mean_returns/252, regime_cov/252, n_regime_days
        )
        returns.append(regime_returns)
    
    returns = np.vstack(returns)
    
    # Convert to prices
    price_data = pd.DataFrame(
        100 * (1 + returns).cumprod(axis=0),
        index=dates,
        columns=tickers
    )
    
    # Current portfolio (suboptimal allocation)
    current_portfolio = {
        'AAPL': 0.25,
        'GOOGL': 0.20,
        'MSFT': 0.20,
        'AMZN': 0.15,
        'TSLA': 0.10,
        'JPM': 0.05,
        'JNJ': 0.03,
        'XOM': 0.02
    }
    
    # Sector mapping
    sectors = {
        'AAPL': 'Technology',
        'GOOGL': 'Technology',
        'MSFT': 'Technology',
        'AMZN': 'Consumer',
        'TSLA': 'Automotive',
        'JPM': 'Financial',
        'JNJ': 'Healthcare',
        'XOM': 'Energy'
    }
    
    print("=== Hierarchical Risk Parity Portfolio Optimization ===\n")
    
    # Test different optimization methods
    for method in ['mv', 'hrp', 'hrp_ml', 'auto']:
        print(f"\n--- Testing {method.upper()} Method ---")
        results = api.analyze_portfolio(
            price_data, 
            current_portfolio, 
            sectors,
            optimization_method=method,
            risk_tolerance='moderate'
        )
        
        print(f"Efficiency Score: {results['efficiency_score']:.1f}%")
        print(f"Optimal Method: {results['optimal_portfolio']['method']}")
        print(f"Expected Return: {results['optimal_portfolio']['expected_return']*100:.1f}%")
        print(f"Volatility: {results['optimal_portfolio']['volatility']*100:.1f}%")
        print(f"Sharpe Ratio: {results['optimal_portfolio']['sharpe_ratio']:.3f}")
    
    # Compare all methods
    print("\n=== Method Comparison ===")
    comparison_df = pd.DataFrame(results['method_comparison'])
    print(comparison_df.to_string(index=False))
    
    # Show HRP clustering visualization
    print("\n=== HRP Asset Clustering ===")
    optimizer = api.optimizer
    hrp_weights = optimizer.optimize_with_hrp(use_ml_enhancement=True)
    
    print("\nAsset Clusters (by correlation):")
    # The linkage matrix shows how assets are grouped
    linkage = optimizer.hrp_model.linkage_matrix
    
    # Identify main clusters
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage, t=0.5, criterion='distance')
    
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: Cluster {clusters[i]}, Weight: {hrp_weights['weights'][ticker]*100:.1f}%")
    
    # Test different market conditions
    print("\n=== Strategy Recommendations by Market Regime ===")
    for regime in ['normal', 'volatile', 'trending']:
        for risk_tol in ['conservative', 'moderate', 'aggressive']:
            strategy = optimizer.recommend_strategy(risk_tol, regime)
            print(f"\n{risk_tol.capitalize()} investor in {regime} market:")
            print(f"  - HRP weight: {strategy['hrp_contribution']*100:.0f}%")
            print(f"  - MV weight: {strategy['mv_contribution']*100:.0f}%")
            print(f"  - Expected Sharpe: {strategy['sharpe_ratio']:.3f}")
    
    print("\n=== Key Advantages of HRP with ML ===")
    print("1. More stable weights in volatile markets")
    print("2. Better out-of-sample performance")
    print("3. Natural diversification through clustering")
    print("4. No matrix inversion (handles more assets)")
    print("5. ML adapts to changing market regimes")
    
    # Show frontier comparison
    print("\n=== Efficient Frontier Comparison ===")
    frontier_df = pd.DataFrame(results['efficient_frontier'])
    print(f"Traditional Frontier Points: {len(frontier_df[frontier_df['method'] == 'HRP-MV Blend'])}")
    print(f"Pure HRP Return: {frontier_df[frontier_df['method'] == 'Pure HRP']['actual_return'].values[0]*100:.1f}%")
    print(f"Pure MV Return: {frontier_df[frontier_df['method'] == 'Pure MV']['actual_return'].values[0]*100:.1f}%")