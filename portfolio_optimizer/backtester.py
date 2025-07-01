import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class PortfolioBacktester:
    """Backtest portfolio optimization strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def backtest_strategy(self,
                         optimizer,
                         price_data: pd.DataFrame,
                         strategy: str = 'hrp_ml',
                         rebalance_frequency: str = 'monthly',
                         lookback_period: int = 252,
                         transaction_cost: float = 0.001) -> Dict:
        """
        Backtest a portfolio strategy
        
        Args:
            optimizer: Portfolio optimizer instance
            price_data: Historical prices
            strategy: 'mv', 'hrp', 'hrp_ml', 'equal_weight'
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            lookback_period: Days to use for optimization
            transaction_cost: Cost per trade as fraction
            
        Returns:
            Backtest results dictionary
        """
        # Initialize
        dates = price_data.index[lookback_period:]
        portfolio_values = []
        weights_history = []
        turnover_history = []
        
        # Set rebalance dates
        rebalance_dates = self._get_rebalance_dates(dates, rebalance_frequency)
        
        # Initial equal weights
        n_assets = len(price_data.columns)
        current_weights = np.ones(n_assets) / n_assets
        current_value = self.initial_capital
        
        print(f"Backtesting {strategy} strategy...")
        print(f"Rebalancing {rebalance_frequency}")
        print(f"Period: {dates[0].date()} to {dates[-1].date()}")
        
        # Main backtest loop
        for i, date in enumerate(dates):
            # Get returns for the day
            if i > 0:
                daily_returns = (price_data.loc[date] / price_data.loc[dates[i-1]] - 1).values
                
                # Update portfolio value
                portfolio_return = np.dot(current_weights, daily_returns)
                current_value *= (1 + portfolio_return)
            
            # Record portfolio value
            portfolio_values.append(current_value)
            
            # Rebalance if needed
            if date in rebalance_dates:
                # Get historical data for optimization
                hist_end = dates[i-1] if i > 0 else date
                hist_start_idx = max(0, i - lookback_period)
                hist_data = price_data.iloc[hist_start_idx:i+1]
                
                # Optimize weights
                try:
                    optimizer.load_price_data(hist_data)
                    
                    if strategy == 'mv':
                        result = optimizer.optimize_portfolio(maximize_sharpe=True)
                        new_weights = np.array([result['weights'][ticker] 
                                              for ticker in price_data.columns])
                    elif strategy == 'hrp':
                        result = optimizer.optimize_with_hrp(use_ml_enhancement=False)
                        new_weights = np.array([result['weights'][ticker] 
                                              for ticker in price_data.columns])
                    elif strategy == 'hrp_ml':
                        result = optimizer.optimize_with_hrp(use_ml_enhancement=True)
                        new_weights = np.array([result['weights'][ticker] 
                                              for ticker in price_data.columns])
                    else:  # equal_weight
                        new_weights = np.ones(n_assets) / n_assets
                    
                    # Calculate turnover and transaction costs
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    transaction_cost_impact = turnover * transaction_cost
                    current_value *= (1 - transaction_cost_impact)
                    
                    # Update weights
                    current_weights = new_weights
                    weights_history.append((date, current_weights.copy()))
                    turnover_history.append((date, turnover))
                    
                except Exception as e:
                    print(f"Optimization failed on {date}: {e}")
                    # Keep current weights
        
        # Convert to series
        portfolio_values = pd.Series(portfolio_values, index=dates)
        
        # Calculate performance metrics
        returns = portfolio_values.pct_change().dropna()
        
        results = {
            'strategy': strategy,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'weights_history': weights_history,
            'turnover_history': turnover_history,
            'metrics': self._calculate_metrics(portfolio_values, returns),
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] / self.initial_capital - 1)
        }
        
        return results
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List:
        """Get rebalancing dates based on frequency"""
        if frequency == 'daily':
            return dates.tolist()
        elif frequency == 'weekly':
            return dates[dates.weekday == 0].tolist()  # Mondays
        elif frequency == 'monthly':
            return dates[dates.is_month_start].tolist()
        elif frequency == 'quarterly':
            return dates[(dates.month.isin([1,4,7,10])) & 
                        (dates.is_month_start)].tolist()
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
    
    def _calculate_metrics(self, portfolio_values: pd.Series, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        # Annual metrics
        days_per_year = 252
        years = len(returns) / days_per_year
        
        # Returns
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # Risk
        annual_volatility = returns.std() * np.sqrt(days_per_year)
        
        # Risk-adjusted
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_return': total_return
        }
    
    def compare_strategies(self, 
                          optimizer,
                          price_data: pd.DataFrame,
                          strategies: List[str] = ['mv', 'hrp', 'hrp_ml', 'equal_weight'],
                          **kwargs) -> pd.DataFrame:
        """Compare multiple strategies"""
        results = {}
        
        for strategy in strategies:
            results[strategy] = self.backtest_strategy(
                optimizer, price_data, strategy, **kwargs
            )
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            strategy: {
                'Annual Return': result['metrics']['annual_return'] * 100,
                'Annual Volatility': result['metrics']['annual_volatility'] * 100,
                'Sharpe Ratio': result['metrics']['sharpe_ratio'],
                'Max Drawdown': result['metrics']['max_drawdown'] * 100,
                'Calmar Ratio': result['metrics']['calmar_ratio'],
                'Final Value': result['final_value']
            }
            for strategy, result in results.items()
        }).T
        
        self.results = results
        return comparison
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.results:
            print("No results to plot. Run compare_strategies first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio values
        ax1 = axes[0, 0]
        for strategy, result in self.results.items():
            ax1.plot(result['portfolio_values'], label=strategy)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdowns
        ax2 = axes[0, 1]
        for strategy, result in self.results.items():
            rolling_max = result['portfolio_values'].expanding().max()
            drawdown = (result['portfolio_values'] - rolling_max) / rolling_max * 100
            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=strategy)
        ax2.set_title('Drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        ax3 = axes[1, 0]
        for strategy, result in self.results.items():
            rolling_sharpe = (result['returns'].rolling(252).mean() / 
                            result['returns'].rolling(252).std() * np.sqrt(252))
            ax3.plot(rolling_sharpe, label=strategy)
        ax3.set_title('Rolling 1-Year Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Weight evolution (for one strategy)
        ax4 = axes[1, 1]
        # Plot weight evolution for HRP ML
        if 'hrp_ml' in self.results:
            weights_history = self.results['hrp_ml']['weights_history']
            if weights_history:
                dates = [w[0] for w in weights_history]
                weights = np.array([w[1] for w in weights_history])
                for i in range(weights.shape[1]):
                    ax4.plot(dates, weights[:, i] * 100, 
                           label=f'Asset {i+1}')
        ax4.set_title('HRP ML Weight Evolution')
        ax4.set_ylabel('Weight (%)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
