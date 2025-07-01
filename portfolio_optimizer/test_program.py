# Quick test to ensure everything works
from portfolio_optimizer import PortfolioOptimizer
import pandas as pd
import numpy as np

print("Testing portfolio optimizer...")

# Create sample data
dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
prices = pd.DataFrame({
    'AAPL': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
    'GOOGL': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
    'MSFT': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
}, index=dates)

# Test optimizer
optimizer = PortfolioOptimizer()
optimizer.load_price_data(prices)

# Test optimization
result = optimizer.optimize_portfolio(maximize_sharpe=True)
print("✓ Optimization successful!")
print(f"Weights: {result['weights']}")
print(f"Expected Return: {result['expected_return']*100:.1f}%")
print(f"Volatility: {result['volatility']*100:.1f}%")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
