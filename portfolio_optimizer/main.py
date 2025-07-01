import sys
import warnings
warnings.filterwarnings('ignore')

from portfolio_optimizer import PortfolioOptimizer, ElektraPortfolioAPI
from data_loader import DataLoader
from backtester import PortfolioBacktester
import pandas as pd

def main():
    """Main function to run portfolio optimization and backtesting"""
    
    # Initialize components
    print("=== Portfolio Optimization System ===\n")
    data_loader = DataLoader()
    api = ElektraPortfolioAPI()
    backtester = PortfolioBacktester(initial_capital=100000)
    
    # Define portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'BRK-B']
    
    # Download data
    print("1. Downloading market data...")
    price_data = data_loader.get_price_data(tickers, period='5y')
    sectors = data_loader.get_sector_mapping(tickers)
    
    print(f"\nData summary:")
    print(f"- Tickers: {', '.join(tickers)}")
    print(f"- Period: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    print(f"- Total days: {len(price_data)}")
    
    # Current portfolio (example)
    current_portfolio = {
        ticker: 1/len(tickers) for ticker in tickers  # Equal weight
    }
    
    # Run optimization analysis
    print("\n2. Running portfolio optimization...")
    analysis = api.analyze_portfolio(
        price_data.tail(504),  # Last 2 years for optimization
        current_portfolio,
        sectors,
        optimization_method='auto',
        risk_tolerance='moderate'
    )
    
    print(f"\nOptimization Results:")
    print(f"- Current Efficiency Score: {analysis['efficiency_score']:.1f}%")
    print(f"- Optimal Method: {analysis['optimal_portfolio']['method']}")
    print(f"- Expected Return: {analysis['optimal_portfolio']['expected_return']*100:.1f}%")
    print(f"- Volatility: {analysis['optimal_portfolio']['volatility']*100:.1f}%")
    print(f"- Sharpe Ratio: {analysis['optimal_portfolio']['sharpe_ratio']:.2f}")
    
    # Show method comparison
    print("\n3. Method Comparison:")
    comparison_df = pd.DataFrame(analysis['method_comparison'])
    print(comparison_df.to_string(index=False))
    
    # Run backtest
    print("\n4. Running backtest...")
    backtest_comparison = backtester.compare_strategies(
        api.optimizer,
        price_data,
        strategies=['mv', 'hrp', 'hrp_ml', 'equal_weight'],
        rebalance_frequency='monthly',
        lookback_period=504,  # 2 years
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    print("\nBacktest Results:")
    print(backtest_comparison.round(2).to_string())
    
    # Plot results
    print("\n5. Generating performance charts...")
    backtester.plot_results()
    
    # Show recommendations
    print("\n6. Top Rebalancing Recommendations:")
    for i, rec in enumerate(analysis['rebalancing_recommendations'][:5], 1):
        print(f"{i}. {rec['action']} {rec['ticker']}: "
              f"{rec['current_weight']*100:.1f}% → {rec['recommended_weight']*100:.1f}%")
    
    # Save results
    print("\n7. Saving results...")
    backtest_comparison.to_csv('backtest_results.csv')
    pd.DataFrame(analysis['optimal_portfolio']['weights'], index=[0]).to_csv('optimal_weights.csv')
    print("Results saved to backtest_results.csv and optimal_weights.csv")

if __name__ == "__main__":
    main()
