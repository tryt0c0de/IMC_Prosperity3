def find_arbitrage(exchange_rates, max_trades=5, start_currency="SeaShells"):
    """
    Find the optimal trading path to maximize profit in the start_currency.
    
    Args:
        exchange_rates: Dictionary with currency pairs as keys and exchange rates as values
        max_trades: Maximum number of trades allowed
        start_currency: The currency we start with and want to maximize
        
    Returns:
        best_path: List of currencies in the optimal trading path
        best_profit: The maximum profit achieved
    """
    currencies = ["Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells"]
    
    # Initialize variables to track best path and profit
    best_profit = 1.0  # Starting with 1 unit of start_currency
    best_path = [start_currency]
    
    # Function to explore all possible paths using DFS
    def dfs(current_currency, current_path, current_value, trades_made):
        nonlocal best_profit, best_path
        
        # If we've made enough trades and returned to start currency, check if this is better
        if trades_made > 0 and current_currency == start_currency:
            if current_value > best_profit:
                best_profit = current_value
                best_path = current_path.copy()
        
        # If we've reached max trades, stop exploring this path
        if trades_made >= max_trades:
            return
            
        # Try converting to each other currency
        for next_currency in currencies:
            if next_currency != current_currency:
                key = (current_currency, next_currency)
                if key in exchange_rates:
                    new_value = current_value * exchange_rates[key]
                    new_path = current_path + [next_currency]
                    dfs(next_currency, new_path, new_value, trades_made + 1)
    
    # Start the DFS from the start currency
    dfs(start_currency, [start_currency], 1.0, 0)
    
    return best_path, best_profit

# Define the exchange rates from the provided table
exchange_rates = {
    ("Snowballs", "Snowballs"): 1,
    ("Snowballs", "Pizza's"): 1.45,
    ("Snowballs", "Silicon Nuggets"): 0.52,
    ("Snowballs", "SeaShells"): 0.72,
    
    ("Pizza's", "Snowballs"): 0.7,
    ("Pizza's", "Pizza's"): 1,
    ("Pizza's", "Silicon Nuggets"): 0.31,
    ("Pizza's", "SeaShells"): 0.48,
    
    ("Silicon Nuggets", "Snowballs"): 1.95,
    ("Silicon Nuggets", "Pizza's"): 3.1,
    ("Silicon Nuggets", "Silicon Nuggets"): 1,
    ("Silicon Nuggets", "SeaShells"): 1.49,
    
    ("SeaShells", "Snowballs"): 1.34,
    ("SeaShells", "Pizza's"): 1.98,
    ("SeaShells", "Silicon Nuggets"): 0.64,
    ("SeaShells", "SeaShells"): 1
}

# Find the best arbitrage path
best_path, best_profit = find_arbitrage(exchange_rates, max_trades=5, start_currency="SeaShells")

# Show the results
print(f"Best path: {' -> '.join(best_path)}")
print(f"Starting with 1 SeaShell, you end with {best_profit:.6f} SeaShells")
print(f"Profit: {(best_profit - 1) * 100:.2f}%")

# Calculate the value after each step for clarity
print("\nStep-by-step calculation:")
value = 1.0
for i in range(len(best_path) - 1):
    from_currency = best_path[i]
    to_currency = best_path[i + 1]
    rate = exchange_rates[(from_currency, to_currency)]
    new_value = value * rate
    print(f"{value:.6f} {from_currency} → {new_value:.6f} {to_currency} (rate: {rate})")
    value = new_value