
import os
import subprocess
import re

max_profit = -1e10

for i in range(2,3):
    i_exp = 10**i

    command = ["prosperity3bt", r"/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/Trader.py", "1","--parameters",f"{i_exp}"]
    # In the case of multiple parameters we have to put f"{i},{j},{k}" with NO SPACES after the comma, 
    # If there is no parameter we can delete after "0" and leave the list until that point

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr
    print(full_output)

    # Use regex to extract the log file path
    match = re.search(r'backtests[\\/][\d\-_.]+\.log', full_output)
    print(match)
    if match:
        print(99999)
        log_path = match.group(0)
        print("Log file path:", log_path, print(type(log_path)))
            # Extract profit information from the output
        profit_pattern = r'KELP: ([\d,]+)\nRAINFOREST_RESIN: ([\d,]+)\nTotal profit: ([\d,]+)'
        profit_match = re.search(profit_pattern, full_output)
        
        if profit_match:
            kelp_profit = int(profit_match.group(1).replace(',', ''))
            resin_profit = int(profit_match.group(2).replace(',', ''))
            total_profit = int(profit_match.group(3).replace(',', ''))
            
            print(f"Extracted profits:")
            print(f"KELP: {kelp_profit}")
            print(f"RAINFOREST_RESIN: {resin_profit}")
            print(f"Total: {total_profit}")
        else:
            print("Could not extract profit information from the output")

        print(total_profit)
        if total_profit > max_profit:
            max_profit = total_profit
            max_profit_parameters = i
        # Rename the log file to include the parameters and the profit in a custom folder
        custom_log_path = log_path.replace(log_path.split('/')[-1], f"Tutorial/backtests/max_tester_parameters_{i}_pnl_{total_profit}.log")
        print(custom_log_path)
        os.rename(log_path, custom_log_path)
 
print(f"Max profit: {max_profit}")
print(f"Max profit parameters: {max_profit_parameters}")
