
import os
import subprocess
import re

max_profit = 0
for i in [5,10,15,20,25,30,35,40,45,50]:
    command = ["prosperity3bt", r"C:\Users\Usuario1\Desktop\quant\IMC_Prosperity\Tutorial\trader_filippo.py", "0","--parameters",f"{i}"]
    # In the case of multiple parameters we have to put f"{i},{j},{k}" with NO SPACES after the comma, 
    # If there is no parameter we can delete after "0" and leave the list until that point

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr

    #print(full_output)

    # Use regex to extract the log file path
    match = re.search(r'backtests[\\/][\d\-_.]+\.log', full_output)
    if match:
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

        if total_profit > max_profit:
            max_profit = total_profit
            max_profit_parameters = i
        # Rename the log file to include the parameters and the profit in a custom folder
        custom_log_path = log_path.replace(log_path.split('/')[-1], f"Tutorial/backtests/filippo_tester_parameters_{i}_pnl_{total_profit}.log")
        print(custom_log_path)
        os.rename(log_path, custom_log_path)
 
print(f"Max profit: {max_profit}")
print(f"Max profit parameters: {max_profit_parameters}")
