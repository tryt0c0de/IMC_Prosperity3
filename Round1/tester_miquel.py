
import os
import subprocess
import re

def test_single_day(rolling_window, day_number):
    command = ["prosperity3bt", r"C:\Users\Usuario1\Desktop\quant\IMC_Prosperity\Round1\trader_miquel.py", f"1--{day_number}", "--parameters", f"{rolling_window}"]
    
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr


    # Use regex to extract the log file path
    match = re.search(r'backtests[\\/][\d\-_.]+\.log', full_output)
    if match:
        log_path = match.group(0)
        print("Log file path:", log_path, print(type(log_path)))
        
        # Extract profit information from the output
        profit_pattern = r"KELP: ([\d,]+)\nRAINFOREST_RESIN: ([\d,]+)\nSQUID_INK: ([\-\d,]+)\nTotal profit: ([\-\d,]+)"
        profit_match = re.search(profit_pattern, full_output)
        
        if profit_match:
            kelp_profit = int(profit_match.group(1).replace(',', ''))
            resin_profit = int(profit_match.group(2).replace(',', ''))
            squid_ink_profit = int(profit_match.group(3).replace(',', ''))
            total_profit = int(profit_match.group(4).replace(',', ''))
            
            print(f"Extracted profits:")
            print(f"KELP: {kelp_profit}")
            print(f"RAINFOREST_RESIN: {resin_profit}")
            print(f"SQUID_INK: {squid_ink_profit}")
            print(f"Total: {total_profit}")
            
            # Rename the log file to include the parameters and the profit in a custom folder
            custom_log_path = log_path.replace(log_path.split('/')[-1], f"Round1/backtests/miquel_tester_day_{day_number}_rolling_window_{rolling_window}_pnl_{total_profit}.log")
            print(custom_log_path)
            os.rename(log_path, custom_log_path)
            
            return total_profit
        else:
            print("Could not extract profit information from the output")
            return 0
    return 0

def test_full_round(rolling_window):
    command = ["prosperity3bt", r"C:\Users\Usuario1\Desktop\quant\IMC_Prosperity\Round1\trader_miquel.py", "1", "--parameters", f"{rolling_window}"]
    
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr


    # Use regex to extract the log file path
    match = re.search(r'backtests[\\/][\d\-_.]+\.log', full_output)
    if match:
        log_path = match.group(0)
        print("Log file path:", log_path, print(type(log_path)))
        
        # Extract total profit from the profit summary at the end (last occurrence)
        total_profit_pattern = r"Profit summary:.*?Total profit: ([\-\d,]+)"
        total_profit_match = re.search(total_profit_pattern, full_output, re.DOTALL)
        
        if total_profit_match:
            total_profit = int(total_profit_match.group(1).replace(',', ''))
            print(f"Total profit across all days: {total_profit}")
            
            # Rename the log file to include the parameters and the profit in a custom folder
            custom_log_path = log_path.replace(log_path.split('/')[-1], f"Round1/backtests/miquel_tester_full_round_rolling_window_{rolling_window}_pnl_{total_profit}.log")
            print(custom_log_path)
            os.rename(log_path, custom_log_path)
            
            return total_profit
        else:
            print("Could not extract total profit information from the output")
            return 0
    return 0

# Choose which test to run
test_type = "full_round"  # Change to "single_day" or "full_round"
day_to_test = -2  # Only used if test_type is "single_day"

max_profit = 0
max_profit_parameters = 0

rolling_windows_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

for i in rolling_windows_to_test:
    if test_type == "single_day":
        total_profit = test_single_day(i, day_to_test)
    else:  # full_round
        total_profit = test_full_round(i)
    
    if total_profit > max_profit:
        max_profit = total_profit
        max_profit_parameters = i

print(f"Max profit: {max_profit}")
print(f"Max profit parameters: {max_profit_parameters}")
