import os
import subprocess
import re
import time
from tqdm import tqdm

def test_single_day(fast, slow, mult, day_number):
    command = ["prosperity3bt", r"/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/Trader.py",
               f"1-{day_number}", "--parameters", f"{fast},{slow},{mult}"]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr

    # Use regex to extract the log file path
    match = re.search(r'backtests[\\/][\d\-_.]+\.log', full_output)
    if match:
        log_path = match.group(0)
        #print("Log file path:", log_path, print(type(log_path)))

        # Extract profit information from the output
        profit_pattern = r"KELP: ([\d,]+)\nRAINFOREST_RESIN: ([\d,]+)\nSQUID_INK: ([\-\d,]+)\nTotal profit: ([\-\d,]+)"
        profit_match = re.search(profit_pattern, full_output)

        if profit_match:
            kelp_profit = int(profit_match.group(1).replace(',', ''))
            resin_profit = int(profit_match.group(2).replace(',', ''))
            squid_ink_profit = int(profit_match.group(3).replace(',', ''))
            total_profit = int(profit_match.group(4).replace(',', ''))

            #print(f"Extracted profits:")
            #print(f"KELP: {kelp_profit}")
            #print(f"RAINFOREST_RESIN: {resin_profit}")
            #print(f"SQUID_INK: {squid_ink_profit}")
            #print(f"Total: {total_profit}")


            # Rename the log file to include the parameters and the profit in a custom folder
            custom_log_path = f"/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/backtests/max_tester_{fast}_{slow}_{mult}_pnl_{total_profit}_{day_number}.log"

            #print(custom_log_path)
            log_path = '/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/' + log_path

            os.rename(log_path, custom_log_path)

            return total_profit
        else:
            #print("Could not extract profit information from the output")
            return 0
    return 0


def test_full_round(fast, slow, mult):
    command = ["prosperity3bt", r"/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/Trader.py", "1",
               "--parameters", f"{fast},{slow},{mult}"]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Combine stdout and stderr (in case the log path is in either)
    full_output = result.stdout + "\n" + result.stderr
    #print(full_output)

    # Use regex to extract the log file path
    pattern = r'backtests/[\\/]?[\d\-_.]+\.log'
    match = re.search(pattern, full_output)
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
            custom_log_path = f"/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/backtests/max_tester_{fast}_{slow}_{mult}_pnl_{total_profit}.log"
            print(custom_log_path)
            log_path = '/Users/maximesolere/Library/Mobile Documents/com~apple~CloudDocs/Maxime/RIBOT/Git/Tutorial/Max/' + log_path

            os.rename(log_path, custom_log_path)

            return total_profit
        else:
            print("Could not extract total profit information from the output")
            return 0
    return 0


# Choose which test to run
test_type = "single_day"  # Change to "single_day" or "full_round"
day_to_test = -2  # Only used if test_type is "single_day"

max_profit = 0
max_profit_parameters = (0, 0, 0)


span_fast = [100, 150, 200]
span_slow = [1000, 1500, 2500]
multiple = [0.2, 0.6, 1]

for i in (span_fast):
    for j in (span_slow):
        for k in (multiple):
            total_profit = 0
            if test_type == "single_day":
                for day in [-2, -1, 0]:
                    profit = test_single_day(i,j,k,day)
                    print(profit)
                    if profit < 0:
                        total_profit += profit
                    total_profit += profit
                print(i,j,k,day,total_profit)

            else:  # full_round
                total_profit = test_full_round(i,j, k)

            if total_profit > max_profit:
                max_profit = total_profit
                max_profit_parameters = (i,j, k)

print(f"Max profit: {max_profit}")
print(f"Max profit parameters: {max_profit_parameters}")
