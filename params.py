import numpy as np

current_system = 'z400'
asset_name = "Quickler"
asset_duration = 5 # in seconds

if current_system == 'rested':
    # Check browser at 100% zoom level. Window at half.
    if asset_name == 'Quickler':
        olymp_hr = (145,-377)
        olymp_up = (2293,-770) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (2293,-681) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (266,-475)
        olymp_day = None
        olymp_account_switch = None
        olymp_demo_account = None
        olymp_usd_account = None
        olymp_amount = None
    if asset_name == 'EURUSD':
        olymp_hr = (164,1100)
        olymp_up = (1800,683) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (1803,763) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (280,1001)
        olymp_day = None
    olymp_browser = (279, -1275)
    olymp_trade_record = (1208, -806)
    olymp_first_trade_record = (659,-649)
    click_start = (136, -241)
    quote_interval_pricewait = 0.75
if current_system == 'z400':
    # Check browser at 100% zoom level
    if asset_name == 'Quickler':
        olymp_hr = (143,963)
        olymp_up = (1812,578) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (1816,660) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (277,866)
        olymp_day = None
        olymp_account_switch = None
        olymp_demo_account = None
        olymp_usd_account = None
        olymp_amount = None
    if asset_name == 'EURUSD':
        olymp_hr = ()
        olymp_up = () # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = () # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = ()
        olymp_day = None
    olymp_browser = (205,100)
    olymp_trade_record = (937,400)
    olymp_first_trade_record = (434,549)
    click_start = (140,1027)
    quote_interval_pricewait = 2

# Data file location
# data_store_location = 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/'
data_store_location = '//DESKTOP-RESTED/data'

# Print coordinate of mouse position
find_pos = False
position_roll_call = False

# Testing for get_quote
test_get_one = False
test_get_some = False
test_build_dataset_last_t = False

# Testing for analysis
test_load_function = False
test_compute_function = False
test_force_trade = False

# Testing with cross_val_trading.
# Will overwrite files in data/training/*today*
test_cross_val_trading = False
test_cross_val_past = False
test_cross_val_specify_test_range = False

if test_cross_val_past:
    test_hour = '05' ; test_minute = '15' ; test_second = '15'
    test_date = '15022022'

"""
lookback_t: expected to be an integer 
In minutes. Larger values allow for wider range of warm_range.
If set to 2, during trading, note that it is less than 2, more like 1+ mins as we get most current with get one.
If cross validating, likely one will have the full minute of data.
Currently supports max of 60.
Larger values can give lower NRMSE, due to higher variance in the data (see test_nrmse in compute)

warm_range: expected to be a list of integers.
Iterated over to "slide" along time in the data In seconds.
USe -1 to max out on warm, then train and test on as close to current as possible. Must be > 0.
Be careful to note what best_params() function output, w.r.t to this param.
A list of warm_range: np.arange(45,(lookback_t-2)*60,30); warm_range = np.append(warm_range,-1)

train_range: expected to be a list of integers.
How many time points to use for training to predict test_range
NG-RC paper used 10 for this.
Use -1 for both train and warm to set warm to just what we need for delay to process 1st data point, 
    then max out train points to predict for test points given.

delay_range: expected to be a list of integers.
How many time points to look behind for each item in train_range in order to derive weights.
High values, in excess of 50 for this machine, can lead to drastic increase in compute times.

ridge_range: expected to be a list of integers.
Tikhonov regularization. NG-RC paper used 2.5e-6 for this. 

threshold_test_nrmse: expected to be a list of integers.
Used in cross_val_ngrc to filter the results of each compute.
If test_nrmse output is not smaller than this, we do not save the param.
Set higher values to allow more results to show up.

"""
if test_cross_val_specify_test_range:
    test_range = [5]  # In seconds.

if test_cross_val_trading:
    lookback_t = 60
    warm_range = [-1]
    delay_range = range(2,20)
    train_range = [-1]
    ridge_range = [2.5e-6]
    threshold_test_nrmse = [1]

else:
    lookback_t = 60
    warm_range = [-1]
    delay_range = range(2,25)
    train_range = [-1]
    ridge_range = [2.5e-6]
    threshold_test_nrmse = [0.1]

# Trade Execution Params
lookback_t_min = 2 # Only read by compute() when predicting for trade.
total_trade = 10
pred_delta_threshold = 0
interval_typew = 0
traderecord_interval_refresh = 5
random_sleep = False
random_sleep_min = 1
random_sleep_max = 15

# Params for how far ahead to predict
time_betw_execution_end_and_trade_open = 1.5 # Updated after every trade.
time_betw_get_end_and_execution_end = 0.5 # Hardcoded. 0.6 for z400. 0.47 for rested.
# Function to change global timings.
def change_time_onthefly(time_ge=None, time_et=None): # https://is.gd/HqFpNJ
    global time_betw_execution_end_and_trade_open
    global time_betw_get_end_and_execution_end
    if time_ge is not None: time_betw_get_end_and_execution_end = time_ge
    if time_et is not None: time_betw_execution_end_and_trade_open = time_et