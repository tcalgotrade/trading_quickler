import numpy as np

current_system = 'z400'
asset_name = "Quickler"
asset_duration = 5 # in seconds

if current_system == 'rested':
    # Check browser at 100% zoom level. Window at half.
    if asset_name == 'Quickler':
        olymp_hr = (162,973)
        olymp_up = (1812,580) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (1812,659) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (187,887)
        olymp_day = None
        olymp_account_switch = (1610,165)
        olymp_demo_account = (1502,267)
        olymp_usd_account = (1502,396)
        olymp_amount = (1852,320)
    if asset_name == 'EURUSD':
        olymp_hr = (164,1100)
        olymp_up = (1800,683) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (1803,763) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (280,1001)
        olymp_day = None
    olymp_browser = (1722, 22)
    olymp_trade_record = (323,702)
    olymp_first_trade_record = (414,703)
    click_start = (132, 1114)
if current_system == 'z400':
    # Check browser at 100% zoom level
    if asset_name == 'Quickler':
        olymp_hr = (86,633)
        olymp_up = (888,340) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (888,389) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (171,572)
        olymp_day = None
        olymp_account_switch = (689,105)
        olymp_demo_account = (695,179)
        olymp_usd_account = (695,263)
        olymp_amount = ()
    if asset_name == 'EURUSD':
        olymp_hr = ()
        olymp_up = () # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = () # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = ()
        olymp_day = None
    olymp_browser = (521, 614)
    olymp_trade_record = (485,357)
    olymp_first_trade_record = (120,458)
    click_start = (73, 722)

# Data file location
data_store_location = 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/'

# Levers for testing
find_pos = False
test_get_one = False
test_get_some = False
test_build_dataset_last_t = False
test_build_dataset = False
test_load_function = False
test_compute_function = False

# Testing with cross_val_trading.
# Will overwrite files in data/training/*today*
test_cross_val_trading = False
test_cross_val_past = False
test_cross_val_specify_test_range = False
test_force_trade = True

if test_cross_val_past:
    test_hour = '16' ; test_minute = '20' ; test_second = '15'

if test_cross_val_specify_test_range:
    test_range = [7,7.5,8,8.5,9,9.5,10]  # In seconds.
    test_points = [test_range[0] - 1.5, test_range[0], test_range[0] + 1.5]

if test_cross_val_trading:
    lookback_t = 3 # Larger values of lookback_t allows for wider range of warm_range. if =2, note that it is actually more like 1+ mins as we get most current with get one.
    warm_range = np.arange(15,lookback_t*60,30) ; warm_range = np.append(warm_range,-1)  # In seconds. -1 to train and test on as close to current as possible. Must be > 0
    train_range = range(5,15) # In seconds
    delay_range = range(11,30) # In seconds
    ridge_range = np.linspace(0,3e-7,5)
    threshold_test_nrmse = [1] # Set to 1 to allow all to show up
else:
    lookback_t = 5  # Larger lookback_t allows for wider range of warm_range. if =2, note that it is actually more like 1+ mins as we get most current with get one.
    warm_range = np.arange(1,(lookback_t-1)*60,40) ; warm_range = np.append(warm_range,-1) # In seconds. -1 to train and test on as close to current as possible. Must be > 0
    train_range = range(5,10) # In seconds
    delay_range = range(25,30) # In seconds
    ridge_range = [1e-7]
    threshold_test_nrmse = [1] # Set to 1 to allow all to show up

# Trade Execution Params
lookback_t_min = 2 # Only read by compute() when predicting for trade.
total_trade = 100
pred_delta_threshold = 0.001
time_to_get_quote_seconds = 2.1
interval_typew = 0
quote_interval_pricewait = 0.75
traderecord_interval_refresh = 3

# Params for how far ahead to predict
time_betw_execution_end_and_trade_open = 1.5 # Updated after every trade.
time_betw_get_end_and_execution_end = 0.47 # Hardcoded. 0.6 for z400. 0.47 for rested.
# Function to change global timings.
def change_time_onthefly(time_ge=None, time_et=None): # https://is.gd/HqFpNJ
    global time_betw_execution_end_and_trade_open
    global time_betw_get_end_and_execution_end
    if time_ge is not None: time_betw_get_end_and_execution_end = time_ge
    if time_et is not None: time_betw_execution_end_and_trade_open = time_et

