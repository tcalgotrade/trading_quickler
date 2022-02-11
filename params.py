import numpy as np

current_system = 'rested'
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
cross_val_past = False
cross_val_specify_test = False
if cross_val_past:
    test_hour = '10' ; test_minute = '15' ; test_second = '15'
if cross_val_specify_test:
    test_range = [7,8.5,10]  # In seconds.
    test_points = [test_range[0] - 0.5, test_range[0], test_range[0] + 0.5]
if test_cross_val_trading:
    lookback_t_min = 2 # Only read by compute() when predicting for trade.
    lookback_t = 10 # Larger values of lookback_t allows for wider range of warm_range. if =2, note that it is actually more like 1+ mins as we get most current with get one.
    number_best_param = 10 # Minimally 1
    warm_range = np.arange(30,lookback_t-1,30) ; warm_range = np.append(warm_range,-1)  # In seconds. -1 to train and test on as close to current as possible. Must be > 0
    train_range = range(2,11) # In seconds
    delay_range = range(10,30) # In seconds
    ridge_range = range(0,3)
    threshold_test_nrmse = [1] # Set to 1 to allow all to show up

if not test_cross_val_trading:
    warm_range = [-1] # In seconds. -1 to train and test on as close to current as possible. Must be > 0
    train_range = range(5,10) # In seconds
    delay_range = range(10,16) # In seconds
    ridge_range = [0]
    threshold_test_nrmse = [1] # Set to 1 to allow all to show up
    lookback_t_min = 2 # Only read by compute() when predicting for trade.
    lookback_t = 2 # Larger lookback_t allows for wider range of warm_range. if =2, note that it is actually more like 1+ mins as we get most current with get one.
    number_best_param = 10 # Minimally 1

# Params for how far ahead to predict
time_taken_by_cross_val = -1 # Updated at cycle1 & every cycle.
time_taken_by_trade_execution = 1.5 # Updated after every trade.
time_betw_cross_val_and_execution = 0.3 # Hardcoded.
def change_time_onthefly(time_cv=None, time_te=None): # https://is.gd/HqFpNJ
    global time_taken_by_cross_val
    global time_taken_by_trade_execution
    if time_cv is not None: time_taken_by_cross_val = time_cv
    if time_te is not None: time_taken_by_trade_execution = time_te

# Trade Execution Params
total_trade = 20
pred_delta_threshold = 0.5
time_to_get_quote_seconds = 2.1
interval_typew = 0
quote_interval_pricewait = 0.5
traderecord_interval_refresh = 3

