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
        olymp_day = ()
    if asset_name == 'EURUSD':
        olymp_hr = (164,1100)
        olymp_up = (1800,683) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (1803,763) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (280,1001)
        olymp_day = ()
    olymp_browser = (1722, 22)
    olymp_trade_record = (323,702)
    click_start = (132, 1114)
if current_system == 'z400':
    # Check browser at 100% zoom level
    if asset_name == 'Quickler':
        olymp_hr = (86,633)
        olymp_up = (888,340) # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = (888,389) # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = (171,572)
        olymp_day = ()
    if asset_name == 'EURUSD':
        olymp_hr = ()
        olymp_up = () # Half: (1812,580) Maximized: (3692, 580)
        oylmp_down = () # Half: (1812,659) Maximized: (3692, 659)
        olymp_date = ()
        olymp_day = ()
    olymp_browser = (521, 614)
    olymp_trade_record = (67,454)
    click_start = (73, 722)


# Data file location
data_store_location = 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/'


# Levers for different function
find_pos = False
test_get_one = False
test_get_some = False
test_build_dataset_last_t = False
test_build_dataset = False
test_load_function = False
test_compute_function = False


# Backtesting with Cross_val_trading
# Make sure date selected correctly.
# Will overwrite files in today's data/training/*today* folder
test_cross_val_trading = True
cross_val_past = False
test_hour = '03' ; test_minute = '15' ; test_second = '12'

# Cross Val Params
warm_range = [-1] # In seconds. -1 to train and test on as close to current as possible. Must be > 0
train_range = range(3,8) # In seconds
delay_range = range(2,8) # In seconds
test_range = [8] # In seconds. Updated on the fly during trading
ridge_range = [100]
threshold_test_nrmse = [0.2] # Set to 1 to allow all to show up
lookback_t_min = 2 # Only read by compute() when predicting for trade.
lookback_t = 2 # Larger lookback_t allows for wider range of warm_range
number_best_param = 10 # Minimally 1

# Trading Params
total_trade = 3
pred_delta_threshold = 0.05
time_to_get_quote_seconds = 2.1
quote_interval_typew = 0.1
quote_interval_pricewait = 1
target_start1_time_second = 60
adjusted_start1_time_second = target_start1_time_second - (time_to_get_quote_seconds*lookback_t)
test_points = [test_range[0]-1, test_range[0], test_range[0]+1]

