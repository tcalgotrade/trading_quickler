# current_system = 'z400'
current_system = 'rested'

if current_system == 'rested':
    # Check browser at 100% zoom level. Window at half.
    olymp_hr = (162,973)
    olymp_min = (199,973)
    olymp_browser = (1722, 22)
    olymp_up = (1812,580) # Half: (1812,580) Maximized: (3692, 580)
    oylmp_down = (1812,659) # Half: (1812,659) Maximized: (3692, 659)
    olymp_date = (187,887)
    olymp_day = ()
    olymp_trade_record = (323,702)
    olymp_trade_record_first = (364,705)
    olymp_trade_record_drag_start = (623,1286)
    olymp_trade_record_drag_end = (1081,1361)
    drag_start = (132,1114)
    drag_end = (476,2048)
if current_system == 'z400':
    # Check browser at 100% zoom level
    olymp_hr = (86,633)
    olymp_min = (111,635)
    olymp_browser = (521,614)
    olymp_up = (888,340) # Half: (1812,580) Maximized: (3692, 580)
    oylmp_down = (888,389) # Half: (1812,659) Maximized: (3692, 659)
    olymp_date = (171,572)
    olymp_day = ()
    drag_start = (62,710)
    drag_end = (290,1021)


# Data file location
data_store_location = 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/'


# Asset name
asset_name = "Quickler"


# Levers for different function
find_pos = False
test_get_one = False
test_get_some = False
test_build_dataset_last_t = False
test_build_dataset = False
test_load_function = False
test_compute_function = False


# Cross_val_trading: look back from a time and lookback_t behind.
force_manual_cross_val_trading = False
forced_hour = 12
forced_min = 18


# Cross Val Params
warm_range = [1]
train_range = [20,40,60] # In seconds
delay_range = range(2,12) # In seconds
test_range = [10] # In seconds
ridge_range = [2.5e-6]
threshold_test_nrmse = [0.15]


# Trading Params
total_trade = 3
lookback_t = 3
pred_delta_threshold = 0.05
time_to_get_quote_seconds = 2.1
quote_interval_typew = 0.1
quote_interval_pricewait = 1
target_start1_time_second = 60
adjusted_start1_time_second = target_start1_time_second - (time_to_get_quote_seconds*lookback_t)
test_points = [test_range[0]-1, test_range[0], test_range[0]+1]

