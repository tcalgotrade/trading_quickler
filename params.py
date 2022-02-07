# Find coordinates
# import utility as ut
# ut.find_mouse_pos()

# Required coordinates. Works only on my screen. Check browser at 100% zoom level
olymp_hr = (162,973)
olymp_min = (199,973)
olymp_browser = (1722, 22)
olymp_up = (1812,580) # Half: (1812,580) Maximized: (3692, 580)
oylmp_down = (1812,659) # Half: (1812,659) Maximized: (3692, 659)
olymp_date = (187,887)
olymp_day_1 = (252,1058)
olymp_day_7 = (209,1107)
drag_start = (132,1114)
drag_end = (476,2048)
task_bar = (0,866)
notepad_pp = (42,1027)
pycharm = (44, 1082)


# Data file location
data_store_location = 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/'


# Levers for testing different function
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
warm_range = [0]
train_range = [50,75,100,125] # In seconds
delay_range = range(2,15) # In seconds
test_range = [10] # In seconds
ridge_range = [0]
threshold_test_nrmse = [0.15]

# Trading Params
total_trade = 5
lookback_t = 4
pred_delta_threshold = 0.1
time_to_get_quote_seconds = 2.1
target_start1_time_second = 60 ; target_start2_time_second = 45
adjusted_start1_time_second = target_start1_time_second - (time_to_get_quote_seconds*lookback_t)
adjusted_start2_time_second = target_start2_time_second - (time_to_get_quote_seconds*lookback_t)
test_points = [test_range[0]-1, test_range[0], test_range[0]+1]

