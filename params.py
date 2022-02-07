import numpy as np
import pyautogui as pag

def find_mouse_pos():
    return print(' Position is : ', pag.position())
# find_mouse_pos()

def show_keys():
    return print(pag.KEYBOARD_KEYS)
# show_keys()

# Required coordinates. Works only on my screen. Check browser at 100% zoom level
olymp_hr = (162,973)
olymp_min = (199,973)
olymp_browser = (322, 1043)
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

# Cross Val Params
warm_range = [1]
train_range = [20]
delay_range = range(2,10)
test_range = [10]
ridge_range = [0]
threshold_test_nrmse = [0.2]

# To use cross_val_trading to look back at a particular minute and lookback_t before it.
force_manual_cross_val_trading = False
forced_hour = 19
forced_min = 58

# Trading Params
total_trade = 20 ; lookback_t = 1
time_to_get_quote_seconds = 3.7
target_start1_time_second = 60 ; target_start2_time_second = 45
adjusted_start1_time_second = target_start1_time_second - (time_to_get_quote_seconds*lookback_t)
adjusted_start2_time_second = target_start2_time_second - (time_to_get_quote_seconds*lookback_t)
test_points = [test_range[0]-1, test_range[0], test_range[0]+1]
pred_delta_threshold = 0.1 ; percent_correct_dir = 0
which_start = [1]

