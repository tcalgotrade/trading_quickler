import os.path
import pyautogui as pag
import datetime
import time
from tkinter import Tk
import pickle
import params as pr
import logging as lg


def process_current_datetime(hour=None, min=None):

    hour = list(str(hour))
    min = list(str(min))

    if len(hour) == 1:
        hour_front = '0'
        hour_back = hour[0]
    else:
        hour_front = hour[0]
        hour_back = hour[1]

    if len(min) == 1:
        min_front = '0'
        min_back = min[0]
    else:
        min_front = min[0]
        min_back = min[1]

    return (hour_front,hour_back,min_front,min_back)


def tab_switch(tab, wait=0.3, refresh=False):
    pag.click(x=pr.olymp_browser[0], y=pr.olymp_browser[1])
    pag.keyDown('ctrl')
    pag.press(str(tab))
    pag.keyUp('ctrl')
    if refresh:
        pag.press('f5')
    time.sleep(wait)
    return


def olymptrade_time_and_quote(hour_front, hour_back, min_front, min_back, interval_typew=0, interval_price_wait=0.25):

    # Click on Olymptrade Hour, last digit
    pag.click(x=pr.olymp_hr[0], y=pr.olymp_hr[1])
    pag.typewrite(['backspace', 'backspace'], interval=interval_typew)
    pag.typewrite([hour_front])
    pag.typewrite([hour_back], interval=interval_typew)

    # Click on Olymptrade Min, last digit
    pag.typewrite(['tab'], interval=0)
    pag.typewrite([min_front])
    pag.typewrite([min_back], interval=interval_typew)
    pag.typewrite(['enter'], interval=interval_price_wait)

    # Instead of dragging, we select all and leave it to others to sort data out.
    pag.click(x=pr.drag_start[0], y=pr.drag_start[1])
    pag.hotkey('ctrl', 'a')
    pag.hotkey('ctrl', 'c')

    return


def get_one_now(olymp_hr, olymp_min, drag_start, drag_end, task_bar, notepad_pp, waittime=0):

    # Get date time
    now = datetime.datetime.now
    date = now().strftime("%d%m%Y")
    hour_front = now().strftime("%H")[0]
    hour_back = now().strftime("%H")[1]
    min_front = now().strftime("%M")[0]
    min_back = now().strftime("%M")[1]
    sec = now().second

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    if sec < waittime:
        print('Waiting for enough datapoints before getting one now...')
        time.sleep(waittime-sec)

    olymptrade_time_and_quote(hour_front=hour_front, hour_back=hour_back, min_front=min_front, min_back=min_back)

    # Save clipboard to pickle file
    data = Tk().clipboard_get()
    with open(pr.data_store_location+date+'/'+hour_front+hour_back+min_front+min_back, 'wb') as f:
        pickle.dump(data, f)

    picklename = pr.data_store_location+date+'/'+hour_front+hour_back+min_front+min_back
    return picklename
# get_one_now(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min,drag_start=pr.drag_start, drag_end=pr.drag_end,task_bar=pr.task_bar, notepad_pp=pr.notepad_pp,waittime=0)


def get_some(olymp_hr, olymp_min, drag_start, drag_end, task_bar, notepad_pp,
             hour_list, min_list):

    # Get date
    date = datetime.datetime.now().strftime("%d%m%Y")

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    for hour in hour_list:

        hour_front = process_current_datetime(hour=hour)[0]
        hour_back = process_current_datetime(hour=hour)[1]

        for min in min_list:

            min_front = process_current_datetime(min=min)[2]
            min_back = process_current_datetime(min=min)[3]

            olymptrade_time_and_quote(hour_front=hour_front, hour_back=hour_back, min_front=min_front,
                                      min_back=min_back)

            # Save clipboard to pickle file
            data = Tk().clipboard_get()
            with open(pr.data_store_location+date+'/'+hour_front+hour_back+min_front+min_back, 'wb') as f:
                pickle.dump(data, f)

    return

def build_dataset(hours, mins):

    setup_check = pag.confirm("Is Olymptrade browser window maximized and setup?\n"
                              "Is production_quote.txt open?")
    if setup_check == 'Cancel':
        print('Cancelled by user.')
        return

    get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start,
             drag_end=pr.drag_end, task_bar=pr.task_bar, notepad_pp=pr.notepad_pp,
             hour_list=hours, min_list=mins)
# build_dataset(hours=[10,12,14,16,18,20,22], mins=range(28,49))


def build_dataset_last_t_minutes(t=1, isTrading=0):

    if not isTrading:
        setup_check = pag.confirm("Is Olymptrade browser window maximized and setup?\n"
                                  "Is production_quote.txt open?")
        if setup_check == 'Cancel':
            print('Cancelled by user.')
            return

    # Get time
    start_time = now = datetime.datetime.now()
    current_hour = now.hour
    hour_str = now.strftime("%H")
    current_min = now.minute + 1
    min_str = now.strftime("%M")
    # To use cross_val_trading to look back at a particular minute and lookback_t before it.
    if pr.force_manual_cross_val_trading:
        current_hour = pr.forced_hour
        current_min = pr.forced_min

    if t > current_min:
        hours = [current_hour]
        mins = range(0,current_min)
        get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end, task_bar=pr.task_bar,
                 notepad_pp=pr.notepad_pp, hour_list=hours, min_list=mins)
        if current_hour - 1 < 0:
            hours = [23]
        else:
            hours = [current_hour-1]
        mins = range(60-(t-current_min),60)
        get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end, task_bar=pr.task_bar,
                 notepad_pp=pr.notepad_pp, hour_list=hours, min_list=mins)
    if t <= current_min:
        hours = [current_hour]
        mins = range(current_min-t,current_min)
        get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end, task_bar=pr.task_bar,
                 notepad_pp=pr.notepad_pp, hour_list=hours, min_list=mins)

    print('Built dataset for lookback_t:', t , 'minutes behind this time :', current_hour , current_min)
    print('Took this amount of time:', datetime.datetime.now() - start_time, 'to get', t, 'minutes of data')
    return hour_str, min_str
# build_dataset_last_t_minutes(t=1, isTrading=1)
