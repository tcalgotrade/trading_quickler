import pyautogui as pag
import os.path
import datetime
import time
from tkinter import Tk
import pickle
import params as pr
import utility as ut
import logging as lg


def olymptrade_time_and_quote(hour_front, hour_back, min_front, min_back,
                              interval_typew=0.01, interval_price_wait=0.4):

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

    # Instead of dragging, we click, select all and leave it to load func to clean up.
    pag.click(x=pr.drag_start[0], y=pr.drag_start[1])
    pag.hotkey('ctrl', 'a')
    pag.hotkey('ctrl', 'c')

    return


def clean_get(data):
    start_index = data.rfind("Quote")
    end_index = (len(data) - data.rfind("VerifyMyTradeExecution"))+1
    data = data[start_index+len("Quote"):-end_index]
    return data


def get_one_now():

    # Get date time
    now = datetime.datetime.now()
    date = now.strftime("%d%m%Y")
    hour_front = now.strftime("%H")[0]
    hour_back = now.strftime("%H")[1]
    min_front = now.strftime("%M")[0]
    min_back = now.strftime("%M")[1]

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    olymptrade_time_and_quote(hour_front=hour_front, hour_back=hour_back, min_front=min_front, min_back=min_back)

    # Save clipboard to pickle file
    data = Tk().clipboard_get()
    data = clean_get(data)
    with open(pr.data_store_location+date+'/'+hour_front+hour_back+min_front+min_back, 'wb') as f:
        pickle.dump(data, f)

    picklename = pr.data_store_location+date+'/'+hour_front+hour_back+min_front+min_back
    return picklename, now.hour, now.minute, now.second


def get_some(olymp_hr, olymp_min, drag_start, drag_end, task_bar,
             hours_list, minutes_list):

    # Get date
    date = datetime.datetime.now().strftime("%d%m%Y")

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    # Get quote, from past to now.
    hour_front = ut.process_current_datetime(hour=hours_list[0])[0]
    hour_back = ut.process_current_datetime(hour=hours_list[0])[1]
    for minute in minutes_list[0]:
        minute_front = ut.process_current_datetime(min=minute)[2]
        minute_back = ut.process_current_datetime(min=minute)[3]
        olymptrade_time_and_quote(hour_front=hour_front, hour_back=hour_back, min_front=minute_front,
                                  min_back=minute_back)
        # Save clipboard to pickle file
        data = Tk().clipboard_get()
        data = clean_get(data)
        with open(pr.data_store_location+date+'/'+hour_front+hour_back+minute_front+minute_back, 'wb') as f:
            pickle.dump(data, f)

    if len(hours_list) == 2:
        # Front
        hour_front = ut.process_current_datetime(hour=hours_list[1])[0]
        hour_back = ut.process_current_datetime(hour=hours_list[1])[1]
        for minute in minutes_list[1]:
            minute_front = ut.process_current_datetime(min=minute)[2]
            minute_back = ut.process_current_datetime(min=minute)[3]
            olymptrade_time_and_quote(hour_front=hour_front, hour_back=hour_back, min_front=minute_front,
                                      min_back=minute_back)
            # Save clipboard to pickle file
            data = Tk().clipboard_get()
            data = clean_get(data)
            with open(pr.data_store_location+date+'/'+hour_front+hour_back+minute_front+minute_back, 'wb') as f:
                pickle.dump(data, f)

    return


def build_dataset(hours, mins):

    setup_check = pag.confirm("Is Olymptrade browser window maximized and setup?\n"
                              "Is production_quote.txt open?")
    if setup_check == 'Cancel':
        print('Cancelled by user.')
        return

    get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end,
             task_bar=pr.task_bar, hours_list=hours, minutes_list=mins)


def build_dataset_last_t_minutes(t=1, isTrading=0):

    if not isTrading:
        setup_check = pag.confirm("Is Olymptrade browser window maximized and setup?\n"
                                  "Is production_quote.txt open?")
        if setup_check == 'Cancel':
            print('Cancelled by user.')
            return

    # Get time
    start_time = now = datetime.datetime.now()

    # To use cross_val_trading to look back at a particular minute and lookback_t before it.
    if pr.force_manual_cross_val_trading:
        current_hour = pr.forced_hour
        current_min = pr.forced_min

    hours , minutes = ut.hour_min_to_list_t(now.hour, now.minute, now.second, t=t)
    get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end,
             task_bar=pr.task_bar, hours_list=hours, minutes_list=minutes)

    print('Built dataset for lookback_t:', t , 'minutes behind this time :', now.hour , now.minute)
    print('Took this amount of time:', datetime.datetime.now() - start_time, 'to get', t, 'minutes of data')
    return now.strftime("%H"), now.strftime("%M")


if __name__ == '__main__':
    if pr.test_get_one:
        get_one_now()
    if pr.test_get_some:
        get_some(olymp_hr=pr.olymp_hr, olymp_min=pr.olymp_min, drag_start=pr.drag_start, drag_end=pr.drag_end,
                 task_bar=pr.task_bar, hours_list=[12], minutes_list=[0])
    if pr.test_build_dataset_last_t:
        build_dataset_last_t_minutes(t=15, isTrading=1)
    if pr.test_build_dataset:
        build_dataset(hours=[10,12,14,16,18,20,22], mins=range(28,49))
