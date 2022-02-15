import pyautogui as pag
import os.path
import datetime
from tkinter import Tk
import pickle
import params as pr
import utility as ut
import tenacity as te


def olymptrade_update_hour_now(interval=pr.interval_typew):
    """"
    Input
    interval time for typewrite operation. We use default in params but this can be overwritten.

    Function
    Take current time (hour), clicks on hour filed as dictated by params, and keys in current hour
    """
    current_hour = datetime.datetime.now().strftime('%H')
    hour_front = current_hour[0]
    hour_back = current_hour[1]
    # Click on Olymptrade Hour, last digit
    pag.click(x=pr.olymp_hr[0], y=pr.olymp_hr[1])
    pag.typewrite(['backspace', 'backspace'], interval=pr.interval_typew)
    pag.typewrite([hour_front])
    pag.typewrite([hour_back], interval=pr.interval_typew)
    return

@te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_exponential(multiplier=1, min=0.1, max=0.5) , stop=te.stop_after_attempt(10))
def olymptrade_time_and_quote(hour, minute, interval_price_wait=pr.quote_interval_pricewait):
    """
    Input
    hour_front, hour_back, min_front, min_back : expects to be string or integer, time of quote that you want from Olymptrade
    interval time for typewrite operation & waiting for price. We use default intervals in params but this can be overwritten.
    Interval times can differ from system to system and may need to be tuned.

    Function
    Take hour and minute given, key in hour then minute, using CTRL+A and tab. This is the fastest tested.
    Wait for some time for data to show, then CTRL+A to select all and copy whatever is on screen to system clipboard.
    Get from clipboard, clean_get , then check the timings to make sure it cohere with now.
    Position of clicks taken from params file. Will change depending on browser used, zoom level, window position.

    Using Tenacity lib, this is function is retried a number of times, as specified in the decorator above it.
    We retry if there's ANY exception, and do exponential wait in between each retry.
    Most commonly, we retry if timings do not match.
    The retry can put here so that if we do a long lookback_t, we only repeat this, rather than the whole set.
    """

    # PyAutoGui needs it as string to type.
    hour, minute = ut.stringify_hour_min(hour, minute)

    # Click on Olymptrade Hour, last digit
    pag.click(x=pr.olymp_hr[0], y=pr.olymp_hr[1])

    # Select all
    pag.hotkey('ctrl', 'a')

    # Type in hour
    pag.typewrite([hour[0], hour[1]])

    # Click on Olymptrade Min, last digit
    pag.typewrite(['tab'])

    pag.typewrite([minute[0], minute[1]], interval=interval_price_wait)

    # Instead of dragging, we click, select all and leave it to lock_and_load func to clean up.
    pag.click(x=pr.click_start[0], y=pr.click_start[1])
    pag.hotkey('ctrl', 'a')
    pag.hotkey('ctrl', 'c')

    # Get clipboard
    data = Tk().clipboard_get()

    # We filter out unwanted bits
    data = clean_get(data)

    # Check that timing from what we got is what we want.
    if data[4:6] != minute:
        raise Exception

    return data

def clean_get(data):
    """
    Input
    data is expected to be string due to use of string search methods here.

    Function
    Given some data, we find the largest index number that has the phrase "Quote" and the largest index that has "VerifyMyTradeExecution"
    Then we slice the data (string) to only contain time and quote.

    Output
    A cleaned up string containing just what we need: time and quote, with time on every odd numbered row.

    Sample of end state (Some garble up top is ignored by system and hasn't proved problematic thus far.):
    07:55:00.350
    4934.58
    07:55:00.904
    4934.52
    07:55:01.411
    4934.55 ...
    """
    start_index = data.rfind("Quote")
    end_index = (len(data) - data.rfind("VerifyMyTradeExecution"))+1
    data = data[start_index+len("Quote"):-end_index]
    return data


def get_one_now():
    """"
    Function
    Take current time now, then execute a olymptrade_time_and_quote()
    Check if we're testing. If so, we take time from params.
    Check if folder today is created. If not, we create it.
    Get string from clipboard after the copy.
    Sanity check 1st timing of data. If insane, throw exception.
    Save data as pickle.

    We also discern if we are doing some sort of testing. (unsure if needed as haven't used for some time)

    Output
    picklename - string, this is the both directory and filename of the data we have pickled.
        Expect it to be named after current hour and minute.
        Sample picklename: 'C:/Users/sar02/OneDrive/ML-Data-Stats/trading_quickler/data/training/15022022/0330'
        This is used by lock_and_load to make sure we load and prep data (for compute) that is existent.
    now.hour, now.minute, now.second - integers, this is the time at which this data was retrieved.
    """

    # Get date time
    now = datetime.datetime.now()
    date = now.strftime("%d%m%Y")

    hour, minute = ut.stringify_hour_min(now.hour, now.minute)

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    if pr.test_cross_val_trading and pr.test_cross_val_past:
        hour = pr.test_hour
        minute = pr.test_minute

    data = olymptrade_time_and_quote(hour=hour, minute=minute)

    # Save it.
    with open(pr.data_store_location + date + '/' + hour + minute, 'wb') as f:
        pickle.dump(data, f)

    picklename = pr.data_store_location+date+'/'+hour+minute

    if pr.test_cross_val_trading and pr.test_cross_val_past:
        return picklename, int(pr.test_hour), int(pr.test_second), int(pr.test_second)

    return picklename, now.hour, now.minute, now.second


def get_some(hours_list, minutes_list):
    """
    Input
    hours_list, minutes_list: expects 2 lists. hours_list is a list of integers containing the hours we want to get quote from.
    minutes_list is a list of list of integers containing minutes of the data we want, corresponding to each our in hours_list
        Sameple: hours_list - [1,2] , meaning we want quote from 0100 hours and 0200 hours
        minutes_list - [ range(0,60), range(0,15) ], meaning we want quote from 0100 to 0159 + 0200 to 0214.

    Function
    Check date and create folder for the day.
    Since we have integers coming in, we convert them to string. We need "01" in string when given int of 1.
    Then we iterate through the given list to get quote.

    """
    # Get date
    date = datetime.datetime.now().strftime("%d%m%Y")

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + date + '/'):
        os.mkdir(pr.data_store_location + date + '/')

    # 1st element of hours_list
    hour, _ = ut.stringify_hour_min(hour=hours_list[0])

    # For 1stelement of hours_list, get minutes.
    for minute in minutes_list[0]:
        _, minute = ut.stringify_hour_min(minute=minute)
        data = olymptrade_time_and_quote(hour=hour, minute=minute)

        # Save it.
        with open(pr.data_store_location+date+'/'+hour+minute, 'wb') as f:
            pickle.dump(data, f)

    # 2nd element of hours_list
    if len(hours_list) == 2:

        hour, _ = ut.stringify_hour_min(hour=hours_list[1])

        # For 2nd element of hours_list, get minutes.
        for minute in minutes_list[1]:
            _, minute = ut.stringify_hour_min(minute=minute)
            data = olymptrade_time_and_quote(hour=hour, minute=minute)

            # Save it.
            with open(pr.data_store_location + date + '/' + hour + minute, 'wb') as f:
                pickle.dump(data, f)

    return


def build_dataset_last_t_minutes(t=1):
    """
    Input
    t: Expects an integer to represent minutes.
        This is taken to be minutes including the current minute that we are in, which is not yet finished.

    Function
    Get a set of data from Olymptrade, from now, to given number of minutes ago..
    """
    # Get time
    start_time = now = datetime.datetime.now()
    # To use cross_val_trading to look back at a particular minute and lookback_t before it.
    current_hour = now.hour ; current_min = now.minute ; current_sec = now.second

    if pr.test_cross_val_trading and pr.test_cross_val_past:
        current_hour = int(pr.test_hour); current_min = int(pr.test_minute); current_sec = int(pr.test_second)

    hours , minutes = ut.hour_min_to_list_t(current_hour, current_min, current_sec, t=t)
    get_some(hours_list=hours, minutes_list=minutes)

    print('Built dataset for lookback_t:', t , 'minutes behind this time :', now.hour , now.minute)
    print('Took this amount of time:', datetime.datetime.now() - start_time, 'to get', t, 'minutes of data')
    print('Time per minute:', (datetime.datetime.now() - start_time).total_seconds()/t )
    return


if __name__ == '__main__':
    if pr.test_get_one:
        get_one_now()
    if pr.test_get_some:
        get_some(hours_list=[20,21], minutes_list=[[58,59],[0,1,2]])
    if pr.test_build_dataset_last_t:
        build_dataset_last_t_minutes(t=60)
        print('Time now is:', datetime.datetime.now())

