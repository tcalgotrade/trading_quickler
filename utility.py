import numpy as np
import pyautogui as pag
import params as pr
import time
import datetime


def find_mouse_pos():
    return print(' Position is : ', pag.position())
if pr.find_pos:
    find_mouse_pos()


def mouse_pos_roll_call():

    # Main platform
    pag.alert(text='Go to: Hour')
    print('Hour pos:', pag.position())
    pag.alert(text='Go to: UP')
    print('UP pos:', pag.position())
    pag.alert(text='Go to: DOWN')
    print('DOWN pos:', pag.position())
    pag.alert(text='Go to: DATE')
    print('DATE pos:', pag.position())
    pag.alert(text='Go to: Browser')
    print('Browser pos:', pag.position())
    pag.alert(text='Go to: Click_start')
    print('Click_start pos:', pag.position())
    pag.alert(text='Go to: olymp_asset_button')
    print('olymp_asset_button pos:', pag.position())
    pag.alert(text='Go to: olymp_info_button')
    print('olymp_info_button pos:', pag.position())
    pag.alert(text='Go to: olymp_right_arrow')
    print('olymp_right_arrow pos:', pag.position())
    pag.alert(text='Go to: olymp_quote_history')
    print('olymp_quote_history pos:', pag.position())

    # Trade recrod
    pag.alert(text='Go to: Trade Record')
    print('Trade Record pos:', pag.position())
    pag.alert(text='Go to: 1st Trade Record')
    print('1st Trade Record pos:', pag.position())

    return
if pr.position_roll_call:
    mouse_pos_roll_call()

def show_keys():
    return print(pag.KEYBOARD_KEYS)


def stringify_hour_min(hour=None, minute=None):
    """
    Input
    hour, minute: expects either string or integer

    Function
    We take hour and minute and convert it to double digit representation.
        E.g: if hour is int 1, then we output an hour = '01'

    Output
    hour and minute: in double digit representation.
    """
    if hour is float or minute is float:
        raise Exception

    if hour is not None:
        hour = str(hour)
        if len(str(hour)) == 1:
            hour = '0'+str(hour)

    if minute is not None:
        minute = str(minute)
        if len(str(minute)) == 1:
            minute = '0'+str(minute)

    return hour, minute


def hour_min_to_list_t(hour, minute, t=1):
    """
    Input
    hour, minute: expected to be integers, to represent time
    second: expected to be integers
    t: expected to be integers, how far do we go back?

    Function
    Given a time, we calculate the hours and minutes needed that would represent t minutes ago.
        Includes current minute that one is in.
        Single hour E.g. 1: t = 5 , and we are given hour = 10 , and minute = 15, then output should be
            hours_list = [10] and minutes_list = [range((15+1)-5,15+1)]
        Single hour E.g. 2: t = 5 , and we are given hour = 10 , and minute = 3, then output should be
            hours_list = [9, 10] and minutes_list = [range(59,60),range(0,4)]
        Single hour E.g. 2: t = 6 , and we are given hour = 10 , and minute = 0, then output should be
            hours_list = [9, 10] and minutes_list = [range(55,60),range(0,1)]
        Multi hour E.g. 1: t = 120 , and we are given hour = 10 , and minute = 15, then output should be
            hours_list = [8,9,10] and minutes_list = [range(16,60), range(0,60), range(0,16)]
        Multi hour E.g. 2: t = 120 , and we are given hour = 0 , and minute = 0, then output should be
            hours_list = [22,23,0] and minutes_list = [range(1,60), range(0,60), range(0,1)]
    These lists are then sent to functions that use PyAutoGui to typewrite it in.
    """
    hours_list = None
    minutes_list = None

    # If t is less than where we are at in time, we have enough within the hour.
    if t <= minute+1:
        # Jsut need 1 element
        hours_list = [hour]

        # Form minutes_list
        minutes_list = [range((minute+1)-t, minute+1)]

        return hours_list, minutes_list

    # If t is way more than where are at in time.
    if t > minute+1:
        # Calc how many hours we need.
        number_of_hours = np.int64(np.ceil(t/60))

        # Create list of hours
        hours_list = np.arange((hour)-number_of_hours,hour+1)

        # In the event we cross over to next day, check for negative and add 24.
        hours_list[hours_list < 0] += 24

        # Form minutes_list
        for i in range(0,number_of_hours+1):
            if i == 0: minutes_list = [range((60-(t-((number_of_hours-1)*60+minute+1))),60)]
            elif i == number_of_hours: minutes_list.insert(number_of_hours, range(0,minute+1))
            else:
                minutes_list.insert(i, range(0,60))

        return hours_list, minutes_list


def tab_switch(tab, wait=0.3, refresh=False):
    pag.click(x=pr.olymp_browser[0], y=pr.olymp_browser[1])
    pag.keyDown('ctrl')
    pag.press(str(tab))
    pag.keyUp('ctrl')
    if refresh:
        pag.press('f5')
    time.sleep(wait)
    return


def simple_sched_start(year, month, day, hour, minute,sec=0):
    # https://is.gd/Lb9tlf
    target_time = datetime.datetime(year, month, day, hour, minute, sec)
    while datetime.datetime.now() < target_time:
        time.sleep(10)
    print('Target time:', year,':', month,':', day,':', hour,':', minute, 'Commencing trading now...\n')
    print('/****************************************************************************/\n')
    return


def refresh():
    pag.click(x=pr.olymp_browser[0], y=pr.olymp_browser[1])
    pag.hotkey('f5', interval=5)
    pag.click(x=pr.olymp_info_button[0], y=pr.olymp_info_button[1], interval=5)
    pag.click(x=pr.olymp_right_arrow[0], y=pr.olymp_right_arrow[1], interval=1)
    pag.click(x=pr.olymp_right_arrow[0], y=pr.olymp_right_arrow[1], interval=1)
    pag.click(x=pr.olymp_right_arrow[0], y=pr.olymp_right_arrow[1], interval=1)
    pag.click(x=pr.olymp_quote_history[0], y=pr.olymp_quote_history[1], interval=1)
    return


def date_changer():
    """" Only changes to one day. Each calendar month is different """
    pag.moveTo(x=pr.click_start[0], y=pr.click_start[1])
    pag.typewrite(['home'], interval=0.5)
    pag.click(x=pr.olymp_date[0], y=pr.olymp_date[1])
    pag.click(x=pr.olymp_day[0], y=pr.olymp_day[1])
    return


def filenames(dates,hours,mins,files,store_folder=pr.data_store_location):
    # Create list of file names given a list of date and time + where to store it.
    for date in dates:
        for hr in hours:
            if len(str(hr)) == 1:
                hr_front = '0'
                hr_back = str(hr)
            else:
                hr_front = str(hr)[0]
                hr_back = str(hr)[1]
            for min in mins:
                if len(str(min)) == 1:
                    min_front = '0'
                    min_back = str(min)
                else:
                    min_front = str(min)[0]
                    min_back = str(min)[1]
                files.append(store_folder+str(date)+'/'+hr_front+hr_back+min_front+min_back)
    return files