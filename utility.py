import pyautogui as pag
import params as pr
import time
import datetime


def find_mouse_pos():
    return print(' Position is : ', pag.position())
if pr.find_pos:
    find_mouse_pos()


def mouse_pos_roll_call():
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


def hour_min_to_list_t(hour, minute, second=1, t=1):
    """
    Input
    hour, minute, second: expected to be integers, to represent time
    t: expected to be integers

    Function
    Given a time, we calculate the hours and minutes needed that would represent t minutes ago.
        E.g.: t = 20 , and we are given hour = 10 , and minute = 15, then output should be
        hours_list = [9,10] and minutes_list = [range(55,60), range(0,10)]
    """
    hours_list = None
    minutes_list = None

    # Assuming t is no larger than 60 minutes
    if t <= minute:
        hours_list = [hour]
        if second > 0:
            minutes_list = [range((minute-t)+1, minute+1)]
        else:
            minutes_list = [range((minute-t),minute)]

    if t > minute:
        if second > 0:
            hours_list = [hour - 1, hour]
            minutes_list = [range(60-(t-minute)+1,60),range(0,minute+1)]
        else:
            if t < 60:
                hours_list = [hour - 1]
                minutes_list = [range(60-(t-minute),60)]
            else:
                hours_list = [hour-1]
                minutes_list = [range(60 - (t - minute), 60)]

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