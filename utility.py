import pyautogui as pag
import params as pr
import time
import datetime


def find_mouse_pos():
    return print(' Position is : ', pag.position())


def show_keys():
    return print(pag.KEYBOARD_KEYS)


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


def hour_min_to_list_t(hour, minute, second=1, t=1):

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
    target_time = datetime.datetime(year, month, day, hour, minute,sec)
    while datetime.datetime.now() < target_time:
        time.sleep(10)
    print('Target time:', year,':', month,':', day,':', hour,':', minute, 'Commencing trading now...\n')
    print('/****************************************************************************/\n')
    return


def date_changer():
    pag.moveTo(x=pr.drag_start[0], y=pr.drag_start[1])
    pag.typewrite(['home'], interval=0.05)
    pag.click(x=pr.olymp_date[0], y=pr.olymp_date[1])
    pag.click(x=pr.olymp_day_7[0], y=pr.olymp_day_7[1])
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