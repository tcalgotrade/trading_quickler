import numpy as np
import get_quote as gq
import analysis as an
import pyautogui as pag
from tkinter import Tk
import datetime
import time
import multiprocessing
import pickle
import params as pr
import utility as ut
import logging

pag.FAILSAFE = True

def execute(signal):
    """
    :param signal: 1 buy up, 0 buy down, -1 take no action.
    :return: Log down action taken.
    """

    # Click back to Olymptrade
    pag.click(x=pr.olymp_browser[0], y=pr.olymp_browser[1])

    if signal == 1:
        pag.click(x=pr.olymp_up[0], y=pr.olymp_up[1])
        print('Bought UP')
    if signal == 0:
        pag.click(x=pr.oylmp_down[0], y=pr.oylmp_down[1])
        print('Bought DOWN')
    if signal == -1:
        print('No trade taken: all results are outside of pred_delta tolerance.')

    print('>>> Time @ execute complete : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))
    return


def end(cycle, trade):
    # Print some numbers
    print('\n/****************************************************************************/\n')
    print('Final Cycle # : ', cycle)
    print('Final Trade executed: ', trade)
    return


def checks(trade_params=None, df=None, start1_time_second= None, start2_time_second = None,
           day_change_chk=False, trade_start_chk=False, cycle1_warmup_chk=False, params_chk=False, time_mismatch_chk=False, sec_mismatch_chk=False,
           timed_start1_chk=False, timed_start2_chk=False):
    """ All checks leading up to trade execution checks."""

    if day_change_chk:
        # Check if current time is 2359. If so, break and stop trading.
        # There is currently no way to change date via PyAutoGUI.
        if datetime.datetime.now().hour == 23 and datetime.datetime.now().minute == 59:
            print('We are entering a new day in one minute.')
            return 1

    # Pop a prompt to make sure manual setup is good.
    if trade_start_chk:
        setup_check = pag.confirm(text="1) BROWSER WINDOW AT HALF\n 2) AT QUOTE HISTORY?\n 3) ZOOM LEVEL CORRECT?\n 4) CURRENT SYSTEM PARAM?",
                                  title='>>> CHECKLIST <<<')
        # Print key params for logging
        print('Started at:', datetime.datetime.now(), '\n')
        print('Cross Val Params:')
        print('warm_range:', pr.warm_range,'\ntrain_range:', pr.train_range,'\ndelay_range:', pr.delay_range,'\ntest_range:', pr.test_range)
        print('ridge_range:',pr.ridge_range,'\nthreshold_test_nrmse:',pr.threshold_test_nrmse, '\n')
        print('Trading Params:')
        print('total_trade:', pr.total_trade,'\nlookback_t:', pr.lookback_t,'\nadjusted_start1_time_second:', pr.adjusted_start1_time_second)
        print('test_points:',pr.test_points,'\npred_delta_threshold:',pr.pred_delta_threshold)
        print('time_to_get_quote_seconds:',pr.time_to_get_quote_seconds, '\ntarget_start1_time_second:',pr.target_start1_time_second)
        if setup_check == 'CANCEL':
            print('Cancelled by user.')
            return 2

    if cycle1_warmup_chk:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t+1, isTrading=1)
        print('First cycle warmed up.')
        print('/****************************************************************************/\n')
        return 3

    if params_chk and trade_params is not None:
        # Check if params are still default, if so, skip one iteration.
        default_param = []
        for i in range(0,pr.number_best_param): default_param.append([])
        if trade_params == default_param:
            print('Params still default. Go on to next iteration.')
            print('/****************************************************************************/\n')
            return 4

    if time_mismatch_chk and df is not None:
        now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        now = datetime.datetime.strptime(now, '%H:%M:%S.%f')
        data_time = df['time'].astype(str)
        data_last_time = datetime.datetime.strptime(data_time.iloc[-1], '%H:%M:%S.%f')
        diff = now - data_last_time
        if diff.total_seconds() < 0:
            print('Time mismatch - Diff of ', diff.total_seconds())
            return 5

    if timed_start1_chk and start1_time_second is not None:
        # Timing the start of the cross val. We want to make sure we cross val on latest data and predict as ASAP.
        # About 3 secs for every get_quote action
        if datetime.datetime.now().second < start1_time_second:
            print('Waiting', start1_time_second - datetime.datetime.now().second, 'seconds for right start')
            time.sleep(start1_time_second - datetime.datetime.now().second)
        else:
            print('Waiting', (60 - datetime.datetime.now().second) + start1_time_second, 'seconds for right start')
            time.sleep((60 - datetime.datetime.now().second) + start1_time_second)
        return

    return


def get_latest_trade_record():
    """
    Sample record pulled from webiiste
    ['%', '5132.46', 'February', '08,', '13:05:03', '5133.17', 'February', '08,', '13:05:08', '1.00', '1.80', 'WIN']
    """
    # Swwtich tab. Assume trade record page is on tab 2
    ut.tab_switch(tab=2)
    # Wait for trade to be registered
    time.sleep(pr.asset_duration)
    # Refresh to make sure we have latest trade.
    pag.hotkey('f5')
    # Wait for page to load
    time.sleep(7)
    pag.click(x=pr.olymp_trade_record[0], y=pr.olymp_trade_record[1])
    pag.hotkey('ctrl', 'a')
    pag.hotkey('ctrl', 'c')
    data = Tk().clipboard_get()
    start_index = data.rfind("Status")
    data = data[start_index+len("Status")+len("Quickler	80%"):start_index+len("Status")+len("Quickler	80%")+77]
    record = []
    for x in data.split():
        record.append(x)
    print('\nLast trade result:')
    print('OpenPrice:', record[1], 'ClosePrice:', record[5])
    print('OpenTime:', record[4], 'CloseTime:',record[8])
    if float(record[-1]) != 0:
        record.append('WIN')
        print('Outcome: WIN!!!!!!\n')
    else:
        record[-1] = 'LOSE'
        print('Outcome: LOSE.\n')
    ut.tab_switch(tab=1)
    return record[8]

def update_test_range_param(data_time=None, close_time=None):
    # Convert incoming dataframe to string
    data_time = data_time.astype(str)
    # Get last row's time
    data_last_time = data_time.iloc[-1]
    # Change string to datetime object
    data_last_time = datetime.datetime.strptime(data_last_time, '%H:%M:%S.%f')
    close_time = datetime.datetime.strptime(close_time, '%H:%M:%S')
    # Calc difference between
    diff = close_time - data_last_time
    # Only effect change
    if diff.total_seconds() > pr.asset_duration and diff.total_seconds() < 14:
        pr.test_range = [diff.total_seconds()]
    print('Diff between data last time and trade close time:', diff.total_seconds())
    print('Changed pr.test_range to::',pr.test_range)
    return


def trade_execution(cycle, trade):

    # We cross validate for best params.
    best_param, picklename, get_one_second = an.cross_val_trading(lookback_t=pr.lookback_t)

    # Check if params are still default, if so, skip one iteration.
    if checks(trade_params=best_param, params_chk=True) == 4:
        cycle += 1
        return 1 , cycle, trade

    # Print params to be used.
    print('Using this params for this cycle: [train, delay, NRMSE, lookback_t, test] ', best_param)

    # Load dataframe
    df = an.load(picklename=picklename, lookback=pr.lookback_t_min ,seconds=get_one_second)
    print('Loaded pickle used for prediction for trading ... :', picklename)
    print('Dataframe Statistics:')
    print('>>> Time @ first row:', df['time'].iloc[0])
    print('>>> Time @ last row:', df['time'].iloc[-1])
    print('Mean of quote history:', np.mean(df['quote']))
    print('Std Dev of quote history:', np.std(df['quote']))

    # Check current min/sec vs min/sec of last row of dataframe.
    time_mismatch = 0
    if checks(df=df, time_mismatch_chk=True) == 5:
        time_mismatch = 1

    # Compute what trade actions to take
    # Returns a tuple when trading : (action to take: 1:buy up,0:buy down,-1:do nth, pred_delta)
    results = []
    for test_point in pr.test_points:
        results.append(
            an.compute_ngrc(df, isDebug=0, isInfo=0, warmup=-1, train=best_param[0][0], k=best_param[0][1], test=test_point,
                            ridge_param=pr.ridge_range[0], isTrg=0, isTrading=1))
    print('>>> Time @ compute complete : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Consolidate results for trade execution.
    test_predictions_quote_delta = []
    for result in results:
        test_predictions_quote_delta.append(round(result[1],2))

    action_sum = 0
    for result in results:
        action_sum += result[0]

    # Calc basic stats of delta of price prediction.
    mean_pred_delta = np.mean(test_predictions_quote_delta)
    stdev_pred_delta = np.std(test_predictions_quote_delta)
    print('All results pred delta:', test_predictions_quote_delta)
    print('Average of all results pred delta:', mean_pred_delta)
    print('Std Dev of all results pred delta:', stdev_pred_delta)

    # Trade Execution
    if time_mismatch != 1:
        print('Time Mismatch: NO')
        if action_sum == 0 or action_sum == len(results) or action_sum == -len(results):
            print('Direction agreement: YES ')
            # Check if mean of delta is above threshold.
            if abs(mean_pred_delta) > pr.pred_delta_threshold:
                print('Threshold met: YES')
                execute(signal=results[0][0])
                # Check if agreement is no action
                if results[0][0] != -1:
                    trade += 1
                    close_time = get_latest_trade_record()
                    update_test_range_param(data_time=df['time'], close_time=close_time)
                    if trade == pr.total_trade:
                        end(cycle,trade)
                        return -1, cycle, trade
            else:
                print('No execution. Criteria not met.')

    cycle += 1
    print('/****************************************************************************/\n')
    return 0 , cycle, trade


if __name__ == '__main__':
    # Initialize logging
    # tradelog_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tradelog_name = "trade_execution_"+str(tradelog_datetime)+".log"
    # logging.basicConfig(filename=tradelog_name, level=logging.DEBUG) # https://www.loggly.com/?p=76609

    multiprocessing.freeze_support() ; cycle = 1 ; trade = 0

    while True:
        print('Cycle # : ', cycle) ; print('Trade executed: ', trade, '\n')

        # Some checks to make sure we're good before trading.
        if checks(day_change_chk=True) == 1:
            # Sleep for few minutes for data to build up.
            # If we do not sleep, will have issues with hour_min_to_list_t
            time.sleep((pr.lookback_t+1)*60)
            # Change date with PyAutoGUI
            ut.date_changer()
        if cycle == 1:
            if checks(cycle, trade_start_chk=True) == 2:
                break
            checks(cycle1_warmup_chk=True)

        # We time our getting of data, do cross val do prediction and execute trade if within NRMSE
        flow_control = trade_execution(cycle=cycle, trade=trade)

        if flow_control[0] == -1:
            break
        if flow_control[0] == 1 or flow_control[0] == 0:
            cycle = flow_control[1]
            trade = flow_control[2]

        # Buld data up again in case the previous gets is not clean or full
        gq.build_dataset_last_t_minutes(t=pr.lookback_t , isTrading=1)