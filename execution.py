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


def checks(trade_params=None, df=None, day_change_chk=False, trade_start_chk=False, cycle1_warmup_chk=False, params_chk=False, time_mismatch_chk=False):
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
        print('warm_range:', pr.warm_range,'\ntrain_range:', pr.train_range,'\ndelay_range:', pr.delay_range)
        print('ridge_range:',pr.ridge_range,'\nthreshold_test_nrmse:',pr.threshold_test_nrmse, '\n')
        print('Trading Params:')
        print('total_trade:', pr.total_trade,'\nlookback_t:', pr.lookback_t)
        print('pred_delta_threshold:',pr.pred_delta_threshold)
        print('time_to_get_quote_seconds:',pr.time_to_get_quote_seconds,'\n')
        if setup_check == 'Cancel':
            print('\nCancelled by user.')
            return 2

    if cycle1_warmup_chk:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t+1, isTrading=1)
        print('First cycle warmed up.\n')
        return 3

    if params_chk and trade_params is not None:
        # Check if params are still default, if so, skip one iteration.
        default_param = []
        for i in range(0,pr.number_best_param): default_param.append([])
        if trade_params == default_param:
            print('Params still default. Go on to next iteration.')
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
    return


def get_latest_trade_record(isPrint):
    """
    Sample record pulled from webiiste
    ['%', '5132.46', 'February', '08,', '13:05:03', '5133.17', 'February', '08,', '13:05:08', '1.00', '1.80', 'WIN']
    """
    # Swwtich tab. Assume trade record page is on tab 2
    ut.tab_switch(tab=2)
    # Wait for trade to be registered
    time.sleep(pr.asset_duration-1)
    # Refresh to make sure we have latest trade.
    pag.hotkey('f5', interval=pr.demotrade_interval_refresh)
    pag.click(x=pr.olymp_trade_record[0], y=pr.olymp_trade_record[1], interval=0.1)
    pag.hotkey('ctrl', 'a', interval=0.1)
    pag.hotkey('ctrl', 'c', interval=0.1)
    data = Tk().clipboard_get()
    start_index = data.rfind("Status")
    data = data[start_index+len("Status")+len("Quickler	80%"):start_index+len("Status")+len("Quickler	80%")+77]
    record = []
    for x in data.split():
        record.append(x)
    if isPrint:
        print('Last trade result:')
        print('OpenPrice:', record[1], 'ClosePrice:', record[5])
        print('OpenTime:', record[4], 'CloseTime:',record[8])
        if float(record[-1]) != 0:
            record.append('WIN')
            print('Outcome: WIN!!!!!!\n')
        else:
            record[-1] = 'LOSE'
            print('Outcome: LOSE.\n')
    ut.tab_switch(tab=1)
    return record[4] # time of trade open on platform.


def demo_trade():
    pag.click(x=pr.olymp_account_switch[0], y=pr.olymp_account_switch[1], interval=0.5)
    pag.click(x=pr.olymp_demo_account[0], y=pr.olymp_demo_account[1], interval=0.5)
    pag.click(x=pr.olymp_up[0], y=pr.olymp_up[1], interval=0.5)
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    return datetime.datetime.strptime(now, '%H:%M:%S.%f')


def trade_execution(cycle, trade):

    # We cross validate for best params.
    best_param, picklename, get_one_second, updated_test_time = an.cross_val_trading(lookback_t=pr.lookback_t)

    # Check if params are still default, if so, skip one iteration.
    if checks(trade_params=best_param, params_chk=True) == 4:
        cycle += 1
        return 1 , cycle, trade

    # Print params to be used.
    print('Using this params for this cycle: [train, delay, NRMSE, lookback_t, test] ')
    print(best_param)

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
    if pr.cross_val_specify_test: test_points = pr.test_points
    else: test_points = [updated_test_time-0.5,updated_test_time, updated_test_time+0.5]
    for t in test_points:
        results.append(
            an.compute_ngrc(df, isDebug=0, isInfo=0, warmup=-1, train=best_param[0][0], k=best_param[0][1], test=t,
                            ridge_param=pr.ridge_range[0], isTrg=0, isTrading=1))
    print('Updated_test_time chosen for trade prediction:', updated_test_time)
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
    print('Std Dev of all results pred delta:', stdev_pred_delta, '\n')

    # Trade Execution
    if time_mismatch != 1:
        print('Time Mismatch: NO')
        if action_sum == 0 or action_sum == len(results):
            print('Direction agreement: YES ')
            # Check if mean of delta is above threshold.
            if abs(mean_pred_delta) > pr.pred_delta_threshold:
                print('Threshold met: YES')
                execute(signal=results[0][0])
                # Check if agreement is no action
                if results[0][0] != -1:
                    trade += 1
                    get_latest_trade_record(isPrint=True)
                    if trade == pr.total_trade:
                        end(cycle,trade)
                        return -1, cycle, trade

    if time_mismatch == 1: print('No execution - Time Mismatch: YES')
    if action_sum != 0 and action_sum != len(results): print('No execution - Direction agreement: NO')
    if abs(mean_pred_delta) < pr.pred_delta_threshold: print('No execution - Threshold met: NO')

    cycle += 1
    return 0 , cycle, trade


if __name__ == '__main__':
    # Initialize logging
    # tradelog_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tradelog_name = "trade_execution_"+str(tradelog_datetime)+".log"
    # logging.basicConfig(filename=tradelog_name, level=logging.DEBUG) # https://www.loggly.com/?p=76609

    multiprocessing.freeze_support() ; cycle = 1 ; trade = 0 ;
    while True:
        print('\n/****************************************************************************/\n')
        print('Cycle # : ', cycle) ; print('Trade executed: ', trade, '\n')

        # Some checks to make sure we're good before trading.
        if checks(day_change_chk=True) == 1:
            # Sleep for few minutes for data to build up.
            # If we do not sleep, will have issues with hour_min_to_list_t
            time.sleep((pr.lookback_t+1)*60)
            if pr.olymp_day is not None:
                # Change date with PyAutoGUI only if we have date
                ut.date_changer()
            else:
                break
        if cycle == 1:
            if checks(cycle, trade_start_chk=True) == 2:
                break
            an.update_time_betw_execute_trade_open()
            gq.olymptrade_update_hour()
            checks(cycle1_warmup_chk=True)
            an.cross_val_trading(lookback_t=pr.lookback_t)

        # We time our getting of data, do cross val do prediction and execute trade if within NRMSE
        flow_control = trade_execution(cycle=cycle, trade=trade)

        if flow_control[0] == -1:
            break
        if flow_control[0] == 1 or flow_control[0] == 0:
            cycle = flow_control[1]
            trade = flow_control[2]

        # Get time between trade execution to trade close by doing a demo trade.
        an.update_time_betw_execute_trade_open()
        # Buld data up again in case the previous gets is not clean or full
        gq.build_dataset_last_t_minutes(t=pr.lookback_t , isTrading=1)