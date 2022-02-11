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
import tenacity as te
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

    now = datetime.datetime.now().strftime("%H:%M:%S.%f")
    print('>>> Time @ execute complete : ', now)
    return datetime.datetime.strptime(now, '%H:%M:%S.%f')


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
        # Print key params for logging
        print('Started at:', datetime.datetime.now(), '\n')
        print('Cross Val Params:')
        print('warm_range:', pr.warm_range,'\ntrain_range:', pr.train_range,'\ndelay_range:', pr.delay_range)
        print('ridge_range:',pr.ridge_range,'\nthreshold_test_nrmse:',pr.threshold_test_nrmse, '\n')
        print('Trading Params:')
        print('total_trade:', pr.total_trade,'\nlookback_t:', pr.lookback_t)
        print('pred_delta_threshold:',pr.pred_delta_threshold)
        print('time_to_get_quote_seconds:',pr.time_to_get_quote_seconds,'\n')
        setup_check = pag.confirm(text="1) BROWSER WINDOW AT HALF\n 2) AT QUOTE HISTORY?\n 3) ZOOM LEVEL CORRECT?\n 4) CURRENT SYSTEM PARAM?",
                                  title='>>> CHECKLIST <<<')
        if setup_check == 'Cancel':
            print('\nCancelled by user.')
            return 2

    if cycle1_warmup_chk:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t + 1)
        print('First cycle warmed up.\n')
        return 3

    if params_chk and trade_params is not None:
        # Check if params are still default, if so, skip one iteration.
        default_param = [0,0,1.,0,0,0]
        for t in trade_params:
            if t == default_param:
                print('One or more params still default. Go on to next iteration.\n')
                return 4

    if time_mismatch_chk and df is not None:
        now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        now = datetime.datetime.strptime(now, '%H:%M:%S.%f')
        data_time = df['time'].astype(str)
        if len(data_time.iloc[-1]) == 8:
            data_last_time_string = data_time.iloc[-1] +'.000'
        else:
            data_last_time_string = data_time.iloc[-1]
        data_last_time = datetime.datetime.strptime(data_last_time_string, '%H:%M:%S.%f')
        diff = now - data_last_time
        if diff.total_seconds() < 0:
            print('Time mismatch - Diff of ', diff.total_seconds())
            return 5
    return

@te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_fixed(0.2) , stop=te.stop_after_attempt(3))
def get_latest_trade_record(isPrint):
    """
    Sample record pulled from webiiste
    ['4928.86', 'February', '11,', '23:27:21', '4928.75', 'February', '11,', '23:27:26', '1.00', '0.00', 'Quickler', '80%', '4920.87', 'February', '11,', '23:18:23', '4920.33', 'February', '11,', '23:18:28', '1.00', '1.80', 'Quickler', '80%', '4925.80', 'February', '11,', '23:13:01', '4926.14', 'February', '11,', '23:13:06', '1.00', '1.80', 'Quickler', '80%', '4869.05', 'February', '11,', '17:50:38', '4868.63', 'February', '11,', '17:50:43', '1.00', '1.80', 'Quickler', '80%', '4866.80', 'February', '11,', '17:33:32', '4866.59', 'February', '11,', '17:33:37', '1.00', '0.00', 'Quickler', '80%', '4864.64', 'February', '11,', '17:22:08', '4866.26', 'February', '11,', '17:22:13', '1.00', '0.00', 'Quickler', '80%', '4858.03', 'February', '11,', '17:19:01', '4859.62', 'February', '11,', '17:19:06', '1.00', '1.80', 'Quickler', '80%', '4864.51', 'February', '11,', '17:11:27', '4864.04', 'February', '11,', '17:11:32', '1.00', '1.80', 'Quickler', '80%', '4865.21', 'February', '11,', '17:10:58'...
    """
    # Swwtich tab. Assume trade record page is on tab 2
    ut.tab_switch(tab=2)
    # Wait for trade to be registered
    time.sleep(pr.asset_duration-1)
    # Refresh to make sure we have latest trade.
    pag.hotkey('f5', interval=pr.traderecord_interval_refresh)
    pag.click(x=pr.olymp_trade_record[0], y=pr.olymp_trade_record[1], interval=0.2)
    pag.hotkey('ctrl', 'a', interval=0.1)
    pag.hotkey('ctrl', 'c', interval=0.1)
    data = Tk().clipboard_get()
    start_index = data.find("Quickler	80%")
    data = data[start_index+len("Quickler	80%"):]
    end_index = data.find("Quickler	80%")
    data = data[:end_index]
    record = []
    for x in data.split():
        record.append(x)
    if isPrint:
        print('Last trade result:')
        print('OpenPrice:', record[0], 'ClosePrice:', record[4])
        print('OpenTime:', record[3], 'CloseTime:',record[7])
        if record[-1] != '0.00':
            print('Outcome: WIN!!!\n')
        elif record[-1] == '0.00':
            print('Outcome: LOSS...\n')
        else:
            print('Outcome: Refund.\n')
    ut.tab_switch(tab=1)
    return record[3]


def demo_trade():
    pag.click(x=pr.olymp_account_switch[0], y=pr.olymp_account_switch[1], interval=0.5)
    pag.click(x=pr.olymp_demo_account[0], y=pr.olymp_demo_account[1], interval=0.5)
    pag.click(x=pr.olymp_up[0], y=pr.olymp_up[1], interval=0.5)
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    return datetime.datetime.strptime(now, '%H:%M:%S.%f')


def trade_small():
    pag.click(x=pr.olymp_amount[0], y=pr.olymp_amount[1], interval=0.1)
    pag.hotkey('ctrl', 'a')
    pag.typewrite(['1'])
    pag.click(x=pr.olymp_up[0], y=pr.olymp_up[1], interval=0.5)
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    return datetime.datetime.strptime(now, '%H:%M:%S.%f')


def update_time_betw_execution_end_and_trade_open(execute_time, trade_open_time):
    print('>>> Time @ trade execution:',execute_time)
    trade_open_time = datetime.datetime.strptime(trade_open_time, '%H:%M:%S')
    # Calc difference between. We should expect trade_open_time to be later.
    diff = trade_open_time - execute_time
    pr.change_time_onthefly(time_et=diff.total_seconds()+0.5)
    print('*** Updated time_betw_execution_end_and_trade_open to:', diff.total_seconds()+0.5, '\n')
    return


def update_time_betw_get_end_and_execution_end(execute_time, start_get_end):
    print('>>> Time @ trade execution:',execute_time)
    # Calc difference between. We should expect trade_open_time to be later.
    start_get_end = datetime.datetime.strptime(start_get_end, '%H:%M:%S.%f')
    diff = execute_time - start_get_end
    pr.change_time_onthefly(time_ge=diff.total_seconds())
    print('*** Updated time_betw_get_end_and_execution_end to:', diff.total_seconds(), '\n')
    return


def trade_execution(cycle, trade):

    # We cross validate for best params.
    best_param, test_range_center = an.cross_val_trading(lookback_t=pr.lookback_t)

    # Check if any params are still default, if so, skip one iteration.
    if checks(trade_params=best_param, params_chk=True) == 4:
        cycle += 1
        return 1 , cycle, trade

    # Print params to be used.
    print('Using these params for this cycle:')
    print('[[train, delay, NRMSE, lookback_t, test, ridge]]:')
    for bp in best_param:
        print(bp)

    # Build + get new data since cross val may have taken some time.
    # Build data up + get one for the duration that build last took
    gq.build_dataset_last_t_minutes(t=pr.lookback_t_min)
    picklename, get_one_hour, get_one_minute, get_one_second = gq.get_one_now()

    start_get_end = datetime.datetime.now().strftime('%H:%M:%S.%f')

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
    for p in best_param:
        results.append(an.compute_ngrc(df, warmup=-1, train=p[0], k=p[1], test=p[4],
                            ridge_param=p[5], isTrg=0, isTrading=1))
    print('*** test_range_center used for trade prediction:', test_range_center)

    # Consolidate results for trade execution.
    test_predictions_quote_delta = []
    for result in results:
        test_predictions_quote_delta.append(round(result[2],3))

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
                executed_time = execute(signal=results[0][0])
                # Check if agreement is no action
                if results[0][0] != -1:
                    trade += 1
                    trade_opened_time = get_latest_trade_record(isPrint=True)
                    update_time_betw_get_end_and_execution_end(executed_time, start_get_end)
                    update_time_betw_execution_end_and_trade_open(executed_time, trade_opened_time)
                    if trade == pr.total_trade:
                        end(cycle,trade)
                        return -1, cycle, trade

    if time_mismatch == 1: print('No execution - Time Mismatch: YES')
    if 0 < action_sum < len(results) or action_sum < 0: print('No execution - Direction agreement: NO')
    if abs(mean_pred_delta) < pr.pred_delta_threshold: print('No execution - Threshold met: NO')

    cycle += 1
    return 0 , cycle, trade


if __name__ == '__main__':
    # Initialize logging
    # tradelog_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tradelog_name = "trade_execution_"+str(tradelog_datetime)+".log"
    # logging.basicConfig(filename=tradelog_name, level=logging.DEBUG) # https://www.loggly.com/?p=76609

    multiprocessing.freeze_support() ; cycle = 1 ; trade = 0
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

        # We time our getting of data, do cross val do prediction and execute trade if within NRMSE
        flow_control = trade_execution(cycle=cycle, trade=trade)

        if flow_control[0] == -1:
            break
        if flow_control[0] == 1 or flow_control[0] == 0:
            cycle = flow_control[1]
            trade = flow_control[2]

