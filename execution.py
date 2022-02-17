import random
import glob
import os
import numpy as np
import get_quote as gq
import analysis as an
import pyautogui as pag
from tkinter import Tk
import datetime
import time
import multiprocessing
import params as pr
import utility as ut
import tenacity as te
import mlp_svc_explore as mse
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


def end(cycle, trade, total_wins):
    # Print some numbers
    print('\n/****************************************************************************/\n')
    print('Final Cycle # : ', cycle)
    print('Final Trade executed: ', trade)
    print('Final Wins: ', total_wins)
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
        print('Params:')
        print('warm_range:', pr.warm_range,'\ntrain_range:', pr.train_range,'\ndelay_range:', pr.delay_range)
        print('ridge_range:', pr.ridge_range)
        print('threshold_test_nrmse:', pr.threshold_test_nrmse)
        print('lookback_t:', pr.lookback_t)
        print('total_trade:', pr.total_trade)
        print('pred_delta_threshold:', pr.pred_delta_threshold)
        print('traderecord_interval_refresh:', pr.traderecord_interval_refresh,'\n')
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
                print('Params:')
                for p in trade_params:
                    print(p)
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
            print('')
            return 5
    return

@te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_exponential(multiplier=1, min=0.2, max=3) , stop=te.stop_after_attempt(10))
def get_latest_trade_record(isPrint, approach):

    # Swwtich tab. Assume trade record page is on tab 2
    ut.tab_switch(tab=2)

    # Wait for trade to be registered
    time.sleep(pr.asset_duration-1)

    # Refresh to make sure we have latest trade.
    pag.hotkey('f5', interval=pr.traderecord_interval_refresh)

    # Click different depending on method.
    if approach == 1:
        pag.click(x=pr.olymp_trade_record[0], y=pr.olymp_trade_record[1], interval=1)
    if approach == 2:
        pag.click(x=pr.olymp_first_trade_record[0], y=pr.olymp_first_trade_record[1], interval=1)

    pag.hotkey('ctrl', 'a', interval=0.1)
    pag.hotkey('ctrl', 'c', interval=0.1)
    data = Tk().clipboard_get()

    # When pop up DOES NOT show. Less precise trade open.
    if approach == 1:
        """
        Sample record pulled from webiiste
        ['4928.86', 'February', '11,', '23:27:21', '4928.75', 'February', '11,', '23:27:26', '1.00', '0.00']
        """
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
            if record[-1] != '0.00' and record[-1] != record[-2]:
                print('Outcome: WIN!!!\n')
            elif record[-1] == '0.00':
                print('Outcome: LOSS...\n')
            elif record[-1] == record[-2]:
                print('Outcome: Refund.\n')

        # Return 1 if won.
        won = 0
        if record[-1] != '0.00' and record[-1] != record[-2]: won += 1
        ut.tab_switch(tab=1)
        return record[3], won

    # When pop up DOES show. Precise
    if approach == 2:
        """
        Sample record pulled from webiiste
        ['10', 'February', '12:06:45.025', '10', 'February', '12:06:50.025', 'Quote', '5011.31', '5010.84', 'Closed', 'with', 'a', 'profit']
        """
        start_index = data.find("Date and time")
        data = data[start_index+len("Date and time"):]
        end_index = data.rfind("Deal verify")
        data = data[:end_index]
        record = []
        for x in data.split():
            record.append(x)
        if isPrint:
            print('Last trade result:')
            print('OpenPrice:', record[7], 'ClosePrice:', record[8])
            print('OpenTime:', record[2], 'CloseTime:', record[5])
            if record[-1] == 'profit':
                print('Outcome: WIN!!!\n')
            if record[-1] == 'loss':
                print('Outcome: LOSS...\n')
            if record[-1] == 'refund':
                print('Outcome: Refund.\n')

        # Return 1 if won.
        won = 0
        if record[-1] == 'profit': won += 1

        ut.tab_switch(tab=1)
        return record[2], won


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
    if len(trade_open_time) == 8:
        trade_open_time_formatted = datetime.datetime.strptime(trade_open_time, '%H:%M:%S')
    elif len(trade_open_time) == 12:
        trade_open_time_formatted = datetime.datetime.strptime(trade_open_time, '%H:%M:%S.%f')
    else:
        raise Exception
    # Calc difference between. We should expect trade_open_time to be later.
    diff = trade_open_time_formatted - execute_time

    # Sanity Check
    if diff.total_seconds() < 0 or diff.total_seconds() > 5:
        new_time = 1.5
    else:
        new_time = diff.total_seconds()

    if len(trade_open_time) == 8:
        # If we are missing decimals, we add a bit of margin.
        pr.change_time_onthefly(time_et=new_time)
        print('*** Updated time_betw_execution_end_and_trade_open to:', new_time, '\n')
    elif len(trade_open_time) == 12:
        pr.change_time_onthefly(time_et=new_time)
        print('*** Updated time_betw_execution_end_and_trade_open to:', new_time, '\n')
    else:
        raise Exception
    return


def update_time_betw_get_end_and_execution_end(execute_time, start_get_end):
    # Calc difference between. We should expect trade_open_time to be later.
    start_get_end = datetime.datetime.strptime(start_get_end, '%H:%M:%S.%f')
    diff = execute_time - start_get_end

    # Sanity check
    if diff.total_seconds() < 0 or diff.total_seconds() > 5:
        new_time = 2.5
    else:
        new_time = diff.total_seconds()
    pr.change_time_onthefly(time_ge=new_time)
    print('*** Updated time_betw_get_end_and_execution_end to:', new_time, '\n')
    return


@te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_fixed(0.5) , stop=te.stop_after_attempt(3))
def trade_execution(cycle, trade, total_wins):

    if datetime.datetime.now().second < 10:
        print('Not yet 10 seconds into current minute. Hold on...')
        print('')
        time.sleep(10-datetime.datetime.now().second)

    lookback = pr.lookback_t
    if pr.mlpsvc_datacollect:
        lookback = random.choice([60,120,180,240])

    # We cross validate for best params. We can take as long as we want here.
    best_param, test_range_center = an.cross_val_trading(lookback_t=lookback)
    print('')

    # Check if any params are still default, if so, skip one iteration.
    if checks(trade_params=best_param, params_chk=True) == 4:
        cycle += 1
        return 1 , cycle, trade, total_wins

    # Print params to be used.
    print('Using these params for this cycle:')
    print('[[train, delay, NRMSE, lookback_t, test, ridge]]:')
    for bp in best_param:
        print(bp)
    print('')

    # Build + get new data since cross val may have taken some time.
    # Build data up + get one for the duration that build last took
    picklename, get_one_hour, get_one_minute = gq.get_one_now()

    # Record the time we ended this get_one_now.
    # From here, ACCURACY DECREASES the longer we take to decide and execute a trade.
    start_get_end = datetime.datetime.now()
    print('>>> Time @ start_get_end:', start_get_end.strftime('%H:%M:%S.%f'))
    print('')

    # Load dataframe
    df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = an.lock_and_load(picklename=picklename, lookback=lookback)
    print('Loaded pickle used for prediction for trading ... :', picklename)
    print('Dataframe Statistics:')
    print('>>> Time @ first row:', df['time'].iloc[0])
    print('>>> Time @ last row:', df['time'].iloc[-1])
    print('')

    # Remove all get_one_now.
    files = glob.glob(picklename[:-4]+'*')
    for f in files:
        os.remove(f)

    # Check current min/sec vs min/sec of last row of dataframe.
    time_mismatch = 0
    if checks(df=df, time_mismatch_chk=True) == 5:
        time_mismatch = 1

    # For each set of param in best_param, we do a compute.
    results = []
    for p in best_param:
        results.append(
            an.compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array, warmup=-1, train=p[0], k=p[1],
                            test=p[4], ridge_param=p[5], isTrg=False, isTrading=True))
    print('*** test_range_center used for trade prediction:', test_range_center)

    # Consolidate results. Supports multiple results from multiple parameters.
    action_count = 0
    predicted_delta = 0
    for result in results:
        predicted_delta += np.mean(result)
        for item in result:
            if item > 0:
                action_count += 1

    predicted_delta = predicted_delta / len(results)
    print('Action:', action_count, 'Total: ', len(results)*len(results[0]))
    print('Mean of predicted delta:', predicted_delta)
    print('')

    action = None
    direction_agree = 0
    if action_count == 0:
        action = 0
        direction_agree += 1
    if action_count == len(results)*len(results[0]):
        action = 1
        direction_agree += 1

    # Trade Execution
    if time_mismatch != 1:
        print('Time Match?: YES')

        if action == 1 or action == 0:
            print('Direction agreement?: YES')

            if abs(predicted_delta) > pr.pred_delta_threshold:
                print('pred_delta_threshold met?: YES')

                # Hit it!
                executed_time = None
                if action == 1:
                    executed_time = execute(signal=1)
                if action == 0:
                    executed_time = execute(signal=0)
                trade += 1

                # Get trade record + update timings. 2 methods to get record.
                # Check if what we got is legit, if not, try another approach.
                trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=2)
                if len(trade_opened_time) != 12 and ':' not in trade_opened_time:
                    trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=1)
                update_time_betw_get_end_and_execution_end(executed_time, start_get_end.strftime('%H:%M:%S.%f'))
                update_time_betw_execution_end_and_trade_open(executed_time, trade_opened_time)
                total_wins += won

                # Save datapoint, post-trade for MLP-SVC exploration.
                if pr.mlpsvc_datacollect and len(best_param) == 1:
                    mse.data_collection(consolidated_array, np.arange(2,120,1), action, predicted_delta, best_param[0][2], best_param[0][1],
                                        test_range_center, lookback, won)

    # Output why we did not take action.
    if time_mismatch == 1:
        print('No execution - Time Match?: NOS')
    if direction_agree == 0:
        print('No execution - Direction agreement?: NO')
    if abs(predicted_delta) < pr.pred_delta_threshold:
        print('No execution - pred_delta_threshold met?: NO')

    # Check if traded enough. Stop when total trade is reached or break even.
    # 1/(p+1) is the minimum win fraction to break even at p payout.
    # For Quickler, p is 0.8.
    if trade == pr.total_trade or total_wins > 1/(pr.payout+1) * trade:
        end(cycle,trade, total_wins)
        return -1, cycle, trade, total_wins

    print('')
    # Go onto next cycle.
    cycle += 1
    return 0 , cycle, trade, total_wins


if __name__ == '__main__':
    # Initialize logging
    # tradelog_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tradelog_name = "trade_execution_"+str(tradelog_datetime)+".log"
    # logging.basicConfig(filename=tradelog_name, level=logging.DEBUG) # https://www.loggly.com/?p=76609

    multiprocessing.freeze_support() ; cycle = 1 ; trade = 0 ; total_wins = 0
    while True:
        print('\n/****************************************************************************/\n')
        print('Cycle # : ', cycle) ; print(total_wins, ' wins / ', trade ,' trades  \n')

        # Check to see if we are close to change of day.
        if checks(day_change_chk=True) == 1:
            # Sleep for few minutes for data to build up.
            time.sleep((pr.lookback_t+1)*60)
            # Change date by refreshing.
            ut.refresh()
        if cycle == 1:
            if checks(cycle, trade_start_chk=True) == 2:
                break

        # We time our getting of data, do cross val do prediction and execute trade if within NRMSE
        trade_stats = trade_execution(cycle=cycle, trade=trade, total_wins=total_wins)

        if trade_stats[0] == -1:
            break
        if trade_stats[0] == 1 or trade_stats[0] == 0:
            cycle = trade_stats[1]
            trade = trade_stats[2]
            total_wins = trade_stats[3]