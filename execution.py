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
    if diff.total_seconds() < 0 or diff.total_seconds() > 3:
        new_time = 1.5
    else:
        new_time = diff.total_seconds()

    if len(trade_open_time) == 8:
        # If we are missing decimals, we add a bit of margin.
        pr.change_time_onthefly(time_et=new_time+0.5)
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
    if diff.total_seconds() < 0 or diff.total_seconds() > 2:
        new_time = 0.5
    else:
        new_time = diff.total_seconds()
    pr.change_time_onthefly(time_ge=new_time)
    print('*** Updated time_betw_get_end_and_execution_end to:', new_time, '\n')
    return

# @te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_fixed(0.5) , stop=te.stop_after_attempt(10))
# def trade_execution1(cycle, trade, total_wins):
#
#     # We cross validate for best params.
#     best_param, test_range_center = an.cross_val_trading(lookback_t=pr.lookback_t)
#
#     # Check if any params are still default, if so, skip one iteration.
#     if checks(trade_params=best_param, params_chk=True) == 4:
#         cycle += 1
#         return 1 , cycle, trade, total_wins
#
#     # Print params to be used.
#     print('Using these params for this cycle:')
#     print('[[train, delay, NRMSE, lookback_t, test, ridge]]:')
#     for bp in best_param:
#         print(bp)
#
#     # Build + get new data since cross val may have taken some time.
#     # Build data up + get one for the duration that build last took
#     picklename, get_one_hour, get_one_minute, get_one_second = gq.get_one_now()
#
#     start_get_end = datetime.datetime.now().strftime('%H:%M:%S.%f')
#
#     # Load dataframe
#     df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = an.lock_and_load(picklename=picklename, lookback=pr.lookback_t_min, seconds=get_one_second)
#     print('Loaded pickle used for prediction for trading ... :', picklename)
#     print('Dataframe Statistics:')
#     print('>>> Time @ first row:', df['time'].iloc[0])
#     print('>>> Time @ last row:', df['time'].iloc[-1])
#     print('Mean of quote - 20 time pts:', np.mean(df['quote'].iloc[-20:]))
#     print('Std Dev of quote - 20 time pts:', np.std(df['quote'].iloc[-20:]))
#
#     # Check current min/sec vs min/sec of last row of dataframe.
#     time_mismatch = 0
#     if checks(df=df, time_mismatch_chk=True) == 5:
#         time_mismatch = 1
#
#     # Compute what trade actions to take
#     # Returns a list of test_predictions_quote delta when trading.
#     # ^ between last data and test_range_center + 1 second.
#     results = []
#     for p in best_param:
#         results.append(
#             an.compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array, warmup=-1, train=p[0], k=p[1],
#                             test=p[4], ridge_param=p[5], isTrg=False, isTrading=True))
#     print('*** test_range_center used for trade prediction:', test_range_center)
#
#     action = 0
#     for result in results:
#         if result > 0:
#             action += 1
#
#     # Calc basic stats of delta of price prediction.
#     mean_pred_delta = np.mean(results)
#     stdev_pred_delta = np.std(results)
#     print('All results pred delta:', results)
#     print('Average of all results pred delta:', mean_pred_delta)
#     print('Std Dev of all results pred delta:', stdev_pred_delta, '\n')
#
#     # Force a trade for testing purposes.
#     if pr.test_force_trade:
#         time_mismatch = 0
#         action = 0
#         mean_pred_delta = 999
#
#     # Trade Execution
#     if time_mismatch != 1:
#         print('Time Mismatch: NO')
#
#         if action == len(results) or action == 0:
#             print('Direction agreement: YES ')
#
#             # Check if mean of delta is above threshold.
#             if abs(mean_pred_delta) > pr.pred_delta_threshold:
#                 print('Threshold met: YES')
#
#                 # Hit it!
#                 if action == len(results):
#                     executed_time = execute(signal=1)
#                 if action == 0:
#                     executed_time = execute(signal=0)
#                 trade += 1
#
#                 # Get trade record + update timings. 2 methods to get record.
#                 # Check if what we got is legit, if not, try another approach.
#                 update_time_betw_get_end_and_execution_end(executed_time, start_get_end)
#                 trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=2)
#                 if len(trade_opened_time) != 12 and ':' not in trade_opened_time:
#                     trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=1)
#                 update_time_betw_execution_end_and_trade_open(executed_time, trade_opened_time)
#                 total_wins += won
#
#                 # Sleep for a random period of time.
#                 if pr.random_sleep:
#                     sleeptime = np.random.uniform(60*pr.random_sleep_min, 60*pr.random_sleep_max)
#                     print("Just put in a trade. Sleeping for:", sleeptime, "seconds")
#                     time.sleep(sleeptime)
#
#     # Output why we did not take action.
#     if time_mismatch == 1: print('No execution - Time Mismatch: YES')
#     if 0 < action < len(results): print('No execution - Direction agreement: NO')
#     if abs(mean_pred_delta) < pr.pred_delta_threshold: print('No execution - Threshold met: NO')
#
#     # Check if we have traded enough.
#     if trade == pr.total_trade:
#         end(cycle,trade, total_wins)
#         return -1, cycle, trade, total_wins
#
#     # Go onto next cycle.
#     cycle += 1
#     return 0 , cycle, trade, total_wins


@te.retry(retry=te.retry_if_exception_type(Exception), wait=te.wait_fixed(0.5) , stop=te.stop_after_attempt(10))
def trade_execution(cycle, trade, total_wins):

    # Build + get new data since cross val may have taken some time.
    # Build data up + get one for the duration that build last took
    picklename, get_one_hour, get_one_minute, get_one_second = gq.get_one_now()

    start_get_end = datetime.datetime.now()

    # Load dataframe
    df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = an.lock_and_load(picklename=picklename, lookback=pr.lookback_t, seconds=get_one_second)
    print('Loaded pickle used for prediction for trading ... :', picklename)
    print('Dataframe Statistics:')
    print('>>> Time @ first row:', df['time'].iloc[0])
    print('>>> Time @ last row:', df['time'].iloc[-1])
    print('>>> Time @ start_get_end:', start_get_end)
    print('')

    # Check current min/sec vs min/sec of last row of dataframe.
    time_mismatch = 0
    if checks(df=df, time_mismatch_chk=True) == 5:
        time_mismatch = 1

    # Compute what trade actions to take
    # Returns a list of test_predictions_quote delta when trading.
    # ^ between last data and test_range_center + 1 second.
    results = []
    for d in pr.delay_range:
        for t in pr.train_range:
            results.append(
                an.compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array,
                                warmup=-1, train=t, k=d, test=pr.test_time,
                                ridge_param=pr.ridge_range[0], isTrg=False, isTrading=True))

    action_count = 0
    mean_delta = 0
    for result in results:
        mean_delta += np.mean(result)
        for item in result:
            if item > 0:
                action_count += 1

    mean_delta = mean_delta / len(results)
    print('Action:', action_count, 'Total: ', len(results)*len(results[0]))
    print('Mean of predicted delta:', mean_delta)
    print('')

    # if action_count == len(results)*len(results[0]) and mean_delta > 0:
    #     action = 1
    # if action_count == 0 and mean_delta < 0:
    #     action = 0
    random_action = np.random.randint(0,2)
    action_fraction = action_count / (len(results)*len(results[0]))

    action = random_action
    # svc_action = mse.svc_trading(consolidated_array, random_action, action_fraction, mean_delta, pr.delay_range)
    # print('svc action vs random_action:', svc_action, random_action)
    # if svc_action == random_action:
    #     action = svc_action

    # Trade Execution
    direction_agree = 0
    if time_mismatch != 1:
        print('Time Match?: YES')

        if action == 1 or action == 0:
            direction_agree += 1
            print('Direction agreement?: YES')

            if abs(mean_delta) > pr.pred_delta_threshold:
                print('pred_delta_threshold met?: YES')

                # Hit it!
                if action == 1:
                    execute(signal=1)
                if action == 0:
                    execute(signal=0)
                trade += 1

                end_execution = datetime.datetime.now()
                diff = end_execution - start_get_end
                time_betw_get_and_exec = diff.total_seconds()
                print('>>> Time elapsed betw get_one end & execution end:', time_betw_get_and_exec)
                print('')

                # Get trade record + update timings. 2 methods to get record.
                # Check if what we got is legit, if not, try another approach.
                trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=2)
                if len(trade_opened_time) != 12 and ':' not in trade_opened_time:
                    trade_opened_time, won = get_latest_trade_record(isPrint=True, approach=1)
                total_wins += won
                trade_outcome = won
                mse.data_collection(consolidated_array, time_betw_get_and_exec, action, action_fraction, mean_delta, pr.test_time, np.arange(1,300,5), trade_outcome)
                gq.build_dataset_last_t_minutes(t=pr.lookback_t_min)


    # Output why we did not take action.
    if time_mismatch == 1:
        print('No execution - Time Match?: NOS')
    if direction_agree == 0:
        print('No execution - Direction agreement?: NO')
    if abs(mean_delta) < pr.pred_delta_threshold:
        print('No execution - pred_delta_threshold met?: NO')

    # Check if we have traded enough.
    if trade == pr.total_trade or total_wins > 1 * trade:
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
            if pr.olymp_day is not None or pr.olymp_day != ():
                # Change date with PyAutoGUI only if we have date
                ut.date_changer()
                gq.build_dataset_last_t_minutes(t=pr.lookback_t)
            else:
                break
        if cycle == 1:
            if checks(cycle, trade_start_chk=True) == 2:
                break
            gq.build_dataset_last_t_minutes(t=pr.lookback_t)

        # We time our getting of data, do cross val do prediction and execute trade if within NRMSE
        trade_stats = trade_execution(cycle=cycle, trade=trade, total_wins=total_wins)

        if trade_stats[0] == -1:
            break
        if trade_stats[0] == 1 or trade_stats[0] == 0:
            cycle = trade_stats[1]
            trade = trade_stats[2]
            total_wins = trade_stats[3]