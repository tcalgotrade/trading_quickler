import numpy as np
import get_quote as gq
import analysis as an
import pyautogui as pag
import datetime
import time
import multiprocessing
import pickle
import params as pr
import logging

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

    print('Time @ execute complete : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))
    return


def end(cycle, trade):

   # Print some numbers
    print('Final Cycle # : ', cycle)
    print('Final Trade executed: ', trade)

    return


def checks(trade_params=None, df=None, start1_time_second= None, start2_time_second = None,
           day_change_chk=False, trade_start_chk=False, cycle1_warmup_chk=False, params_chk=False, min_mismatch_chk=False, sec_mismatch_chk=False,
           timed_start1_chk=False, timed_start2_chk=False):


    """ All checks leading up to trade execution checks."""

    if day_change_chk:
        # Check if current time is 2359. If so, break and stop trading.
        # There is currently no way to change date via PyAutoGUI.
        if datetime.datetime.now().hour == 23 and datetime.datetime.now().minute == 59:
            print('We have entered a new day.')
            return 1

    # Pop a prompt to make sure manual setup is good.
    if trade_start_chk:
        setup_check = pag.confirm("Is Olymptrade browser window maximized and setup?")
        # Print key params for logging
        print('Started at:', datetime.datetime.now(), '\n')
        print('Cross Val Params:')
        print('warm_range:', pr.warm_range,'\ntrain_range:', pr.train_range,'\ndelay_range:', pr.delay_range,'\ntest_range:', pr.test_range)
        print('ridge_range:',pr.ridge_range,'\nthreshold_test_nrmse:',pr.threshold_test_nrmse, '\n')
        print('Trading Params:')
        print('total_trade:', pr.total_trade,'\nlookback_t:', pr.lookback_t,'\nadjusted_start1_time_second:', pr.adjusted_start1_time_second)
        print('test_points:',pr.test_points,'\npred_delta_threshold:',pr.pred_delta_threshold,'\npercent_correct_dir:',pr.percent_correct_dir)
        print('time_to_get_quote_seconds:',pr.time_to_get_quote_seconds, '\ntarget_start1_time_second:',pr.target_start1_time_second,'\ntarget_start2_time_second:', pr.target_start2_time_second, '\n')
        if setup_check == 'Cancel':
            print('Cancelled by user.')
            return 2

    if cycle1_warmup_chk:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t, isTrading=1)
        print('First cycle warmed up.')
        print('/****************************************************************************/\n')
        return 3

    if params_chk and trade_params is not None:
        # Check if params are still default, if so, skip one iteration.
        if trade_params == [[0, 0, 0, 1.0], [0, 0, 0, 1.0]]:
            print('Params still default. Go on to next iteration.')
            print('/****************************************************************************/\n')
            return 4

    if min_mismatch_chk and df is not None:
        # Check current min vs current min of last row of dataframe.
        # If there's more than a diff of 1, sth went wrong during get_quote: do nothing.
        current_min = datetime.datetime.now().minute
        if abs(float(df['time'].iloc[-1].minute) - float(current_min)) > 1:
            print('Minute mismatch between data & current time:', current_min, ' vs ', df['time'].iloc[-1].minute)
            return 5

    if sec_mismatch_chk and df is not None:
        # Check current time now and the last time in data.
        # Platform have shown that it could give future data even though we are not in the future yet.
        # A refresh of the site might be needed.
        last_sec = df['time'].iloc[-1].second
        now_sec = datetime.datetime.now().second
        print('Last row second:', last_sec, 'Current sec:', now_sec)
        return

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

    if timed_start2_chk and start2_time_second is not None:
        if datetime.datetime.now().second < start2_time_second:
            print('Waiting', start2_time_second - datetime.datetime.now().second, 'seconds for right start')
            time.sleep(start2_time_second - datetime.datetime.now().second)
        else:
            print('Not in time for start2.',)
        return

    return


def trade_execution(cycle, trade, which_start):

    picklename = an.cross_val_trading(t=pr.lookback_t)

    # Open best param pickle files
    with open('params_current_trading', 'rb') as f:
        params = pickle.load(f)
    print('We have these params to use:', params)

    # Check if params are still default, if so, skip one iteration.
    if checks(trade_params=params, params_chk=True) == 4:
        cycle += 1
        return 1 , cycle, trade

    # Load dataframe
    df = an.load(picklename=picklename, isInfo=0)
    print('Loaded pickle for prediction for trading ... :', picklename)
    print('Dataframe Statistics:')
    print('Time @ first row:', df['time'].iloc[0])
    print('Time @ last row:', df['time'].iloc[-1])
    print('Mean of quote history:', np.mean(df['quote']))
    print('Std Dev of quote history:', np.std(df['quote']))

    # Check current min/sec vs min/sec of last row of dataframe.
    min_mismatch = 0
    if checks(df=df, min_mismatch_chk=True) == 5:
        min_mismatch = 1
    checks(df=df, sec_mismatch_chk=True)

    # Compute what trade actions to take
    # Returns a tuple when trading : (action to take: 1:buy up,0:buy down,-1:do nth, pred_delta)
    results = []
    for test_point in pr.test_points:
        results.append(an.compute_ngrc(df, isDebug=0, isInfo=0,
                                       warmup=0, train=params[1][0], k=params[1][1], test=test_point,
                                       ridge_param=pr.ridge_range[0], which_start=which_start,
                                       isTrading=1, isTrg=0))
    print('Time @ compute complete : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Consolidate results for trade execution.
    all_pred_delta = []
    for result in results:
        all_pred_delta.append(round(result[1],2))

    action_sum = 0
    for result in results:
        action_sum += result[0]

    # Calc basic stats of delta of price prediction.
    mean_pred_delta = np.mean(all_pred_delta)
    stdev_pred_delta = np.std(all_pred_delta)
    print('All results pred delta:', all_pred_delta)
    print('Average of all results pred delta:', mean_pred_delta)
    print('Std Dev of all results pred delta:', stdev_pred_delta)

    # Trade Execution
    if action_sum == 0 or action_sum == len(results) or action_sum == -len(results):
        print('Direction agreement: YES ')
        # Check if mean of delta is above threshold.
        if abs(mean_pred_delta) > pr.pred_delta_threshold and min_mismatch != 1:
            print('Threshold: YES Minute mismatch: NO')
            if params[0][2] >= pr.lookback_t * len(pr.test_range) * len(pr.warm_range) * pr.percent_correct_dir:
                print('Percent_correct_dir: YES')
                execute(signal=results[0][0])
                # Check if agreement is no action
                if results[0][0] != -1:
                    trade += 1
                    if trade == pr.total_trade:
                        end(cycle,trade)
                        return -1, cycle, trade
            else:
                print('No execution - Percent_correct_dir: NO')
        else:
            print('No execution - Threshold: NO or Minute mismatch: YES')
    else:
        print('No execution - Direction agreement: NO ')
    cycle += 1
    print('/****************************************************************************/\n')
    return 0 , cycle, trade


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


if __name__ == '__main__':
    # Initialize logging
    # tradelog_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tradelog_name = "trade_execution_"+str(tradelog_datetime)+".log"
    # logging.basicConfig(filename=tradelog_name, level=logging.DEBUG) # https://www.loggly.com/?p=76609

    multiprocessing.freeze_support() ; cycle = 1 ; trade = 0 ; start_flag = 1

    while True:
        print('Cycle # : ', cycle) ; print('Trade executed: ', trade, '\n')

        # Some checks to make sure we're good before trading
        if checks(day_change_chk=True) == 1:
            time.sleep(4*60)
            date_changer()
        if cycle == 1:
            if checks(cycle, trade_start_chk=True) == 2:
                break
            checks(timed_start1_chk=True, start1_time_second=pr.adjusted_start1_time_second)
            checks(cycle1_warmup_chk=True)

        # Start1
        if start_flag == 1:
            checks(timed_start1_chk=True, start1_time_second=pr.adjusted_start1_time_second)
        pr.which_start[0] = 1 ; start_flag = 2

        flow_control = trade_execution(cycle=cycle, trade=trade, which_start=pr.which_start[0])

        if flow_control[0] == -1:
            break
        if flow_control[0] == 1 or flow_control[0] == 0:
            cycle = flow_control[1]
            trade = flow_control[2]

        # Start2
        print('Cycle # : ', cycle) ; print('Trade executed: ', trade, '\n')
        if start_flag == 2:
            checks(timed_start2_chk=True, start2_time_second=pr.adjusted_start2_time_second)
        pr.which_start[0] = 2 ; start_flag = 1

        flow_control = trade_execution(cycle=cycle, trade=trade, which_start=pr.which_start[0])

        if flow_control[0] == -1:
            break
        if flow_control[0] == 1 or flow_control[0] == 0:
            cycle = flow_control[1]
            trade = flow_control[2]
