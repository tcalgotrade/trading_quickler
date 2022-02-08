import os.path
import pickle
import numpy as np
import sys
import datetime
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import istarmap
import warnings
import itertools
import get_quote as gq
import re
import glob
import tqdm
import traceback
import params as pr
import utility as ut
import logging as lg


# Display dataframe & arrays in full glory
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Turn off runtime warnings
warnings.filterwarnings('ignore')


def load(picklename, seconds, lookback=pr.lookback_t, isDebug=False):

    # Build and process a dataframe containing t minutes of quote history data.
    # We assume pickle files of past t minutes already existed. This should be handled by main trading loop.
    try:
        # Get hour and min from picklename.
        hour = int(picklename[-4:][0:2])
        minute = int(picklename[-4:][2:4])
        hours_list, minutes_list = ut.hour_min_to_list_t(hour, minute, seconds, t=lookback)

        # We gp through the timings and build up time series of minutes = lookback_t
        unpickled = ''
        hour_front = ut.process_current_datetime(hour=hours_list[0])[0]
        hour_back = ut.process_current_datetime(hour=hours_list[0])[1]
        for minute in minutes_list[0]:
            minute_front = ut.process_current_datetime(min=minute)[2]
            minute_back = ut.process_current_datetime(min=minute)[3]
            current_iter_picklename = picklename[:-4]+hour_front+hour_back+minute_front+minute_back
            unpickled += pd.read_pickle(current_iter_picklename)

        if len(hours_list) == 2:
            # Front
            hour_front = ut.process_current_datetime(hour=hours_list[1])[0]
            hour_back = ut.process_current_datetime(hour=hours_list[1])[1]
            for minute in minutes_list[1]:
                minute_front = ut.process_current_datetime(min=minute)[2]
                minute_back = ut.process_current_datetime(min=minute)[3]
                current_iter_picklename = picklename[:-4]+hour_front+hour_back+minute_front+minute_back
                unpickled += pd.read_pickle(current_iter_picklename)



        # Split it into a proper dataframe
        df = pd.DataFrame([x.split(' ') for x in unpickled.split('\n')])
        # Check if first and last row is valid. If not, we drop it.
        # We want first row to be time. Last row to be quote price.
        # In event that get_quote first row is not time, drop first row
        # Check for 2 rows
        for i in range(0,3):
            if len(df[0].iloc[0]) != 12 or ':' not in df[0].iloc[0]:
                df.drop(df.head(1).index, inplace=True)
            # In event that get_quote last row is time, drop last row
            if len(df[0].iloc[-1]) != 7 or '.' not in df[0].iloc[-1]:
                df.drop(df.tail(1).index, inplace=True)


        # Separate odd and even rows, https://is.gd/w91SzO
        # Not actually used. Left here for future ref.
        time = df.iloc[:-1:2]
        closing_price = df.iloc[1::2]

        # Split raw into 2 col, odd rows containing time stamp is col 1, even rows containing close price is col 2
        df = df[:-1:2].assign(quote=df[1::2].values)  # https://is.gd/zH0lPW

        # Rename col
        df.rename(columns={0: 'time'}, inplace=True)  # https://is.gd/b4WZMj

        # Convert into datetime dtype, https://www.geeksforgeeks.org/?p=267976
        df['time'] = pd.to_datetime(df['time'])  # Note diff betw to_datetime & to_timedelta
        df['time_diff'] = df['time'].diff()  # Need date and time to calc this diff
        df['time'] = df['time'].dt.time  # Now we can safely remove date part, keeping only time.

        # Convert quote to float
        df['quote'] = pd.to_numeric(df['quote'])

        # Compute diff between last & current quote
        df['quote_diff'] = df['quote'].diff()

        # Rearrange cols for tidiness, https://is.gd/QNzlbu
        df = df[['time', 'time_diff', 'quote', 'quote_diff']]

        # Drop of 1st row (oldest data point) as it contains NaN
        df = df.iloc[1:, :]

        # Reset index numbers
        df.reset_index(inplace=True)

        # Delete unwanted "index" col
        df.drop(['index'], inplace=True, axis=1)

        if isDebug:
            print('Data : \n', df, '\n')
            print('Shape of dataframe: \n', df.shape)
            print('Max time diff: ', df['time_diff'].max())
            print('Min time diff: ', df['time_diff'].min())
            print('Avg time diff: ', df['time_diff'].mean())
            print('Std Dev time diff: ', df['time_diff'].std())
            print('Max quote: ', df['quote'].max())
            print('Min quote: ', df['quote'].min())
            print('Avg quote: ', df['quote'].mean())
            print('Std Dev quote: ', df['quote'].std())
            print('Types of our cols: \n', df.dtypes, '\n')

    except Exception:
        print('File loading threw an exception: ... ')
        print(traceback.format_exc())
        return

    return df


def compute_ngrc(df, isDebug, isInfo, warmup, train, k, test, ridge_param, isTrg=0, isTrading=0):

    # total variance of data
    total_var = np.var(df['quote'])

    # Get rows and cols of dataframe
    r, c = df.shape

    # time step
    dt = np.round(df['time_diff'].dt.total_seconds().mean(),2)
    # units of time to warm up NVAR. need to have warmup_pts >= 1, in seconds
    warmup = warmup
    # units of time to train for, in seconds
    traintime = train
    # units of time to test for, in seconds
    testtime = test
    # ridge parameter for regression
    ridge_param = ridge_param

    # discrete-time versions of the times defined above
    if isTrg:
        traintime_pts = round(traintime / dt)
        testtime_pts = round(testtime / dt)
        warmup_pts = round(warmup / dt)
        if warmup == 0: warmup_pts = r - (traintime_pts+testtime_pts)
        warmtrain_pts = warmup_pts + traintime_pts
        maxtime_pts = warmtrain_pts + testtime_pts

    if isTrading:
        # When trading, we do not need test data.
        traintime_pts = round(traintime / dt) # Number of 'dt' to trg on
        testtime_pts = round(testtime / dt) # Number of 'dt' to predict ahead
        warmup_pts = round(warmup / dt)
        if warmup == 0: warmup_pts = r - traintime_pts # Number of 'dt' to shift so we trg on latest data we have, We ignore param 'warm'
        warmtrain_pts = warmup_pts + traintime_pts # Number of 'dt' sum of warm and train
        maxtime_pts = warmtrain_pts # Number of dt sum of warm and train, less test because we're trading now


    # input dimension
    d = 2
    # number of time delay taps
    k = k
    # size of linear part of feature vector
    dlin = k * d
    # size of nonlinear part of feature vector
    dnonlin = int(dlin * (dlin + 1) / 2)
    # total size of feature vector: constant + linear + nonlinear
    dtot = 1 + dlin + dnonlin

    ##
    ## NVAR
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin, maxtime_pts))

    """ 
    Fill in the linear part of the feature vector for all times
    Colon here used to 'select' a bunch of rows in matric x : i.e 0:2 => select rows 0 and 1, exclude 2
    'j' here is used as index for time; in matrix 'x', time is along axis=1 (cols)
    When d=3 & delay=0, d * delay : d * (delay + 1) => 0:3 , i.e. rows 0,1,2 are selected 
    Since delay increments with the for-loop, say d=3, delay=1 , => 3:6 , i.e. rows 3,4,5 selected
    Diagonal of x all have same value: value at t=0 
    Bottom half of diagonal all 0
    When d>1 , need to think how to structure dataframe 
    Seems like s=1 , where 's' is the spacing between time step
    Is index 0 suppose to be oldest or newest time in data? : oldest
    """
    try:
        consolidated_array = np.array([df['quote'].to_numpy(),df['quote_diff'].to_numpy()])
        for delay in range(k):
            for j in range(delay, maxtime_pts):
                x[d * delay: d * (delay + 1), j] = consolidated_array[:, j-delay]
    except Exception:
        print('Something went wrong when computing : x ')
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param)
        print('x shape:',x.shape)
        print('Dataframe shape', df.shape)
        print(traceback.format_exc())
        return -1, 0


    """
    Fill in the non-linear part
    How to edit this for RBF or higher order of polynomial?
    out_train[dlin + 1 + cnt] : refers to the row number 'dlin + 1 + cnt'
    x[row, warmup_pts - 1:warmtrain_pts - 1] * x[column, warmup_pts - 1:warmtrain_pts - 1]
        >> generating quadratric terms 
    """
    try:
        """
        create an array to hold the full feature vector for training time
        (use ones so the constant term is already 1)
        """
        out_train = np.ones((dtot, traintime_pts))

        """
        Copy over the linear part (shift over by one to account for constant)
        1st row is all ones , for bias term
        Next dlin # of rows has just training points (i.e. less warmup points), select out of 'x'
        """
        out_train[1:dlin + 1, :] = x[:, warmup_pts - 1:warmtrain_pts - 1]

        cnt = 0
        for row in range(dlin):
            for column in range(row, dlin):
                # shift by one for constant
                out_train[dlin + 1 + cnt] = x[row, warmup_pts - 1:warmtrain_pts - 1] \
                                            * x[column, warmup_pts - 1:warmtrain_pts - 1]
                cnt += 1
    except Exception:
        print('Something went wrong when computing : out_train ')
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param)
        print('x shape:',x.shape)
        print('out_train:',out_train.shape)
        print('Dataframe shape', df.shape)
        print('dtot, traintime_pts, dlin, warmup_pts, warmtrain_pts, dt:', dtot, traintime_pts, dlin, warmup_pts, warmtrain_pts, dt)
        print(traceback.format_exc())
        return -1, 0


    """
    Ridge regression: compute W_out 
    This is equation 3 of NG-RC paper
    '@' here invokes a matrix multiplication, https://is.gd/r8bbR7 
    """
    try:
        W_out = (x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1:warmtrain_pts - 1]) \
                @ out_train[:, :].T \
                @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(dtot))
    except Exception:
        print('Something went wrong when computing : W_out ')
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param)
        print('Dataframe shape', df.shape)
        print(traceback.format_exc())
        return -1, 0

    # apply W_out to the training feature vector to get the training output
    try:
        x_predict = x[0:d, warmup_pts - 1:warmtrain_pts - 1] + W_out @ out_train[:, 0:traintime_pts]
    except Exception:
        print('Something went wrong when computing : x_predict ')
        print(traceback.format_exc())
        return -1, 0

    # calculate NRMSE between true quote and training output
    trg_nrmse = np.sqrt(np.mean((x[0:d, warmup_pts:warmtrain_pts] - x_predict[:, :]) ** 2) / total_var)

    # create a place to store feature vectors for prediction
    out_test = np.zeros(dtot)  # full feature vector
    x_test = np.zeros((dlin, testtime_pts))  # linear part

    try:
        # copy over initial linear feature vector
        x_test[:, 0] = x[:, warmtrain_pts - 1]

        # do prediction
        for j in range(testtime_pts - 1):
            # copy linear part into whole feature vector
            out_test[1:dlin + 1] = x_test[:, j]  # shift by one for constant
            # fill in the non-linear part
            cnt = 0
            for row in range(dlin):
                for column in range(row, dlin):
                    # shift by one for constant
                    out_test[dlin + 1 + cnt] = x_test[row, j] * x_test[column, j]
                    cnt += 1
            # fill in the delay taps of the next state
            x_test[d:dlin, j + 1] = x_test[0:(dlin - d), j]
            # do a prediction
            x_test[0:d, j + 1] = x_test[0:d, j] + W_out @ out_test[:]
    except Exception:
        print('Something went wrong when computing : x_test')
        print('x_test shape:',x_test.shape)
        print('warmtrain_pts:',warmtrain_pts)
        print(traceback.format_exc())
        return -1, 0

    try:
        if isTrg:
            # Calculate NRMSE between ground_truth and test_predictions.
            # Only makes sense if we are training/cross val
            test_nrmse = np.sqrt(np.mean(
                (x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts - 1] - x_test[0:d, 0:testtime_pts]) ** 2) / total_var)
        else:
            test_nrmse = 0
    except Exception:
        print(traceback.format_exc())
        return -1, 0

    try:
        # Pull out relevant data points for ground and test prediction + compute relevant items between 1st pt and last pt.
        # We distinguish whether we're cross val or trading.
        # When cross val: we're interested in test predictions instead of trg predictions.
        # When trading, we're just looking at the test predictions, i.e. predicting ahead to future and output actions to take.
        if isTrg:
            ground_truth = x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts - 1]
            test_predictions = x_test[0:d, 0:testtime_pts]
            # We compute price diff between then and now
            ground_truth_quote_delta = ground_truth[0, -1] - ground_truth[0, 0]
            test_predictions_quote_delta = test_predictions[0, -1] - test_predictions[0, 0]
            # We sum up price diffs to find out whether we will be up or down at desired test time.
            ground_truth_quotediff_sum = np.sum(ground_truth[1,1:])
            test_predictions_quotediff_sum = np.sum(test_predictions[1,1:])
        if isTrading:
            test_predictions = x_test[0:d, 0:testtime_pts]
            # We compute price diff between then and now
            test_predictions_quote_delta = test_predictions[0, -1] - test_predictions[0, 0]
            # We sum up price diffs to find out whether we will be up or down at desired test time.
            test_predictions_quotediff_sum = np.sum(test_predictions[1,1:])

        if isDebug:
            print('warm:', warmup)
            print('train:', train)
            print('delay:', k)
            print('test:', test)
            print('ridge:', ridge_param)
            print('isTrg:', isTrg)
            print('isTrading:', isTrading,'\n')
            print('Dataframe:\n',df,'\n')
            print('x : \n', x, '\n')
            print('x shape:',x.shape, '\n')
            print('out_train: \n', out_train, '\n')
            print('W_out: \n', W_out, '\n')
            print('x_predict: \n', x_predict, '\n')
            print('x_predict shape:', x_predict.shape,'\n')
            print('x_test: \n', x_test, '\n')
            print('x_test shape:', x_test.shape,'\n')
            print('ground_truth: \n', ground_truth)
            print('test_predictions: \n', test_predictions, '\n')
            print('Shape of ground_truth:', ground_truth.shape)
            print('Shape of test_predictions:', test_predictions.shape)
            print('ground_truth_quote_delta:', ground_truth_quote_delta)
            print('test_predictions_quote_delta:', test_predictions_quote_delta)
            print('ground_truth_quotediff_sum:', ground_truth_quotediff_sum)
            print('test_predictions_quotediff_sum:', test_predictions_quotediff_sum)

        if isInfo:
            if isTrg:
                print('Avg Time Diff : ', df['time_diff'].mean().total_seconds())
                print('training nrmse: ' , trg_nrmse)
                print('test nrmse: ' , test_nrmse)
                print('Shape of ground_truth:', ground_truth.shape)
                print('Shape of test_predictions:', test_predictions.shape)
                print('ground_truth: \n', ground_truth)
                print('test_predictions: \n', test_predictions, '\n')
                print('ground_truth_quote_delta:', ground_truth_quote_delta)
                print('test_predictions_quote_delta:', test_predictions_quote_delta)
                print('ground_truth_quotediff_sum:', ground_truth_quotediff_sum)
                print('test_predictions_quotediff_sum:', test_predictions_quotediff_sum)
            if isTrading:
                print('training nrmse: ' , trg_nrmse)
                print('Shape of test_predictions:', test_predictions.shape)
                print('test_predictions: \n', test_predictions, '\n')
                print('test_predictions_quote_delta:', test_predictions_quote_delta)
                print('test_predictions_quotediff_sum:', test_predictions_quotediff_sum)

        # Return actions to do. 0 is buy down. 1 is buy up. -1 is do nothing.
        if isTrg:
            if ground_truth_quote_delta < 0 and test_predictions_quote_delta < 0 \
                    and ground_truth_quotediff_sum < 0 and test_predictions_quotediff_sum < 0:
                return 0, trg_nrmse, test_nrmse, ground_truth_quote_delta, test_predictions_quote_delta, \
                       ground_truth_quotediff_sum, test_predictions_quotediff_sum
            if ground_truth_quote_delta > 0 and test_predictions_quote_delta > 0 \
                    and ground_truth_quotediff_sum > 0 and test_predictions_quotediff_sum > 0:
                return 1, trg_nrmse, test_nrmse, ground_truth_quote_delta, test_predictions_quote_delta, \
                       ground_truth_quotediff_sum, test_predictions_quotediff_sum
            else:
                return -1, trg_nrmse, test_nrmse, ground_truth_quote_delta, test_predictions_quote_delta, \
                       ground_truth_quotediff_sum, test_predictions_quotediff_sum

        if isTrading:
            if test_predictions_quote_delta < 0 and test_predictions_quotediff_sum < 0:
                return 0, trg_nrmse, test_predictions_quote_delta, test_predictions_quotediff_sum
            if test_predictions_quote_delta > 0 and test_predictions_quotediff_sum > 0:
                return 1, trg_nrmse, test_predictions_quote_delta, test_predictions_quotediff_sum
            else:
                return 1, trg_nrmse, test_predictions_quote_delta, test_predictions_quotediff_sum

    except Exception:
        print(traceback.format_exc())
        return -1, 0


def cross_val_ngrc(picklename, seconds, warm, train, delay, test, ridge, threshold_test_nrmse, lookback_t):

    # Load dataframe from pickle
    df = load(picklename=picklename,lookback=lookback_t ,seconds=seconds)

    # Get current date
    date = datetime.datetime.now().strftime("%d%m%Y")

    # Compute NG-RC result and normalized RMSE
    result = compute_ngrc(df, isDebug=0, isInfo=0, warmup=warm, train=train, k=delay, test=test, ridge_param=ridge,
                          isTrg=1, isTrading=0)

    # When result is to take an action, check if within NRMSE threshold. If so, we save param set to pickle.
    if result[0] == 1 or result[0] == 0:
        if result[2] < threshold_test_nrmse:

            param = (picklename, seconds, warm, train, delay, test, ridge,
                      np.around(result[1], 4), np.around(result[2], 4), threshold_test_nrmse, lookback_t)

            # Save cross val result
            with open(pr.data_store_location + date + '/cross_val/'+ str(train) + '-' + str(delay) + '-' + str(test) + '-' + str(warm) + '-' + str(ridge) + '-' + str(lookback_t) + '-' + str(round(result[2],2)) + '-' + picklename[-4:] ,'wb') as f:
                pickle.dump(param, f)

    return


def cross_val_multiproc(params):
    # Multiprocessing with starmap + Progress Bar
    try:
        with Pool() as pool:
            for _ in tqdm.tqdm(pool.istarmap(cross_val_ngrc, params),
                               total=len(params)):
                pass
    except Exception:
        print('Multiproc threw exception.')
        print(traceback.format_exc())
    return


def cross_val_trading(t):

    # Get date & time
    start_time = now = datetime.datetime.now()
    print('>>> Time @ Cross Val start : ', start_time.strftime("%H:%M:%S.%f"))

    # We have already warmed up prior to cycle start. So we now get one quote at current time.
    picklename, get_one_hour, get_one_minute, get_one_second = gq.get_one_now()

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/'):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/')

    # Check if cross val dir created. if not, create it.
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val'):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val')

    # Get all possible combinations of params
    lookback_t_range = range(pr.lookback_t_min, t+1)
    bag_of_params = list(itertools.product([picklename], [get_one_second], pr.warm_range, pr.train_range, pr.delay_range, pr.test_range, pr.ridge_range, pr.threshold_test_nrmse, lookback_t_range))
    print('# of combinations:', len(bag_of_params))

    # Remove contents from last cross_val. We start anew each cross val during trading. No storing of files between cross vals
    files = glob.glob(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val/*')
    for f in files:
        os.remove(f)

    # Cross val with multiprocessing for speed!
    cross_val_multiproc(params=bag_of_params)

    # Find best params. We save the one with lowest test NRMSE
    # [[train, delay, NRMSE, lookback_t, test]]
    best_param = [[0,0,1.,0,0],[0,0,1.,0,0],[0,0,1.,0,0]]
    # Open every param pickle file in cross_val folder
    for file in os.listdir(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val'):
        # Load pickle
        with open(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val/'+file, 'rb') as f:
            current_param = pickle.load(f)
        # Init nrmse value for compare & clarity.
        current_nrmse = current_param[8]
        # Save 2 sets of lowest NRMSE
        if current_nrmse < best_param[0][2]:
            best_param[0][0] = current_param[3]
            best_param[0][1] = current_param[4]
            best_param[0][2] = current_nrmse
            best_param[0][3] = current_param[3]
            best_param[0][4] = current_param[5]
        if best_param[1][2] > current_nrmse > best_param[0][2]:
            best_param[1][0] = current_param[3]
            best_param[1][1] = current_param[4]
            best_param[1][2] = current_nrmse
            best_param[1][3] = current_param[3]
            best_param[1][4] = current_param[5]
        if best_param[0][2] > current_nrmse > best_param[1][2]:
            best_param[2][0] = current_param[3]
            best_param[2][1] = current_param[4]
            best_param[2][2] = current_nrmse
            best_param[2][3] = current_param[3]
            best_param[2][4] = current_param[5]

    if pr.force_manual_cross_val_trading:
        print('2 sets of best param [train, delay, NRMSE, lookback_t, test]:', best_param)

    print('Cross Val took this amount of time:', datetime.datetime.now()-start_time)
    print('>>> Time @ Cross Val end : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    return best_param, picklename, get_one_second


def cross_val_manual():
    print(' >>> Time @ Start : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Get filenames
    dates = ['04022022']  # [26012022, 27012022, 28012022]
    hours = [14]
    mins = range(45, 46)
    files = ut.filenames(dates=dates, hours=hours, mins=mins, files=[])

    # Get all possible combi of params
    # Get all possible combinations of params
    bag_of_params = list(itertools.product(files, pr.warm_range, pr.train_range, pr.delay_range, pr.test_range, pr.ridge_range, pr.threshold_test_nrmse))
    print('# of params:', len(bag_of_params))

    cross_val_multiproc(params=bag_of_params)
    print(' >>> Time @ End : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    return


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Force a manual cross val for trading
    if pr.force_manual_cross_val_trading:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t,isTrading=0)
        cross_val_trading(t=pr.lookback_t)


    # Quick test load.
    if pr.test_load_function:
        df = load(picklename=pr.data_store_location + '08022022/0847', t=pr.lookback_t ,seconds=15, isDebug=True)

    # Quick test compute
    if pr.test_compute_function:
        df = load(picklename=pr.data_store_location + '08022022/1359', t=pr.lookback_t ,seconds=15, isDebug=True)
        result = compute_ngrc(df, isDebug=0, isInfo=0, warmup=0, train=70, k=9, test=10, ridge_param=0, isTrg=1,
                              isTrading=0)
        print('Result:', result)

