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


def lock_and_load(picklename, seconds, lookback=pr.lookback_t, isDebug=False):

    # Build and process a dataframe containing t minutes of quote history data.
    # We assume pickle files of past t minutes already existed. This should be handled by main trading loop.
    try:
        # Get hour and min from picklename.
        hour = int(picklename[-4:][0:2])
        minute = int(picklename[-4:][2:4])
        hours_list, minutes_list = ut.hour_min_to_list_t(hour, minute, seconds, t=lookback)

        if pr.test_cross_val_trading and pr.test_cross_val_past:
            hours_list, minutes_list = ut.hour_min_to_list_t(int(pr.test_hour), int(pr.test_minute), int(pr.test_second), t=lookback)
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
        time = df.iloc[:-1:2] ; closing_price = df.iloc[1::2]

        """
        Sample:
        00:00:00.499
        5269.92
        00:00:01.004
        5269.99
        00:00:01.561
        5269.76
        00:00:02.065
        5269.17
        ...
        """
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

        # Rearrange cols for tidiness, https://is.gd/QNzlbu
        df = df[['time', 'time_diff', 'quote']]

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

        # Prepare outputs for compute. Pandas ops consolidated here so we can use Numba for speed.

        # Dataframe shape
        rows_in_df, cols_in_df = df.shape

        # total variance of data
        total_var = np.float64(np.var(df['quote']))

        # time step
        dt = np.float64(np.round(df['time_diff'].dt.total_seconds().mean(), 2))

        # For initializing into x in compute
        consolidated_array = np.array([df['quote'].to_numpy()], dtype=np.float64)

        return df, np.float64(rows_in_df), np.float64(cols_in_df), total_var, dt, consolidated_array

    except Exception:
        print('File loading threw an exception: ... ')
        print(traceback.format_exc())
        return


def compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array,
                 warmup, train, k, test, ridge_param,
                 isDebug=False, isInfo=False, isTrg=False, isTrading=False):

    # units of time to warm up NVAR. need to have warmup_pts >= 1, in seconds
    warmuptime = np.float64(warmup)
    # units of time to train for, in seconds
    traintime = np.float64(train)
    # units of time to test for, in seconds
    testtime = np.float64(test)
    # ridge parameter for regression
    ridge_param = np.float64(ridge_param)

    # Convert to numba types
    rows_in_df = np.int64(rows_in_df) ; cols_in_df = np.int64(cols_in_df)
    total_var = np.float64(total_var) ; dt = np.float64(dt)

    # discrete-time versions of the times defined above
    # Note difference between time and number of time points
    # traintime_pts - Number of 'dt' to trg on
    # testtime_pts - Number of 'dt' to predict ahead
    # warmup_pts - Number of 'dt' to have before train and test
    # warmtrain_pts - Number of 'dt' sum of warmup_pts and traintime_pts
    # maxtime_pts - # Number of dt sum of warm and train+tes if trg, less test if trading
    traintime_pts = np.int64(np.round(traintime / dt))
    testtime_pts = np.int64(np.round(testtime / dt))
    warmup_pts = np.round(warmuptime / dt)
    if isTrg:
        if traintime_pts < np.int64(0) or testtime_pts < np.int64(0):
            print('Invalid train & test time points :', traintime_pts, testtime_pts, dt)
            return
        if warmuptime == np.float64(-1) or warmuptime < np.float64(-1):
            warmup_pts = rows_in_df - (traintime_pts+testtime_pts)
        if warmuptime < np.float64(-1):
            print('Invalid warmuptime seconds while trg', warmuptime)
        warmup_pts = np.int64(warmup_pts)
        warmtrain_pts = warmup_pts + traintime_pts
        maxtime_pts = np.int64(warmtrain_pts + testtime_pts)

    if isTrading:
        # When trading, we do not need test data.
        if traintime_pts < np.int64(0) or testtime_pts < np.int64(0):
            print('Invalid train & test time points :', traintime_pts, testtime_pts, dt)
            return
        if warmuptime == np.float64(-1) or warmuptime < np.float64(-1):
            warmup_pts = rows_in_df - (traintime_pts)
        if warmuptime < np.float64(-1):
            print('Invalid warmuptime seconds while trg', warmuptime)
        warmup_pts = np.int64(warmup_pts)
        warmtrain_pts = warmup_pts + traintime_pts
        maxtime_pts = np.int64(warmtrain_pts)

    if maxtime_pts > rows_in_df: print('Not enough data for desired maxtime_pts vs rows', maxtime_pts, rows_in_df) ; return

    # input dimension
    d = np.int64(1)
    # number of time delay taps
    k = np.int64(k)
    # size of linear part of feature vector
    dlin = np.int64(k * d)
    # size of nonlinear part of feature vector
    dnonlin = np.int64(dlin * (dlin + 1) / 2)
    # total size of feature vector: constant + linear + nonlinear
    dtot = np.int64(1 + dlin + dnonlin)

    ##
    ## NVAR
    ##

    # create an array to hold the linear part of the feature vector, https://is.gd/OaiCHN
    x = np.zeros((dlin, maxtime_pts), dtype=np.float64)

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
        for delay in np.arange(k):
            for j in np.arange(delay, maxtime_pts):
                x[d * delay: d * (delay + 1), j] = consolidated_array[:, j-delay]
    except Exception:
        print('Something went wrong when computing : x ')
        return


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
        out_train = np.ones((dtot, traintime_pts), dtype=np.float64)

        """
        Copy over the linear part (shift over by one to account for constant)
        1st row is all ones , for bias term
        Next dlin # of rows has just training points (i.e. less warmup points), select out of 'x'
        """
        out_train[1:dlin + 1, :] = x[:, warmup_pts - 1:warmtrain_pts - 1]

        cnt = 0
        for row in np.arange(dlin):
            for column in np.arange(row, dlin):
                # shift by one for constant
                out_train[dlin + 1 + cnt] = x[row, warmup_pts - 1:warmtrain_pts - 1] \
                                            * x[column, warmup_pts - 1:warmtrain_pts - 1]
                cnt += 1
    except Exception:
        print('Something went wrong when computing : out_train ')
        print(traceback.format_exc())
        return


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
        return

    # apply W_out to the training feature vector to get the training output
    try:
        x_predict = x[0:d, warmup_pts - 1:warmtrain_pts - 1] + W_out @ out_train[:, 0:traintime_pts]
    except Exception:
        print('Something went wrong when computing : x_predict ')
        return

    # calculate NRMSE between true quote and training output
    trg_nrmse = np.float64(np.sqrt(np.mean((x[0:d, warmup_pts:warmtrain_pts] - x_predict[:, :]) ** 2) / total_var))

    # create a place to store feature vectors for prediction
    out_test = np.zeros(dtot, dtype=np.float64)  # full feature vector
    x_test = np.zeros((dlin, testtime_pts+1), dtype=np.float64)  # linear part

    try:
        # copy over initial linear feature vector
        x_test[:, 0] = x[:, warmtrain_pts - 1]

        # do prediction
        for j in np.arange(testtime_pts):
            # copy linear part into whole feature vector
            out_test[1:dlin + 1] = x_test[:, j]  # shift by one for constant
            # fill in the non-linear part
            cnt = 0
            for row in np.arange(dlin):
                for column in np.arange(row, dlin):
                    # shift by one for constant
                    out_test[dlin + 1 + cnt] = x_test[row, j] * x_test[column, j]
                    cnt += 1
            # fill in the delay taps of the next state
            x_test[d:dlin, j + 1] = x_test[0:(dlin - d), j]
            # do a prediction
            x_test[0:d, j + 1] = x_test[0:d, j] + W_out @ out_test[:]
    except Exception:
        print('Something went wrong when computing : x_test')
        return

    try:
        if isTrg:
            # Calculate NRMSE between ground_truth and test_predictions.
            # Only makes sense if we are training/cross val
            test_nrmse = np.float64(np.sqrt(np.mean((x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts - 1] - x_test[0:d, 0:testtime_pts]) ** 2) / total_var))
        else:
            test_nrmse = np.float64(0)
    except Exception:
        return

    try:
        # Pull out relevant data points for ground and test prediction + compute relevant items between 1st pt and last pt.
        # We distinguish whether we're cross val or trading.
        # When cross val: we're interested in test predictions instead of trg predictions.
        # When trading, we're just looking at the test predictions, i.e. predicting ahead to future and output actions to take.
        if isTrg:
            ground_truth = x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts]
            test_predictions = x_test[0:d, 0:]
            # We compute between end of last quote in data with every time point ahead.
            # i - (i+1) , i - (i+2), i - (i+3) ...
            ground_truth_quote_delta = np.cumsum(np.diff(ground_truth))
            test_predictions_quote_delta = np.cumsum(np.diff(test_predictions))

        if isTrading:
            test_predictions = x_test[0:d, 0:]
            # We compute between end of last quote in data with every time point ahead.
            # i - (i+1) , i - (i+2), i - (i+3) ...
            test_predictions_quote_delta = np.cumsum(np.diff(test_predictions))

        if isDebug:
            print('warm:', warmup)
            print('train:', train)
            print('delay:', k)
            print('test:', test)
            print('ridge:', ridge_param)
            print('isTrg:', isTrg)
            print('isTrading:', isTrading,'\n')
            print('Dataframe shape: rows, cols', rows_in_df, cols_in_df)
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

        if isInfo:
            if isTrg:
                print('training nrmse: ' , trg_nrmse)
                print('test nrmse: ' , test_nrmse)
                print('Shape of ground_truth:', ground_truth.shape)
                print('Shape of test_predictions:', test_predictions.shape)
                print('ground_truth: \n', ground_truth)
                print('test_predictions: \n', test_predictions, '\n')
                print('ground_truth_quote_delta:', ground_truth_quote_delta)
                print('test_predictions_quote_delta:', test_predictions_quote_delta)
            if isTrading:
                print('training nrmse: ' , trg_nrmse)
                print('Shape of test_predictions:', test_predictions.shape)
                print('test_predictions: \n', test_predictions, '\n')
                print('test_predictions_quote_delta:', test_predictions_quote_delta)

        # Return actions to do. 0 is buy down. 1 is buy up. -1 is do nothing.
        if isTrg:
                return test_nrmse

        if isTrading:
                return test_predictions_quote_delta

    except Exception:
        return


def cross_val_ngrc(picklename, seconds, warm, train, delay, test, ridge, threshold_test_nrmse, lookback_t):

    # Load dataframe from pickle
    df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = lock_and_load(picklename=picklename, lookback=lookback_t, seconds=seconds)

    # Get current date
    date = datetime.datetime.now().strftime("%d%m%Y")
    start_time = datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")
    if pr.test_cross_val_trading and pr.test_cross_val_past:
        start_time = pr.test_hour+pr.test_minute+datetime.datetime.now().strftime("%S%f")

    # Compute NG-RC result and normalized RMSE
    result = compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array, warmup=warm, train=train, k=delay,
                          test=test, ridge_param=ridge, isTrg=True, isTrading=False)

    # When result is to take an action, check if within NRMSE threshold. If so, we save param set to pickle.
    if result < threshold_test_nrmse:
        param = (picklename, seconds, warm, train, delay, test, ridge, np.around(result, 4), threshold_test_nrmse, lookback_t)

        # Save cross val result
        with open(pr.data_store_location + date + '/cross_val_'+pr.current_system+'/'+ start_time ,'wb') as f:
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


def get_best_params(test_range, now):
    best_param = []
    # [[train, delay, test NRMSE, lookback_t, test, ridge]]
    for i in range(0, len(test_range)): best_param.append([0,0,1.,0,0,0])
    # Open every param pickle file in cross_val folder
    for file in os.listdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val_' + pr.current_system + '/'):
        # Load pickle
        with open(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val_' + pr.current_system + '/' + file,
                  'rb') as f:
            # Sample param pickle: (picklename, seconds, warm, train, delay, test, ridge,
            #        test nrmse >>> np.around(result, 4), threshold_test_nrmse, lookback_t)
            current_param = pickle.load(f)
        # Init nrmse value for clarity.
        current_nrmse = current_param[7] ; current_test = current_param[5]
        # Save set of lowest NRMSE for each item in test_range
        # [[train, delay, test NRMSE, lookback_t, test, ridge]]
        for i in range(0, len(test_range)):
            if current_nrmse < best_param[i][2] and len(test_range) >= i+1 and current_test == test_range[i]:
                best_param.pop(i)
                best_param.insert(i, [current_param[3], current_param[4], current_nrmse, current_param[9],
                                      current_test, current_param[6]])

    return best_param


def cross_val_trading(lookback_t):

    # Get date & time
    start_time = now = datetime.datetime.now()
    print('>>> Time @ Cross Val start : ', start_time.strftime("%H:%M:%S.%f"))

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/'):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/')

    # Check if cross val dir created. if not, create it.
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val_'+pr.current_system):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val_'+pr.current_system)

    # Build data up + get one for the duration that build last took
    gq.build_dataset_last_t_minutes(t=pr.lookback_t)
    picklename, get_one_hour, get_one_minute, get_one_second = gq.get_one_now()

    # Get all possible combinations of params
    if pr.test_cross_val_trading and pr.test_cross_val_specify_test_range:
        test_range = pr.test_range
        test_range_center = np.mean(test_range)
    else:
        test_range_center = pr.time_betw_execution_end_and_trade_open + pr.asset_duration + pr.time_betw_get_end_and_execution_end
        test_range = [test_range_center+pr.asset_duration]
    bag_of_params = list(itertools.product([picklename], [get_one_second], pr.warm_range, pr.train_range, pr.delay_range, test_range, pr.ridge_range, pr.threshold_test_nrmse, [lookback_t]))
    print('# of combinations:', len(bag_of_params))

    # Remove contents from last cross_val. We start anew each cross val during trading. No storing of files between cross vals
    files = glob.glob(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val_'+pr.current_system+'/*')
    for f in files:
        os.remove(f)

    # Cross val with multiprocessing for speed!
    cross_val_multiproc(params=bag_of_params)

    # Find best params. We save the one with lowest test NRMSE
    # [[train, delay, NRMSE, lookback_t, test, ridge]]
    best_param = get_best_params(test_range=test_range, now=now)

    if pr.test_cross_val_trading:
        print('Best params')
        print('[[train, delay, NRMSE, lookback_t, test, ridge]]:')
        for bp in best_param:
            print(bp)

    print('Cross Val took this amount of time:', datetime.datetime.now()-start_time)
    print('>>> Time @ Cross Val end : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    return best_param, test_range_center


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Force a manual cross val for trading
    if pr.test_cross_val_trading:
        gq.build_dataset_last_t_minutes(t=pr.lookback_t)
        cross_val_trading(lookback_t=pr.lookback_t)

    # Quick test lock_and_load.
    if pr.test_load_function:
        df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = lock_and_load(picklename=pr.data_store_location + '14022022/1100', lookback=pr.lookback_t, seconds=15, isDebug=True)
        print('Inverse FFT of quote:', np.fft.irfft(consolidated_array, n=10))

    # Quick test compute
    if pr.test_compute_function:
        df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = lock_and_load(picklename=pr.data_store_location + '14022022/1100', lookback=pr.lookback_t, seconds=15, isDebug=True)
        result = compute_ngrc(rows_in_df, cols_in_df, total_var, dt, consolidated_array, warmup=-1, train=100, k=60,
                              test=10, ridge_param=1e-8, isDebug=False, isInfo=True, isTrg=True, isTrading=False)
        print('Result:', result)

