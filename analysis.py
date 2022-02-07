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
import logging as lg


# Display dataframe & arrays in full glory
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Turn off runtime warnings
warnings.filterwarnings('ignore')


def load(picklename=None, filename=None, isInfo=0):

    # Load via .txt and pickle.
    try:
        if filename is not None:
            df = pd.read_csv(filename, sep=" ", header=None)
        if picklename is not None:
            unpickled = pd.read_pickle(picklename)
            df = pd.DataFrame([x.split(' ') for x in unpickled.split('\n')])

        # Check if there's 2 columns
        row, col = df.shape
        if col == 2:
            df = df.iloc[:, :-1]

        # We don't want first 65 rows.
        df = df.iloc[66:, :1]
        # Remove last 2 rows.
        df = df.iloc[:-2, :1]

        # Check if first and last row is valid. If not, we drop it. We want first row to be time. Last row to be quote price.
        rows_to_check = 1
        for row in range(0, rows_to_check):
            # In event that get_quote last row is time, drop last row
            if len(df[0].iloc[-1]) != 7:
                df.drop(df.tail(1).index, inplace=True)

            # In event that get_quote first row is not time, drop first row
            if ':' not in df[0].iloc[0]:
                df.drop(df.head(1).index, inplace=True)

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

        # Drop of 1st row as it contains NaN
        # df = df.iloc[1:, :]

        # Reset index numbers
        df.reset_index(inplace=True)

        # Delete unwanted "index" col
        df.drop(['index'], inplace=True, axis=1)

        if isInfo:
            print('Data : \n', df.head(n=200), '\n')
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
# df = load(picklename=pr.data_store_location+'01022022/1530', isInfo=1)
# df = load(picklename=pr.data_store_location+'07022022/1204', isInfo=0)

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

def compute_ngrc(df, isDebug, isInfo, warmup, train, k, test, ridge_param, which_start, isTrg=0, isTrading=0):

    # total variance of data
    total_var = np.var(df['quote'])

    # time step
    dt = np.round(df['time_diff'].mean().total_seconds(), 2)
    # units of time to warm up NVAR. need to have warmup_pts >= 1, in seconds
    if which_start == 1:
        warmup = (pr.target_start1_time_second - warmup) - (train+test) # Computing on warmup points on minute prior
    if which_start == 2:
        warmup = (pr.target_start2_time_second - warmup) - (train+test) # Computing on warmup points on current minute.
    # units of time to train for, in seconds
    traintime = train
    # units of time to test for, in seconds
    testtime = test
    # ridge parameter for regression
    ridge_param = ridge_param

    # discrete-time versions of the times defined above
    warmup_pts = 0 ; traintime_pts = 0 ; warmtrain_pts = 0 ; maxtime = 0 ; testtime_pts = 0 ; maxtime_pts = 0
    if isTrg:
        warmup_pts = round(warmup / dt)
        traintime_pts = round(traintime / dt)
        warmtrain_pts = warmup_pts + traintime_pts
        maxtime = warmup + traintime + testtime
        testtime_pts = round(testtime / dt)
        maxtime_pts = round(maxtime / dt)

    if isTrading:
        # When trading, we do not need test data.
        total_rows = df.shape[0] # Number of 'dt' in dataframe
        traintime_pts = round(traintime / dt) # Number of 'dt' to trg on
        warmup_pts = total_rows - traintime_pts # Number of 'dt' to shift so we trg on latest data we have, We ignore param 'warm'
        warmtrain_pts = warmup_pts + traintime_pts # Number of 'dt' sum of warm and train
        testtime_pts = round(testtime / dt) # Number of 'dt' to predict ahead
        maxtime_pts = warmtrain_pts # Number of dt sum of warm and train, less test because we're trading now


    # input dimension
    d = 1
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
        for delay in range(k):
            for j in range(delay, maxtime_pts):
                x[d * delay: d * (delay + 1), j] = df['quote'].iloc[j - delay]
    except Exception:
        print('Something went wrong when computing : x ')
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param,  'Which_Start', which_start)
        print('Dataframe shape', df.shape)
        print('Quote series:', df['quote'])
        # print(traceback.format_exc())
        return -1, 0

    """
    create an array to hold the full feature vector for training time
    (use ones so the constant term is already 1)
    """
    out_train = np.ones((dtot, traintime_pts))

    """
    Fill in the non-linear part
    How to edit this for RBF or higher order of polynomial?
    out_train[dlin + 1 + cnt] : refers to the row number 'dlin + 1 + cnt'
    x[row, warmup_pts - 1:warmtrain_pts - 1] * x[column, warmup_pts - 1:warmtrain_pts - 1]
        >> generating quadratric terms 
    """
    try:
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
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param,  'Which_Start', which_start)
        print('Dataframe shape', df.shape)
        # print(traceback.format_exc())
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
        print('Warm:', warmup, 'Train:', train,  'Delay', k,  'Test', test,  'Ridge', ridge_param,  'Which_Start', which_start)
        print('Dataframe shape', df.shape)
        # print(traceback.format_exc())
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
        print(traceback.format_exc())
        return -1, 0

    try:
        if isTrg:
            # Calculate NRMSE between true and prediction
            test_nrmse = np.sqrt(np.mean(
                (x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts - 1] - x_test[0:d, 0:testtime_pts]) ** 2) / total_var)
        else:
            test_nrmse = 0
    except Exception:
        print(traceback.format_exc())
        return -1, 0

    try:
        # Pull out relevant data points for true and prediction + compute change between 1st pt and last pt.
        true = x[0:d, warmtrain_pts - 1:warmtrain_pts + testtime_pts - 1]
        pred = x_test[0:d, 0:testtime_pts]
        test_delta = true[0, -1] - true[0, 0]
        pred_delta = pred[0, -1] - pred[0, 0]

        if isDebug:
            print('warm:',warmup)
            print('train:',train)
            print('delay:',k)
            print('test:',test)
            print('ridge:',ridge_param)
            print('which_start:',which_start)
            print('isTrg:',isTrg)
            print('isTrading:',isTrading)
            print('x : \n', x, '\n')
            print('out_train: \n', out_train, '\n')
            print('W_out: \n', W_out, '\n')
            print('x_predict: \n', x_predict, '\n')
            print('x_test: \n', x_test, '\n')
            print('Test Data: \n', true)
            print('Prediction : \n', pred, '\n')

        if isInfo:
            if isTrg:
                print('Avg Time Diff : ', df['time_diff'].mean().total_seconds())
                print('training nrmse: ' + str(trg_nrmse))
                print('test nrmse: ' + str(test_nrmse))
                print('Test Data Delta : ', test_delta)
                print('Prediction Delta : ', pred_delta, '\n')
            if isTrading:
                print('Avg Time Diff : ', df['time_diff'].mean().total_seconds())
                print('training nrmse: ' + str(trg_nrmse))
                print('Prediction Delta : ', pred_delta, '\n')

        if isTrg:
            pred_delta_tolr = 0
            if test_delta < 0 and pred_delta < 0:
                return 0, trg_nrmse, test_nrmse
            if test_delta > 0 and pred_delta > 0:
                return 1, trg_nrmse, test_nrmse
            else:
                return -1, trg_nrmse, test_nrmse

        if isTrading:
            if pred_delta < 0:
                return 0, pred_delta
            if pred_delta > 0:
                return 1, pred_delta
            else:
                return -1, pred_delta
    except Exception:
        print(traceback.format_exc())
        return -1, 0
# df = load(picklename=pr.data_store_location+'04022022/1445', isInfo=1)
# result = compute_ngrc(df, isDebug=1, isInfo=1, warmup=10, train=10, k=14, test=9, ridge_param=0.1,
#                       isTrading=0, isTrg=1, which_start=1)


def cross_val_ngrc(file, warm, train, delay, test, ridge, threshold_test_nrmse, which_start):

    # Load dataframe from pickle
    df = load(picklename=file)

    # Get current date
    date = datetime.datetime.now().strftime("%d%m%Y")

    # Compute NG-RC result and normalized RMSE
    result = compute_ngrc(df, isDebug=0, isInfo=0, warmup=warm, train=train, k=delay, test=test, ridge_param=ridge,
                          which_start=which_start, isTrading=0, isTrg=1)

    # When result is to take an action, check if within NRMSE threshold. If so, we save param set to pickle.
    if result[0] == 1 or result[0] == 0:
        if result[2] < threshold_test_nrmse:

            param = (file, warm, train, delay, test, ridge,
                      np.around(result[1], 4), np.around(result[2], 4), threshold_test_nrmse)

            # Save cross val result
            with open(pr.data_store_location + date + '/cross_val/'+ str(train) + '-' + str(delay) + '-' + str(test) + '-' + str(warm) + '-' + str(ridge) + '-' + str(round(result[2],2))+ '-' + file[-4:] ,'wb') as f:
                pickle.dump(param, f)

            # print('Params -', 'warm:', warm, 'trg:', train, 'k:', delay, 'test:', test, 'ridge:', ridge,
            #       'trg_nrmse:', np.around(result[1], 4), 'test nrmse:', np.around(result[2], 4))

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
    print('Time @ Cross Val start : ', start_time.strftime("%H:%M:%S.%f"))

    # Build dataset
    nowtime_build_last_t_hour , nowtime_build_last_t_min = gq.build_dataset_last_t_minutes(t=t, isTrading=1)

    # Check if folder for today exists
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/'):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/')

    # Check if cross val dir created. if not, create it.
    if not os.path.isdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val'):
        os.mkdir(pr.data_store_location + now.strftime("%d%m%Y") + '/cross_val')

    # Build file names for last t minutes of data; Make sure we use same time as build_dataset_last_t_minutes
    files = []
    hour = int(nowtime_build_last_t_hour)
    min = int(nowtime_build_last_t_min)+1

    # To use cross_val_trading to look back at a particular minute and lookback_t before it.
    if pr.force_manual_cross_val_trading:
        hour = pr.forced_hour
        min = pr.forced_min

    if t > min:
        hours = [hour]
        mins = range(0,min)
        files = filenames(dates=[now.strftime("%d%m%Y")], hours=hours, mins=mins, files=files)
        # Check is we are 00 hours.
        if hour - 1 < 0:
            hours = [23]
        else:
            hours = [hour-1]
        mins = range(60-(t-(min)),60)
        files = filenames(dates=[now.strftime("%d%m%Y")], hours=hours, mins=mins, files=files)
    if t <= min:
        hours = [hour]
        mins = range((min)-t,min)
        files = filenames(dates=[now.strftime("%d%m%Y")], hours=hours, mins=mins, files=files)

    # Get all possible combinations of params
    bag_of_params = list(itertools.product(files, pr.warm_range, pr.train_range, pr.delay_range, pr.test_range, pr.ridge_range, pr.threshold_test_nrmse, pr.which_start))
    print('# of combinations:', len(bag_of_params))

    # Remove contents from last cross_val. We start anew each cross val during trading.
    files = glob.glob(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val/*')
    for f in files:
        os.remove(f)

    # Cross val with multiprocessing for speed!
    cross_val_multiproc(params=bag_of_params)

    # Find best params. We save 2 pairs: one with most count of a single train-delay pair that was cross validated + one with lowest NRMSE
    # [[most count: train, delay, count, NRMSE],[lowest NRMSE: train, delay, count, NRMSE]]
    best_params = [[0,0,0,1.],[0,0,0,1.]]
    bag_of_key_params = list(itertools.product(pr.train_range, pr.delay_range))
    # Iterate over each unique pair of train and delay
    for i in bag_of_key_params:
        count = 0
        pattern = str(i[0])+'-'+str(i[1])+"-"
        # For each pair, calc count within cross_val folder
        for file in os.listdir(pr.data_store_location+now.strftime("%d%m%Y")+'/cross_val'):
            if re.match(pattern=pattern, string=file):
                count += 1
                # Get NRMSE & check if sensible. Rounding can cause value to be 0.7 or 0.45.
                current_nrmse = float(file[-9:-5])
                if file[-9:-5][0] == '-':
                    current_nrmse = float(file[-8:-5])
                # Save the pair with most count.
                if count > best_params[0][2]:
                    best_params[0][0] = i[0]
                    best_params[0][1] = i[1]
                    best_params[0][2] = count
                    best_params[0][3] = current_nrmse
                # Save the pair with lowest NRMSE
                if current_nrmse < best_params[1][3]:
                    best_params[1][0] = i[0]
                    best_params[1][1] = i[1]
                    best_params[1][2] = count
                    best_params[1][3] = current_nrmse

    # Pickle best params
    with open('params_current_trading', 'wb') as f:
        pickle.dump(best_params, f)

    print('Cross Val took this amount of time:', datetime.datetime.now()-start_time)
    print('Time @ Cross Val end : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    picklename = pr.data_store_location+now.strftime("%d%m%Y")+'/'+nowtime_build_last_t_hour+nowtime_build_last_t_min
    return picklename

def cross_val_manual():
    print(' Time @ Start : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Get filenames
    dates = ['04022022']  # [26012022, 27012022, 28012022]
    hours = [14]
    mins = range(45, 46)
    files = filenames(dates=dates, hours=hours, mins=mins, files=[])

    # Get all possible combi of params
    # Get all possible combinations of params
    bag_of_params = list(itertools.product(files, pr.warm_range, pr.train_range, pr.delay_range, pr.test_range, pr.ridge_range, pr.threshold_test_nrmse, pr.which_start))
    print('# of params:', len(bag_of_params))

    cross_val_multiproc(params=bag_of_params)
    print(' Time @ End : ', datetime.datetime.now().strftime("%H:%M:%S.%f"))

    return

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if pr.force_manual_cross_val_trading:
        cross_val_trading(t=1)
        # Open pickle files
        with open('params_current_trading', 'rb') as f:
            contents = pickle.load(f)
        print('Best parameters:', contents)
