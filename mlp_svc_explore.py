import numpy as np
import sklearn as skl
import pickle
import os
import analysis as an
import params as pr

def data_collection(consolidated_array, time_delta, action, action_fraction, mean_delta,
                    test_time, time_range, trade_outcome):

    # Calculate stats, with reference to last data and looking back as per time_range.
    max = np.array([], dtype=np.float64)
    min = np.array([], dtype=np.float64)
    mean = np.array([], dtype=np.float64)
    stdev = np.array([], dtype=np.float64)
    diff = np.array([], dtype=np.float64)

    for time in time_range:
        max = np.append(max, consolidated_array[:,-1:]-np.max(consolidated_array[:,-time:]))
        min = np.append(min, consolidated_array[:,-1:]-np.min(consolidated_array[:,-time:]))
        mean = np.append(mean, consolidated_array[:,-1:]-np.mean(consolidated_array[:, -time:]))
        stdev = np.append(stdev, np.std(consolidated_array[:, -time:]))
        diff = np.append(diff, np.sum(np.diff(consolidated_array[:, -time:])))

    # Consolidate
    data_point = np.array([max, min, mean, stdev, diff], dtype=np.float64).flatten()
    data_point = np.append(data_point, [action, action_fraction, mean_delta, time_delta, test_time, trade_outcome])
    data_point = np.round(data_point, 4)
    data_point_with_quote = data_point.tolist()
    data_point_with_quote.append(consolidated_array)

    print('MSE: Saving this datapoint to pickle:')
    print(data_point_with_quote[:-1])

    # Add to data pickle
    if not os.path.isfile(pr.data_store_location+'mlp_svc_input.npy'):
        f = open(pr.data_store_location+'mlp_svc_input.npy', 'wb')
        pickle.dump(data_point_with_quote, f)
        f.close()
        return
    else:
        data = np.load(pr.data_store_location+'mlp_svc_input.npy', allow_pickle=True)
        np.save(pr.data_store_location+'mlp_svc_input.npy', np.vstack([data, data_point_with_quote]))
        return

def training():
    data = np.load('mlp_svc_input.npy', allow_pickle=True)

    return


if __name__ == '__main__':

    # df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = an.lock_and_load(
    #     picklename=pr.data_store_location + '14022022/1100', lookback=pr.lookback_t, seconds=15, isDebug=True)
    # data_collection(consolidated_array, 0.57, 1, 0.5, -0.234444, 10, np.arange(3, 60, 7), 1)
    # data_collection(consolidated_array, 0.61, 1, 0.75, -0.112222, 10, np.arange(3, 60, 7), 1)

    all = np.load('mlp_svc_input.npy', allow_pickle=True)
    print(':', all)