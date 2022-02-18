import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os
import analysis as an
import params as pr

def data_collection(consolidated_array, time_range,
                    action, predicted_delta, test_nrmse, delay , test_range_center, lookback,
                    trade_outcome):

    # Calculate stats, with reference to last data and looking back as per time_range.
    # max = np.array([], dtype=np.float64)
    # min = np.array([], dtype=np.float64)
    # mean = np.array([], dtype=np.float64)
    stdev = np.array([], dtype=np.float64)
    diff = np.array([], dtype=np.float64)

    for time in time_range:
        # max = np.append(max, consolidated_array[:,-1:]-np.max(consolidated_array[:,-time:]))
        # min = np.append(min, consolidated_array[:,-1:]-np.min(consolidated_array[:,-time:]))
        # mean = np.append(mean, consolidated_array[:,-1:]-np.mean(consolidated_array[:, -time:]))
        stdev = np.append(stdev, np.std(consolidated_array[:, -time:]))
        diff = np.append(diff, np.sum(np.diff(consolidated_array[:, -time:])))

    # Consolidate
    data_point = np.array([stdev, diff], dtype=np.float64).flatten()
    data_point = np.append(data_point, [action, predicted_delta, test_nrmse, delay ,
                                        test_range_center, lookback, trade_outcome])
    data_point = np.round(data_point, 5)
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
    isSaveParams = True

    data = np.load(pr.data_store_location+'mlp_svc_input.npy', allow_pickle=True)
    cols_to_delete = None
    if cols_to_delete is not None:
        # Expects a tuple
        data = np.delete(data, cols_to_delete, axis=1)
    rows, cols = data.shape
    print('Data shape:', data.shape)
    trg_percent = 0.8
    np.random.shuffle(data)

    X_trg = data[:round(trg_percent*rows),:-2]
    Y_trg = np.int32(data[:round(trg_percent*rows),-2:-1])
    X_test = data[round((trg_percent)*rows):, :-2]
    Y_test = np.int32(data[round((trg_percent)*rows):, -2:-1])

    clf = make_pipeline(StandardScaler(), SVC(C=1, probability=False, kernel='linear', degree=2))
    # clf = make_pipeline(StandardScaler(),  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 2), random_state=1))
    # clf = tree.DecisionTreeClassifier()
    # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf.fit(X_trg,Y_trg)
    test_score = clf.score(X_test, Y_test)
    print('Test score:', test_score)

    if isSaveParams:
        # Save to pickle
        f = open('svc_model', 'wb')
        pickle.dump(clf, f)
        f.close()

    return
# training()


def predict():
    f = open('svc_model', 'rb')
    svc = pickle.load(f)
    f.close()

    data = np.load(pr.data_store_location+'mlp_svc_input.npy', allow_pickle=True)
    cols_to_delete = None
    if cols_to_delete is not None:
        # Expects a tuple
        data = np.delete(data, cols_to_delete, axis=1)
    rows, cols = data.shape
    print('Data shape:', data.shape)
    trg_percent = 0.8
    np.random.shuffle(data)

    X_test = data[round((trg_percent)*rows):, :-2]
    Y_test = np.int32(data[round((trg_percent)*rows):, -2:-1])

    pred = svc.predict(X_test)
    print(':', Y_test.T)
    print(':', pred)
    print(':',svc.classes_)

    return
# predict()


def svc_trading(consolidated_array, action, predicted_delta, test_nrmse, delay , test_range_center, lookback):

    # Calculate stats, with reference to last data and looking back as per time_range.
    # max = np.array([], dtype=np.float64)
    # min = np.array([], dtype=np.float64)
    # mean = np.array([], dtype=np.float64)
    stdev = np.array([], dtype=np.float64)
    diff = np.array([], dtype=np.float64)

    for time in np.arange(2,120,1):
        # max = np.append(max, consolidated_array[:,-1:]-np.max(consolidated_array[:,-time:]))
        # min = np.append(min, consolidated_array[:,-1:]-np.min(consolidated_array[:,-time:]))
        # mean = np.append(mean, consolidated_array[:,-1:]-np.mean(consolidated_array[:, -time:]))
        stdev = np.append(stdev, np.std(consolidated_array[:, -time:]))
        diff = np.append(diff, np.sum(np.diff(consolidated_array[:, -time:])))

    # Consolidate
    data_point = np.array([stdev, diff], dtype=np.float64).flatten()
    data_point = np.append(data_point, [action, predicted_delta, test_nrmse, delay ,
                                        test_range_center, lookback])
    data_point = np.round(data_point, 5).reshape(1,-1)

    f = open('svc_model', 'rb')
    svc = pickle.load(f)
    f.close()

    pred = svc.predict(data_point)

    return pred


if __name__ == '__main__':

    # df, rows_in_df, cols_in_df, total_var, dt, consolidated_array = an.lock_and_load(
    #     picklename=pr.data_store_location + '14022022/1100', lookback=pr.lookback_t, isDebug=False)
    # svc_trading(consolidated_array, 1, 0.75, -0.7, np.arange(3, 60, 7))
    # data_collection(consolidated_array, 0.57, 1, 0.5, -0.234444, 10, np.arange(3, 60, 7), 1)
    # data_collection(consolidated_array, 0.61, 1, 0.75, -0.112222, 10, np.arange(3, 60, 7), 1)

    all = np.load(pr.data_store_location+'mlp_svc_input.npy', allow_pickle=True)
    # print(':', all)

