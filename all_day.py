import datetime
import time
import numpy as np
import multiprocessing
import params as pr
import get_quote as gq
import analysis as an
import execution as ex
import utility as ut

all_day_cross_val = False
all_day_build_last_t = False

if __name__ == '__main__':
    multiprocessing.freeze_support()

    if all_day_cross_val:
        while True:
            if ex.checks(day_change_chk=True) == 1:
                time.sleep((pr.lookback_t+1)*60)
                if pr.olymp_day is not None or pr.olymp_day != ():
                    ut.date_changer()
            best_params, test_time_center =  an.cross_val_trading(pr.lookback_t)
            time.sleep(3)

    if all_day_build_last_t:
        gq.build_dataset_last_t_minutes(t=2)
        while True:
            if ex.checks(day_change_chk=True) == 1:
                time.sleep((pr.lookback_t_min+1)*60)
                # Change date by refreshing.
                ut.refresh()
            time.sleep(59-datetime.datetime.now().second)
            gq.build_dataset_last_t_minutes(t=pr.lookback_t_min)

