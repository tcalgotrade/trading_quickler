import time
import multiprocessing
import params as pr
import analysis as an
import execution as ex
import utility as ut

if __name__ == '__main__':
    multiprocessing.freeze_support()
    while True:
        if ex.checks(day_change_chk=True) == 1:
            time.sleep((pr.lookback_t+1)*60)
            if pr.olymp_day is not None or pr.olymp_day != ():
                ut.date_changer()
        best_params, test_time_center =  an.cross_val_trading(pr.lookback_t)
        time.sleep(3)

