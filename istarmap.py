import multiprocessing.pool as mpp

"""
https://is.gd/Bd59jl
"""

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    This iso we can pass multiple arguments into function for multiprocessing work.
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap