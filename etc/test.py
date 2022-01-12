import multiprocessing as mp
import time

def foo_pool(x):
    return (x, x*x)

time_dict = {}
result_list = []
start_time = time.time()
counter = 1

def log_result(result):
    global time_dict
    global counter
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    i, result = result
    time_dict[i].append(time.time())
    start_time, end_time = time_dict[i][0], time_dict[i][1]
    # time.sleep(0.2)

    counter += 1
    result_list.append(result)


def apply_async_with_callback():
    global time_dict
    pool = mp.Pool()
    for i in range(100):
        time_dict[i] = [time.time()]
        pool.apply_async(foo_pool, args = (i, ), callback = log_result)
        if len(time_dict[i]) > 1:
            time.sleep(0.2 - ((time_dict[i][1] - time_dict[0])))

    pool.close()
    pool.join()
    new_dict = {}
    for key, item in time_dict.items():
        new_dict[key] = item[1] - item[0]

    print(new_dict)
    print(result_list)

if __name__ == '__main__':
    apply_async_with_callback()
    end_time = time.time()
    print(f'program finished in {round(end_time - start_time, 1)} seconds')

