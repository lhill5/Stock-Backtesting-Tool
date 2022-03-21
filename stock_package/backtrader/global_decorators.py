import time

def timer(func):
    def wrapper_func(*args, **kwargs):
        # Do something before the function.
        start_time = time.time()

        func(*args, **kwargs)

        # Do something after the function.
        end_time = time.time()
        print(f'{func.__name__} took {round(end_time - start_time, 2)} seconds')
    return wrapper_func


def timer_method(method):
    def wrapper_func(ref, ticker):
        start_time = time.time()
        data = method(ref, ticker)

        end_time = time.time()
        print(f'{method.__name__} took {round(end_time - start_time, 2)} seconds')

        return data

    return wrapper_func

# @timer
# def my_func(my_arg):
#     '''Example docstring for function'''
#
#     pass