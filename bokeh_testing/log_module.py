import logging

def test():
    logging.debug('This is a debug message')
    logging.info('This is an info message')
    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')

def setup_log():
    fmtstr = '%(asctime)s\n%(name)s - %(levelname)s - function: %(funcName)s - Line: %(lineno)d - %(message)s'
    logging.basicConfig(filename='log_output.log', level=logging.DEBUG, filemode='w', format=fmtstr)


# import logging
#
# datestr = "%m/%d/%Y %I:%M:%S %p "
#
# logging.basicConfig(
#     filename="log_output.log",
#     # level=logging.DEBUG,
#     filemode="w",
#     format=fmtstr,
#     datefmt=datestr,
# )
#
# logging.info("Info messages")
# # logging.warning("warning message")
# # logging.debug("debug message")
#
# # logger = logging.getLogger()
#
#
# # def f():
# #     # try:
# #     flaky_func()
# #
# #     # except Exception:
# #     #     logging.exception()
# #     #     raise
# #
# # def flaky_func():
# #     print('hello')
# #     # x = 5 / 0
# #
# # f()
