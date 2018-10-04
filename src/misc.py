""" tool functions """

import time

def gen_time_str():
    """ tool function to generate time str like 20180927_205959 """
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())
