# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function

import json
import math
import os
import sys
import time
from datetime import datetime
import numpy as np

import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

print_grad = True


class printOut(object):
    def __init__(self, f=None, stdout_print=True):
        ''' 
        This class is used for controlling the printing. It will write in a 
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self, s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) + "  " + str(time.ctime())))
        return time.time()


def get_time():
    '''returns formatted current time'''
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
