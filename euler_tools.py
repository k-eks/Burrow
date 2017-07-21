from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import subprocess
import numpy as np

DEFAULT_EMAIL = "gregor.hofer@mat.ethz.ch"

def submit_euler_job(parameters, **kwargs):
    """Pytonic way to pass a job to the EULER cluster
    parameters ... string or array[string] the job, including parameters, which EULER should run
    kwargs ... keyed arguments passed along, see show_job_keys for possible keys and their description
    """
    # putting the command together
    command = []
    command.append("bsub")
    # parsing kwargs
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            if key == "cpus":
                command.append(["-n", value])
            elif key == "memory":
                command.append(["-R", "\"rusage[mem=%i]\"" % value])
            elif key == "hours":
                command.append(["-W", "%s:00" % value])
            elif key == "notify":
                command.append(["-N", "-u", DEFAULT_EMAIL]) # this should be an input value

    command.append(parameters)
    command = np.hstack(np.asarray(command, dtype=object).flat)
    print("Parsed job for EULER is %s" % command)
    subprocess.call(command)


def show_job_keys():
    print("cpus ... int number of CPUs to request")
    print("memory ... int amount of memory to request per cpu in multiples of 1024")
    print("hours ... int approximate time for calculation, default is four hours")
    print("notify ... unfinished write an email to default user upon job completion")