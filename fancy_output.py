import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")

def okay(message):
    """Just a fancy way to show in green writing that everything is working fine."""
    print ("\033[92m" + message + "\033[0m")

def error(message):
    """Just a fancy way to show in red writing that an error occured."""
    print ("\033[91m" + message + "\033[0m")

def warn(message):
    """Just a fancy way to show in yellow writing that something out of the ordinary happened."""
    print ("\033[93m" + message + "\033[0m")