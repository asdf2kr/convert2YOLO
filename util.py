
import os
import sys
import pickle
import platform
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 20):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total - 1)))
    filledLength = int(round(barLength * iteration / float(total - 1)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear') # window
    sys.stdout.write('\r%s |%s| %s%s %s\n' % (prefix, bar, percent, '%', suffix))
    if iteration == total - 1:
        sys.stdout.write('\n')
    sys.stdout.flush()

def save_commonData(commonData, datasets):
    with open(os.path.join(os.getcwd(), datasets + '.pkl'), 'wb') as f:
        pickle.dump(commonData, f, pickle.HIGHEST_PROTOCOL)
    print("[Info] Save the {} file.".format(os.path.join(os.getcwd(), datasets + '.pkl')))

def load_commonData(datasets):
    with open(os.path.join(os.getcwd(), datasets + '.pkl'), 'rb') as f:
        return pickle.load(f)
    print("[Info] Load the {} file.".format(os.path.join(os.getcwd(), datasets + '.pkl')))
