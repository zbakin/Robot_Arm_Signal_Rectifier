from signal_rectifier import *
import glob

'''
    Usage:
    python jumps_test.py <directory with sample files>
'''
if __name__ == "__main__":

    resultsDir = sys.argv[1]
    resultsPattern = f"{resultsDir}/*.txt"
    glob.glob(resultsPattern)
    errors = 0
    for file in glob.glob(resultsPattern):
        SR = SignalRectifier(file)
        SR.remove_noise_sosfiltfilt()
        if SR.detect_jumps() != [0.75, 1.5, 2.25]:
            errors += 1
            print(f"TEST FAILED: Wrong jump points in file: {file}")
        else:
            print(f"File OK: {file}")

    if errors == 0:
        print("TEST PASSED")
    else:
        print(f"Total number of errors: {errors}")