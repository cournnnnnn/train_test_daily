import time
n = 100
for i in range(n):
    print('\r=={}=='.format(i+1),end="")
    time.sleep(0.5)
