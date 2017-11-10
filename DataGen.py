from sutils import *

'''
The problem to train a network to add 2 two-digit numbers
each addendum is from 0 to 100
the result is a categorical value from 0 to 200
Condition: in the training input set the number 50 is missing
but present in the test set
Check how it works, if remove more than one number
'''

limit_gpu_mem()
import numpy as np

# out of 2 input numbers (0-100) we get an array of 6 digits - array z
# first 3 digits are the first number a1, the second three - the second addendum a2
# input var n is concatenation of a1 and a2, 3 digits each, with leading zeros
def num2dig(n):
    z = np.zeros(6)
    for i in range(0,6):
        d = n % 10
        z[5-i] = d
        n = n // 10
    return z # array out of 6

# generate the dataset of length lenArr and excluding "exclude" numbers
def generator(lenArr, exclude):

    # get all allowed numbers
    a = np.array(range(101))
    c = np.delete(a, exclude)

    x0 = np.random.choice(c, lenArr, replace=True)
    x1 = np.random.choice(c, lenArr, replace=True)

    x6 = np.zeros((lenArr,6)) # the 6 digit representation
    y = x0 + x1  # summ
    for i in range(0, lenArr):
        x6[i] = num2dig(x0[i]*1000 + x1[i])
    return x6, y

len = 500000
len1 = 100000

exclude = [50,51,52]
# exclude = [50]
# exclude = [ i for i in range(0,101,2)]
exclude1 = []

x_train, y_train = generator(len,exclude)
x_test, y_test = generator(len1,exclude1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Save section
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


