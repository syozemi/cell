import numpy as np

class Batch:
    def __init__(self, array):
        self.array = array
        self.index = 0

    def next_batch(self, num):
        length = len(self.array)
        if self.index + num > length:
            x = np.vstack((self.array[self.index:], self.array[0:num-(length-self.index)]))
            self.index = num - (length-self.index)
            return x
        else:
            x = self.array[self.index:self.index+num]
            self.index = self.index + num
            return x

    def all_data(self):
        return self.array

def shuffle_image(image_x, image_t):
    n = image_x.shape[0]
    perm = np.random.permutation(n)
    res_x = np.zeros(tuple(image_x.shape))
    res_t = np.zeros(tuple(image_t.shape))
    
    for i in range(n):
        res_x[i,...] = image_x[perm[i],...]
        res_t[i,...] = image_t[perm[i],...]

    return [res_x, res_t]
