import numpy as np


def get_edclidean_distance(vect1, vect2):
    dist = np.sqrt(np.sum(np.square(vect1 - vect2)))
    # 或者用numpy内建方法
    # dist = numpy.linalg.norm(vect1 - vect2)
    return dist


if __name__ == '__main__':
    vect1 = np.array([1, 2, 3])
    vect2 = np.array([4, 5, 6])

    print(get_edclidean_distance(vect1, vect2))

