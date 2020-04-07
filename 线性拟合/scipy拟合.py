import numpy as np
from scipy.optimize import curve_fit


def f_fit(x, a, b, c):
    return a * np.sin(x) + b * x + c


def f_test(x):
    return 2 * np.sin(x) + 3 * x + 1


def my_curve_fit1(fit_fun, x, y, p_num):  # p_num参数的个数
    size = len(x)
    if p_num <= 0 or p_num > size:
        print('no parameter to fit')
        return
    test_list = [0] * p_num
    test_list[0] = 1
    x_arr = np.array(fit_fun(x, *test_list))
    x_arr.resize((1, size))
    for i in range(1, p_num):
        test_list[i - 1] = 0
        test_list[i] = 1
        temp_x = np.array(fit_fun(x, *test_list))
        temp_x.resize((1, size))
        x_arr = np.append(x_arr, temp_x, axis=0)
    x_mat = np.mat(x_arr)
    y_arr = np.array(y)
    y_arr.resize((1, size))
    y_mat = np.mat(y_arr)
    w = (x_mat * x_mat.T).I * x_mat * y_mat.T
    w0 = w.T
    w0.resize(w0.size)
    return w0


# 为了强行接近库函数的接口,无需参数的个数，但实现有点捞
# 很容易知道参数个数多于样本x的个数无法拟合
def my_curve_fit2(fit_fun, x, y):
    test_list = [0]
    size = len(x)
    p_num = 1  # 参数个数默认为1
    for i in range(0, size):
        try:
            fit_fun(x, *test_list)
            break
        except:
            p_num += 1
            test_list.append(0)
    if p_num > size:
        print('can not fit')
        return
    test_list[0] = 1
    x_arr = np.array(fit_fun(x, *test_list))
    x_arr.resize((1, size))
    for i in range(1, p_num):
        test_list[i - 1] = 0
        test_list[i] = 1
        temp_x = np.array(fit_fun(x, *test_list))
        temp_x.resize((1, size))
        x_arr = np.append(x_arr, temp_x, axis=0)
    x_mat = np.mat(x_arr)
    y_arr = np.array(y)
    y_arr.resize((1, size))
    y_mat = np.mat(y_arr)
    w = (x_mat * x_mat.T).I * x_mat * y_mat.T
    w0 = w.T
    w0.resize(w0.size)
    return w0


x = np.linspace(-2 * np.pi, 2 * np.pi)
y = f_test(x) + 0.3 * np.random.randn(len(x))  # 加入噪音
p_fit, prov = curve_fit(f_fit, x, y)  # 曲线拟合
my_fit1 = my_curve_fit1(f_fit, x, y, 3)
my_fit2 = my_curve_fit2(f_fit, x, y)
print('sicpy库的曲线拟合')
print('a,b,c', p_fit)
print('手写曲线拟合1')
print('a,b,c', my_fit1)
print('手写曲线拟合2')
print('a,b,c', my_fit2)