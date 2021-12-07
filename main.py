import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.cosh(x) + x * np.sinh(x)


def system_ode(x, y):
    result = np.zeros((len(y), 1))
    result[0] = y[1]
    result[1] = func(x) - np.cosh(x) * y[1] - np.sinh(x) * y[0]
    return result


def solution(x):
    return np.exp(-np.sinh(x)) + x


def euler(system, a, b, h, y0):
    xrange = np.arange(a, b, h)
    result = np.zeros((len(y0), len(xrange)))
    result[:, 0] = y0[:, 0]
    for i in range(1, len(xrange)):
        x = xrange[i - 1]
        result[:, i] = result[:, i - 1] + h * system(x, result[:, i - 1])[:, 0]
    return result


def pred_corr(system, a, b, h, y0):
    xrange = np.arange(a, b, h)
    result = np.zeros((len(y0), len(xrange)))
    result[:, 0] = y0[:, 0]
    for i in range(1, len(xrange)):
        prev = system(xrange[i - 1], result[:, i - 1])
        predictor = result[:, i - 1] + h * prev[:, 0]
        result[:, i] = result[:, i - 1] + (h/2) * (prev[:, 0] + system(xrange[i], predictor)[:, 0])
    return result


def runge_kutta4(system, a, b, h, y0):
    xrange = np.arange(a, b, h)
    result = np.zeros((len(y0), len(xrange)))
    result[:, 0] = y0[:, 0]
    for i in range(1, len(xrange)):
        x = xrange[i - 1]
        k1 = system(x, result[:, i - 1])[:, 0]
        k2 = system(x + h/2, result[:, i - 1] + h*k1/2)[:, 0]
        k3 = system(x + h/2, result[:, i - 1] + h*k2/2)[:, 0]
        k4 = system(x + h, result[:, i - 1] + h*k3)[:, 0]
        result[:, i] = result[:, i - 1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return result


def adams3(system, a, b, h, y0):
    xrange = np.arange(a, b, h)
    result = np.zeros((len(y0), len(xrange)))
    result[:, 0:3] = runge_kutta4(system, a, a + 3 * h, h, y0)[:, 0:3]
    for i in range(3, len(xrange)):
        prev1 = system(xrange[i - 1], result[:, i - 1])[:, 0]
        prev2 = system(xrange[i - 2], result[:, i - 2])[:, 0]
        prev3 = system(xrange[i - 3], result[:, i - 3])[:, 0]
        result[:, i] = result[:, i - 1] + h * ((23/12) * prev1 - (16/12) * prev2 + (5/12) * prev3)
    return result


def adams4(system, a, b, h, y0):
    xrange = np.arange(a, b, h)
    result = np.zeros((len(y0), len(xrange)))
    result[:, 0:4] = runge_kutta4(system, a, a + 4 * h, h, y0)[:, 0:4]
    for i in range(4, len(xrange)):
        prev1 = system(xrange[i - 1], result[:, i - 1])[:, 0]
        prev2 = system(xrange[i - 2], result[:, i - 2])[:, 0]
        prev3 = system(xrange[i - 3], result[:, i - 3])[:, 0]
        prev4 = system(xrange[i - 4], result[:, i - 4])[:, 0]
        result[:, i] = result[:, i - 1] + h * ((55/24) * prev1 - (59/24) * prev2 + (37/24) * prev3 - (3/8) * prev4)
    return result


def runge_kutta4_correction(system, a, b, h, y0):
    p = 4
    result1 = runge_kutta4(system, a, b, h, y0)
    result2 = runge_kutta4(system, a, b, h/2, y0)[:, ::2]
    result = result2 + (result2 - result1) / (2**p - 1)
    return result


def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def main1():
    x0 = 0
    xend = 5
    h = 0.01
    xrange = np.arange(x0, xend, h)
    system = system_ode
    y0 = np.array([[1], [0]])
    sol = solution

    number_of_methods = 6

    plt.subplot(2, number_of_methods, 1)
    plt.title("Метод Эйлера")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, euler(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 2)
    plt.title("Схема предиктор-корректор")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, pred_corr(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 3)
    plt.title("Метод Рунге-Кутта\n4 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, runge_kutta4(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 4)
    plt.title("Метод Адамса\n3 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, adams3(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 5)
    plt.title("Метод Адамса\n4 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, adams4(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 6)
    plt.title("Поправка Рунге для\nметода 4 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, runge_kutta4_correction(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, number_of_methods + 1)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(euler(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.subplot(2, number_of_methods, number_of_methods + 2)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(pred_corr(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.subplot(2, number_of_methods, number_of_methods + 3)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(runge_kutta4(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.subplot(2, number_of_methods, number_of_methods + 4)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(adams3(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.subplot(2, number_of_methods, number_of_methods + 5)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(adams4(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.subplot(2, number_of_methods, number_of_methods + 6)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange, abs(runge_kutta4_correction(system, x0, xend, h, y0)[0, :] - sol(xrange)), color='k')

    plt.show()


def main2():
    x0 = 0
    xend = 1
    system = system_ode
    y0 = np.array([[1], [0]])

    hmin = 0.01
    hmax = 0.1
    hstep = 0.001
    hrange = np.arange(hmin, hmax, hstep)

    error = dict()
    error[euler] = np.zeros(len(hrange))
    error[pred_corr] = np.zeros(len(hrange))
    error[runge_kutta4] = np.zeros(len(hrange))
    error[adams3] = np.zeros(len(hrange))
    error[adams4] = np.zeros(len(hrange))
    error[runge_kutta4_correction] = np.zeros(len(hrange))

    for i, h in zip(range(len(hrange)), hrange):
        sol = solution(np.arange(x0, xend, h))
        for key in error:
            error[key][i] = np.max(np.abs(key(system, x0, xend, h, y0)[0, :] - sol))

    hrange = np.log10(hrange)
    for key in error:
        error[key] = np.log10(error[key])

    plt.suptitle('Зависимость логарифма абсолютной погрешности от логарифма шага интегрирования')
    for key, i in zip(error, range(1, len(error) + 1)):
        plt.subplot(1, len(error), i)
        plt.title(key.__name__)
        plt.xlabel("log(h)")
        plt.ylabel("log(max(|Δu|))")
        plt.grid()
        plt.plot(hrange, error[key], color='k')

    for key in error:
        coeffs = least_squares(hrange, error[key])
        print(key.__name__, ": ", coeffs[0], " + ", coeffs[1], "x", sep="")
    plt.show()


if __name__ == '__main__':
    main2()
