import numpy as np
import matplotlib.pyplot as plt


def xrange(a, b, h):
    result = np.zeros(int((b - a) // h) + 2)
    x = a
    for i in range(int((b - a) // h) + 2):
        result[i] = x
        x += h
    return result


def func(x):
    return np.cosh(x) + x * np.sin(x)


def system_ode(x, y):
    result = np.zeros((len(y), 1))
    result[0] = y[1]
    result[1] = func(x) - np.cosh(x) * y[1] - np.sinh(x) * y[0]
    return result


def solution(x):
    return np.exp(-np.sinh(x)) + x


def solution_array(a, b, h):
    x = a
    result = np.zeros(int((b - a) // h) + 2)
    for i in range(int((b - a) // h) + 2):
        result[i] = solution(x)
        x += h
    return result


def euler(system, a, b, h, y0):
    result = np.zeros((len(y0), int((b - a) // h) + 2))
    result[:, 0] = y0[:, 0]
    x = a
    for i in range(1, int((b - a) // h) + 2):
        result[:, i] = result[:, i - 1] + h * system(x, result[:, i - 1])[:, 0]
        x += h
    return result


def pred_corr(system, a, b, h, y0):
    result = np.zeros((len(y0), int((b - a) // h) + 2))
    result[:, 0] = y0[:, 0]
    x = a
    for i in range(1, int((b - a) // h) + 2):
        prev = system(x, result[:, i - 1])
        predictor = result[:, i - 1] + h * prev[:, 0]
        result[:, i] = result[:, i - 1] + (h/2) * (prev[:, 0] + system(x, predictor)[:, 0])
        x += h
    return result


def runge_kutta4(system, a, b, h, y0):
    result = np.zeros((len(y0), int((b - a) // h) + 2))
    result[:, 0] = y0[:, 0]
    x = a
    for i in range(1, int((b - a) // h) + 2):
        k1 = system(x, result[:, i - 1])[:, 0]
        k2 = system(x + h/2, result[:, i - 1] + h*k1/2)[:, 0]
        k3 = system(x + h/2, result[:, i - 1] + h*k2/2)[:, 0]
        k4 = system(x + h, result[:, i - 1] + h*k3)[:, 0]
        result[:, i] = result[:, i - 1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x += h
    return result


def adams4(system, a, b, h, y0):
    result = np.zeros((len(y0), int((b - a) // h) + 2))
    result[:, 0:4] = runge_kutta4(system, a, a + 3 * h, h, y0)[:, 0:4]
    x = a + 3*h
    for i in range(4, int((b - a) // h) + 2):
        prev1 = system(x, result[:, i - 1])[:, 0]
        prev2 = system(x - h, result[:, i - 2])[:, 0]
        prev3 = system(x - 2*h, result[:, i - 3])[:, 0]
        prev4 = system(x - 3*h, result[:, i - 4])[:, 0]
        result[:, i] = result[:, i - 1] + h * ((55/24) * prev1 - (59/24) * prev2 + (37/24) * prev3 - (3/8) * prev4)
        x += h
    return result


def main1():
    x0 = 0
    xend = 1
    h = 0.005
    system = system_ode
    y0 = np.array([[1], [0]])
    sol = solution_array

    number_of_methods = 4

    plt.subplot(2, number_of_methods, 1)
    plt.title("Метод Эйлера")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange(x0, xend, h), euler(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange(x0, xend, h), sol(x0, xend, h), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 2)
    plt.title("Схема предиктор-корректор")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange(x0, xend, h), pred_corr(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange(x0, xend, h), sol(x0, xend, h), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 3)
    plt.title("Метод Рунге-Кутта 4 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange(x0, xend, h), runge_kutta4(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange(x0, xend, h), sol(x0, xend, h), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, 4)
    plt.title("Метод Адамса 4 порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange(x0, xend, h), adams4(system, x0, xend, h, y0)[0, :], color='k', label='Численное значение')
    plt.plot(xrange(x0, xend, h), sol(x0, xend, h), ls='--', color='k', label='Аналитическое значение')

    plt.subplot(2, number_of_methods, number_of_methods + 1)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange(x0, xend, h), abs(euler(system, x0, xend, h, y0)[0, :] - sol(x0, xend, h)), color='k', label='Абсолютная погрешность')

    plt.subplot(2, number_of_methods, number_of_methods + 2)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange(x0, xend, h), abs(pred_corr(system, x0, xend, h, y0)[0, :] - sol(x0, xend, h)), color='k', label='Абсолютная погрешность')

    plt.subplot(2, number_of_methods, number_of_methods + 3)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange(x0, xend, h), abs(runge_kutta4(system, x0, xend, h, y0)[0, :] - sol(x0, xend, h)), color='k', label='Абсолютная погрешность')

    plt.subplot(2, number_of_methods, number_of_methods + 4)
    plt.xlabel("x")
    plt.ylabel("|Δu|")
    plt.grid()
    plt.plot(xrange(x0, xend, h), abs(adams4(system, x0, xend, h, y0)[0, :] - sol(x0, xend, h)), color='k', label='Абсолютная погрешность')

    plt.show()


def main2():
    x0 = 0
    xend = 1
    system = system_ode
    y0 = np.array([[1], [0]])

    hmin = 0.001
    hmax = 0.01
    hstep = 0.0001
    h = hmin

    error = dict()
    error[euler] = np.zeros(int((hmax - hmin) // hstep) + 2)
    error[pred_corr] = np.zeros(int((hmax - hmin) // hstep) + 2)
    error[runge_kutta4] = np.zeros(int((hmax - hmin) // hstep) + 2)
    error[adams4] = np.zeros(int((hmax - hmin) // hstep) + 2)

    for i in range(int((hmax - hmin) // hstep) + 2):
        sol = solution_array(x0, xend, h)
        for key in error:
            error[key][i] = np.max(np.abs(key(system, x0, xend, h, y0)[0, :] - sol))
        h += hstep

    #hrange = np.log(hrange)
    #for key in error:
        #error[key] = np.log(error[key])

    for key, i in zip(error, range(1, len(error) + 1)):
        plt.subplot(1, len(error), i)
        plt.title(key.__name__)
        plt.xlabel("log(h)")
        plt.ylabel("log(max(|Δu|))")
        plt.grid()
        plt.plot(xrange(hmin, hmax, hstep), error[key], color='k', label='Абсолютная погрешность')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main2()
