import numpy as np
import math
from scipy.optimize import minimize



def mnk(data_x, data_y, use_systematic = False, systematic_eror = None):
    slope = (np.mean(data_x*data_y) - np.mean(data_x*np.mean(data_y)))/(np.mean(data_x**2) - (np.mean(data_x))**2)
    slope_sig = np.sqrt((1/(data_x.shape[0] - 1))*(np.sum((data_y - np.mean(data_y))**2)/np.sum((data_x - np.mean(data_x))**2) - slope**2))
    if use_systematic and systematic_eror is not None:
        av_sys_error = np.mean(systematic_eror)
        slope_sig = math.sqrt(av_sys_error**2 + slope_sig**2)
    return slope, slope_sig


def chi2_regression_1d(data_x, data_y, y_err):
    def chi2_d1(coeffs):
        a, b = coeffs
        model = a * data_x + b
        return np.sum(((data_y - model) / y_err) ** 2)
    
    initial_coeffs = [1, 0]

    result = minimize(chi2_d1, initial_coeffs)

    slope, intercept = result.x
    chi2_value = result.fun

    # The Gesse Matric
    hessian_inv = result.hess_inv

    slope_err = math.sqrt(hessian_inv[0, 0])
    intercept_err = math.sqrt(hessian_inv[1, 1])

    return slope, intercept, slope_err, intercept_err, chi2_value

def chi2_regression_2d(data_x, data_y, x_err, y_err):
    def chi2_d2(coeffs):
        a, b = coeffs
        model = a * data_x + b
        sigma_total = np.sqrt(y_err**2 + (a * x_err)**2)
        return np.sum(((data_y - model) / sigma_total) ** 2)
    
    initial_coeffs = [1, 0]

    result = minimize(chi2_d2, initial_coeffs)

    slope, intercept = result.x
    chi2_value = result.fun

    hessian_inv = result.hess_inv
    intercept_err = math.sqrt(hessian_inv[1, 1])

    def weights(a, x_err, y_err):
        return 1 / (y_err**2 + (a * x_err)**2)

    # Среднее значение x
    def weighted_mean(x, w):
        return np.sum(w * x) / np.sum(w)

    # Аналитическое вычисление ошибки наклона
    def slope_error(x, y, x_err, y_err, a):
        w = weights(a, x_err, y_err)
        x_mean = weighted_mean(x, w)
        sigma_a2 = 1 / np.sum(w * (x - x_mean) ** 2)
        return np.sqrt(sigma_a2)
    
    def intercept_error(x, x_err, y_err, a):
        w = weights(a, x_err, y_err)
        sigma_a2 = (1 / np.sum(w)*np.sum(x**2))
        return np.sqrt(sigma_a2)

    # Вычисление ошибки наклона
    slope_err = slope_error(data_x, data_y, x_err, y_err, slope)
    intercept_err = intercept_error(data_x, x_err, y_err, slope)
    return slope, intercept, slope_err, intercept_err, chi2_value

def monte_carlo_linear_model(data_x, data_y, x_err, y_err, count_iter = 100):
    a_values = [1]
    b_values = [0]
    
    residuals = data_y - (a_values[-1] * data_x + b_values[-1])
    rmse = np.sqrt(np.sum(residuals**2)/data_x.shape[0])
    # Цикл Монте-Карло
    for i in range(count_iter):
        # Генерация данных с учетом погрешностей
        x_sample = data_x + np.random.normal(0, x_err)
        y_sample = data_y + np.random.normal(0, rmse)
        
        try:
            result = chi2_regression_1d(x_sample, y_sample, y_err)
            a_values.append(result[0])
            b_values.append(result[1])
        except RuntimeError:
            # Если подгонка не удалась, пропускаем итерацию
            continue

    # Оценка параметров и их неопределенностей
    slope = np.mean(a_values)
    intercept = np.mean(b_values)
    slope_err = np.std(a_values)
    intercept_err = np.std(b_values)
    return slope, intercept, slope_err, intercept_err


def bootstrap_linear(data_x, data_y, x_err, y_err, count_iter = 100):
    
    # Бутстрэп для оценки погрешности коэффициентов
    N = len(data_x)
    slope_bootstrap = []
    intercept_bootstrap = []

    for _ in range(count_iter):
        # Генерация выборки с возвращением
        indices = np.random.choice(N, size=N, replace=True)
        x_bootstrap = data_x[indices] + np.random.normal(0, x_err[indices])
        y_bootstrap = data_y[indices] + np.random.normal(0, y_err[indices])
        
        # Подгонка линейной регрессии для выборки с возвращением
        A = np.vstack([x_bootstrap, np.ones(N)]).T
        m, c = np.linalg.lstsq(A, y_bootstrap, rcond=None)[0]
        
        slope_bootstrap.append(m)
        intercept_bootstrap.append(c)

    # Оценка параметров и погрешностей
    slope_mean = np.mean(slope_bootstrap)
    intercept_mean = np.mean(intercept_bootstrap)

    slope_std = np.std(slope_bootstrap)
    intercept_std = np.std(intercept_bootstrap)

    return slope_mean, intercept_mean, slope_std, intercept_std

def xi_square_approximation(func, initial_coeffs, data_x, data_y, x_err, y_err):
    def chi2_d1(coeffs, x_experiment, y_experiment, y_errors):
        model = func(x_experiment, coeffs)
        return np.sum(((y_experiment - model) / y_errors) ** 2)
    
    init = initial_coeffs
    
    
    N = len(data_x)
    B = 10*N  # Количество повторений бутстрэпа
    bootstrap_result = []

    for _ in range(B):
        # Генерация выборки с возвращением
        indices = np.random.choice(N, size=N, replace=True)
        
        x_bootstrap = data_x[indices] + np.random.normal(0, x_err[indices])
        y_bootstrap = data_y[indices] + np.random.normal(0, y_err[indices])
        y_err_bootstrap = y_err[indices]

        minimize_result = minimize(chi2_d1, init, args=(x_bootstrap, y_bootstrap, y_err_bootstrap))
                
        bootstrap_result.append(minimize_result.x)

    result_val = []
    result_err = []
    for i in range(0, len(bootstrap_result[0])):
        datas = []
        for j in range(0, len(bootstrap_result)):
            datas.append(bootstrap_result[j][i])
        result_val.append(np.mean(datas))
        result_err.append(np.std(datas))

    return result_val, result_err