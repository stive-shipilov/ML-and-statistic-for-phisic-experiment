import numpy as np
from scipy import stats
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.optimize import minimize


def shapiro_test(data):
    # Тест Шапиро Уилка:
    # Применение теста Шапиро-Уилка
    data = data.tolist()
    stat, p_value = stats.shapiro(data)

    print("Статистика W:", stat)
    print("p-значение:", p_value)

    # Интерпретация
    alpha = 0.05
    if p_value > alpha:        
        mean_estimate, std_estimate = stats.norm.fit(data)
        print("Данные нормально распределены (не отвергаем H0)")
        print(f"Среднее значение - {mean_estimate:.3f} ")
        print(f"Дисперсия - {std_estimate:.3f} ")
    else:
        print("Данные не нормально распределены (отвергаем H0)")
        mean_estimate = np.mean(data)
        std_estimate = np.std(data)
        print(f"Среднее значение - {mean_estimate:.3f} ")
        print(f"Дисперсия - {std_estimate:.3f} ")

def White_Test(data_x, data_y):
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    # Основная регрессия
    x_const = sm.add_constant(data_x)  # Добавление константы
    model = sm.OLS(data_y, x_const).fit()
    residuals = model.resid

    # Квадрат остатков
    residuals_squared = residuals**2

    # Расширение переменных для вспомогательной регрессии
    if data_x.ndim == 1:  # Преобразование одномерного массива в двумерный
        data_x = data_x[:, np.newaxis]
    
    X_aux = sm.add_constant(np.column_stack([
        data_x,
        data_x[:, 0]**2,
        data_x[:, 1]**2 if data_x.shape[1] > 1 else np.zeros(len(data_x)),  # Проверка размерности
        data_x[:, 0]*data_x[:, 1] if data_x.shape[1] > 1 else np.zeros(len(data_x))
    ]))

    # Вспомогательная регрессия
    aux_model = sm.OLS(residuals_squared, X_aux).fit()

    # Тестовая статистика
    n = len(data_y)
    R2 = aux_model.rsquared
    white_stat = n * R2

    # Критическое значение
    df = X_aux.shape[1] - 1  # Степени свободы = число предикторов вспомогательной модели
    critical_value = chi2.ppf(0.95, df)

    print(f"White Test Statistic: {white_stat}")
    print(f"Critical Value (95%): {critical_value}")

    if white_stat > critical_value:
        print("Отвергаем H₀: гетероскедастичность обнаружена.")
    else:
        print("Не отвергаем H₀: гетероскедастичность отсутствует.")
    
    return white_stat, critical_value

def linear_determination(data_x, data_y, slope, intercept):
    y_theory = data_x * slope + intercept
    R_squared = 1 - ((data_y - y_theory)**2).sum()/((data_y - np.mean(data_y))**2).sum()
    print(f"Коэфициент детерминации - {R_squared:.3f}")

    return R_squared

def r_squared(func, args, data_x, y_exp):
    y_model = func(data_x, args)
    print(y_model)
    y_mean = np.mean(y_exp)
    ss_res = np.sum((y_exp - y_model) ** 2)
    
    ss_tot = np.sum((y_exp - y_mean) ** 2)
    
    r2 = 1 - (ss_res / ss_tot)
    return r2

