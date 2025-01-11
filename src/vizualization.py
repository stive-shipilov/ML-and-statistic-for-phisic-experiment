import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def polyn_approx_l2(data_x, data_y, degree = 1, alpha = 0): 
    # Преобразование данных
    data_x = data_x.values.reshape(-1, 1)
    data_y = data_y.values  # Оставляем одномерным массивом

    # Создаём модель с полиномиальной регрессией и L2-регуляризацией
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    model.fit(data_x, data_y)

    # Вычисление значений
    x_dense = np.linspace(data_x.min(), data_x.max(), 100).reshape(-1, 1)  # Плотная сетка для графика
    y_fit = model.predict(x_dense)

    coefficients = model['linearregression'].coef_.tolist() 
    coefficients.append(model['linearregression'].intercept_)   

    # Построение графика
    plt.scatter(data_x, data_y, label="Data points")
    plt.plot(x_dense, y_fit, color="red", label="Polynomial fit")
    plt.legend()
    plt.show()

    return coefficients