# ML и статистика для обработки физического эксперимента
## Содержание
- [Аннотация](#аннотация)
- [Составляющие проекта](#составляющие-проекта)
- [Структура проекта](#структура-проекта)
- [Подготовка и извлечение данных](#подготовка-и-извлечение-данных)
- [Регрессии](#регрессии)
- [Статистические критерии](#статистические-критерии)
- [Визуализация](#визуализация)
---

## Аннотация
Данный проект представляет собой библиотеку программ, которые использую сложные статистические модели для обработки физического эксперимента, где упор делается не только на регрессию тех или иных данных, но и на получение корректных данных о погрешностях исследуемых величин. Данный проект предоставляет функции, которые применяют различные методы и подходы как из статистики, так и из некоторых частей машинного обучения, с целбю получения более корректныз результатов в сравнении с классическими подходами. Также в проекте представлены программы, упрощающие обработку физического эксперимента, в части проверки гипотез и визуализации данных.

## Составляющие проекта
Проект состоит из 3 основных частей
### Линейные регрессии и аппроксимации произвольных функций
- Данная программа предоставляет различные функции, для построения линейных регрессий, в зависимости от требуемой точности и быстроты получения результата
- Каждый метод возвращает помимо коэфициентов линейной функции погрешность каждого из них
- В программе представлены такие методы как: метод наименьших квадратов, одномерная линейная регрессия, двумерная взвешенная регрессия, метод Монте-Карло, метод Bootstrap.
- Исследование различиных методов для разных случаев входных данных можно подробно прочитать в [докладе](https://vk.com/id527618351)
### Статистические критерии
- Для аппроксимации сложных функций и получения наилучших параметров, представлена программа, минимизирующая функцию хи квадрат. 
- Представлены программы, проверяющие данные на нормальность распределения, или на гетероскедатичность остатков
- Присутствуют программы, вычисляющие коэфициенты согласия теоретической модели с экспериментальными данными
### Визуализация с помощью методов ML
- Программа аппроксимирующая сложные зависимости, которые невозможно описать какой-либо функцией, полиномами заданной степени, применяющая регулязацию для устранения переобученности при большом количестве данных.
## Структура проекта
- *Получение и извлечение данных*
    - *data_load.py*

- *Функции регрессий и аппроксимаций*
    - *regression.py*

- *Статистические критерии*
    - *statistic.py*

- *Визуализация*
    - *vizualization.py*
    
## Подготовка и извлечение данных 

Для того чтобы подготовить данные, нужно сохранить данные в формате csv. Формат csv:
- 1-ый столбец - значения данных по оси **X**
- 2-ый столбец - значения данных по оси **Y**
- 3-ый столбец - погрешности данных по оси **X**
- 4-ый столбец - погрешности данных по оси **Y**

*Примечание* - если по какой-то из осей отсутствуют погрешности, требуется в csv файле ввести нулевые значения

Функция *read_tables_from_csv* из модуля *data_load*, возвращает объект данных, который содержит переданные значения по обоим осям и их погрешности.

### Пример извлечения данных

```py
from data_load import read_tables_from_csv
from regression import chi2_regression_1d

filepath = 'file.csv'

datas = read_tables_from_csv(filepath)
result = chi2_regression_1d(datas.x, datas.y, datas.y_err)
```
## Регрессии

### Использование метода наимеьших квадратов (МНК)
Функция *mnk*  вычисляет оишбку определения кривой, также, в зависимости от параметра *use_systematic* может также учитывать систематическую погрешность
```py
from data_load import read_tables_from_csv
import regression as rgr

filepath = 'data_t.csv'

datas = read_tables_from_csv(filepath)

systematic = datas.y*0.01

# Без систематической
result = rgr.mnk(datas.x, datas.y)
# С учётом систематической
result = rgr.mnk(datas.x, datas.y, use_systematic = True, systematic_eror = systematic)
```
### Применение методов линейной регрессии
Применение функций линейной регрессии, методами одномерной и двумерной взвешенны регрессий, методом Монте-Карло и Bootstrap. В последних двух методах можно вручную задавать количество итераций в зависимости от желаемой точности.
```py
from data_load import read_tables_from_csv
import regression as rgr

filepath = 'data_t.csv'

datas = read_tables_from_csv(filepath)

systematic = datas.y*0.01

# Одномерная взвешенная регрессия
result = rgr.chi2_regression_1d(datas.x, datas.y, datas.y_err)
# Двумерная взвешенная регрессия
result = rgr.chi2_regression_2d(datas.x, datas.y, datas.x_err, datas.y_err)
# Метод Монте-Карло
result = rgr.monte_carlo_linear_model(datas.x, datas.y, datas.x_err, datas.y_err, 100)
# Bootstrap
result = rgr.bootstrap_linear(datas.x, datas.y, datas.x_err, datas.y_err, 100)
```

### Аппроксимация произвольной функции
Аппроксимация по данным XY и получение значений параметров апроксимации и их погрешностей осуществляется функцией *xi_square_approximation*. Количество подгоняемых параметров может быть произвольно. Они задаются в виде массива.
```py
import numpy as np
from data_load import read_tables_from_csv
import regression as rgr

filepath = 'data_t.csv'

datas = read_tables_from_csv(filepath)


# Инициализируем начальные коэфициенты
coeffs = [1000000, 1**-3]
def theoretical_vah(vlt, initial_coeefs):

    T_e, I_n = initial_coeefs
    e = 1.6 * 10**(-19)
    k = 1.38 * 10**(-23)
    current = I_n * np.tanh((e * vlt) / (2 * k * T_e))
    return current

result_val, result_err = rgr.xi_square_approximation(theoretical_vah, coeffs, datas.x, datas.y, datas.x_err, datas.y_err)

```

## Статистические критерии
Для определения нормальности распределения некоторой величины или проверки на гетероскедатичность остатков отклонения от линейной модели реализованы функции на основе тестов Шапиро-Уилка и Уайта. Также для определения согласия теории с экспермиентов реализованы функции вычисляющий коэфициент детерминации.
```py

import numpy as np
from data_load import read_tables_from_csv
import statistic as stat
import regression as rgr


filepath = 'datas.csv'

datas = read_tables_from_csv(filepath)


# Тест Уайта
stat.White_Test(datas.x, datas.y)


slope, intercept = rgr.chi2_regression_2d(datas.x, datas.y, datas.x_err, datas.y_err)

# Коэфициент детерминации линейной модели 
stat.linear_determination(datas.x, datas.y, slope, intercept)
```

Если функция произвольная

```py
import numpy as np
from data_load import read_tables_from_csv
import statistic as stat
import regression as rgr

filepath = 'datas.csv'
datas = read_tables_from_csv(filepath)

coeffs = [1000000, 1**-3]
def theoretical_vah(vlt, initial_coeefs):

    T_e, I_n = initial_coeefs
    e = 1.6 * 10**(-19)
    k = 1.38 * 10**(-23)
    current = I_n * np.tanh((e * vlt) / (2 * k * T_e))
    return current

result_val, result_err = rgr.xi_square_approximation(theoretical_vah, coeffs, datas.x, datas.y, datas.x_err, datas.y_err)

# Коэфициент детерминации для произвольной функции
stat.r_squared(theoretical_vah, result_val, datas.x, datas.y)
```

## Визуализация
Для визуализации сложных зависимостей, для которых невозможно подобрать функцию для аппроксимацию, можно построить аппроксимирующий полином с L2 регулязацией, который сгладит возможные неточности.
Функция *polyn_approx_l2()* возвращает коэфициенты полинома для построения графика и строит график, на которых отображаются экспериментальные точки и кривая полинома

```py
import numpy as np
from data_load import read_tables_from_csv
import vizualization as vzl

filepath = 'datas.csv'
datas = read_tables_from_csv(filepath)

coefficients = vzl.polyn_approx_l2(datas.x, datas.y, degree = 10, alpha = 0.1)
```

