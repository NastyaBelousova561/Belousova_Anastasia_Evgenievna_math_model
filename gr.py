import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve

# Генерация случайных данных для примера
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 образцов, 2 признака
y_linear = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1  # Линейная зависимость
y_logistic = (y_linear > np.median(y_linear)).astype(int)  # Бинарная зависимая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_linear, test_size=0.3, random_state=0)

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Прогнозирование
y_pred_linear = linear_model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred_linear)
print(f'Mean Squared Error (Linear Regression): {mse}')

# Визуализация результатов линейной регрессии
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Linear Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.legend()
plt.grid()
plt.show()

# Логистическая регрессия
# Разделение данных для логистической регрессии
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X, y_logistic, test_size=0.3, random_state=0)

# Логистическая регрессия
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

# Прогнозирование вероятностей
y_pred_logistic = logistic_model.predict_proba(X_test_logistic)[:, 1]

# Оценка модели
roc_auc = roc_auc_score(y_test_logistic, y_pred_logistic)
print(f'ROC AUC (Logistic Regression): {roc_auc}')

# Визуализация ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test_logistic, y_pred_logistic)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()