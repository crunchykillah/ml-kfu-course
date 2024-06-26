import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
df = pd.read_csv('bikes_rent.csv')

# Пункт 2: Простая линейная регрессия
X = df[['weathersit']].values
y = df['cnt'].values

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Прогноз спроса на основе благоприятности погоды')
plt.xlabel('Благоприятность погоды')
plt.ylabel('Спрос')
plt.show()

# Пункт 3
new_value = np.array([[3]])
predicted_cnt = model.predict(new_value)
print(f'Предсказанное количество аренд: {predicted_cnt[0]}')

# Пункт 4
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop('cnt', axis=1))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('2D график предсказания cnt')
plt.show()

# Пункт 5
lasso = Lasso(alpha=0.1)
lasso.fit(df.drop('cnt', axis=1), y)

# Определение признака, который оказывает наибольшее влияние на cnt
coef = pd.Series(lasso.coef_, index = df.drop('cnt', axis=1).columns)
print(f'Признак, оказывающий наибольшее влияние на cnt: {coef.idxmax()}')
