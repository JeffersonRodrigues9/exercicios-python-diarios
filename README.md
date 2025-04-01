# Projetos-com-python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Gerar dados de exemplo
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 pontos de dados entre 0 e 10
y = 2.5 * X + np.random.randn(100, 1) * 2  # Uma relação linear com um pouco de ruído

# Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Predição
y_pred = model.predict(X)

# Criar o gráfico
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Dados')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regressão Linear')

# Títulos e legendas
plt.title('Exemplo de Machine Learning: Regressão Linear', fontsize=14)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()

# Salvar a imagem
plt.savefig('ml_logo.png', dpi=300)
plt.show()
