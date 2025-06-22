# ========================
# 1. Importação de bibliotecas
# ========================

import pandas as pd              # Usada para manipulação de dados em formato de tabela
import numpy as np               # Operações matemáticas e arrays numéricos

import matplotlib.pyplot as plt  # Geração de gráficos simples
import seaborn as sns            # Gráficos estatísticos mais elaborados

# Módulos do scikit-learn para machine learning
from sklearn.model_selection import train_test_split               # Divisão de dados em treino e teste
from sklearn.preprocessing import LabelEncoder, StandardScaler     # Codificação e padronização dos dados
from sklearn.ensemble import RandomForestClassifier                # Modelo Random Forest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Métricas de avaliação

# ========================
# 2. Carregamento dos dados
# ========================

# Lê o dataset de churn
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Visualiza as primeiras linhas do dataset
df.head()

# ========================
# 3. Análise exploratória
# ========================

# Verifica estrutura dos dados: tipos de colunas, nulos, etc.
df.info()

# Estatísticas descritivas (média, min, max, etc.)
df.describe()

# Frequência dos valores da variável alvo (Churn)
print(df['Churn'].value_counts(normalize=True))

# Gráfico da distribuição de churn
sns.countplot(x='Churn', data=df)
plt.title('Distribuição de Churn')
plt.show()

# ========================
# 4. Limpeza e tratamento
# ========================

# Remove coluna ID que não agrega valor à predição
df = df.drop(columns=['customerID'])

# Verifica valores nulos
print(df.isnull().sum())

# Converte TotalCharges de texto para número (alguns registros estão como string vazia)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Preenche valores ausentes com a mediana
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# ========================
# 5. Codificação de variáveis categóricas
# ========================

# Identifica colunas categóricas
cat_cols = df.select_dtypes(include='object').columns

# Inicializa o codificador para variáveis binárias
le = LabelEncoder()

# Aplica codificação adequada
for col in cat_cols:
    if df[col].nunique() == 2:
        # Label Encoding para variáveis com 2 categorias (binárias)
        df[col] = le.fit_transform(df[col])
    else:
        # One-Hot Encoding para variáveis com mais de 2 categorias
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ========================
# 6. Separação das variáveis
# ========================

# Define variável alvo
y = df['Churn']

# Define variáveis explicativas
X = df.drop('Churn', axis=1)

# ========================
# 7. Divisão treino/teste
# ========================

# Divide os dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 8. Padronização dos dados
# ========================

# Padroniza os dados para média 0 e desvio 1
scaler = StandardScaler()

# Aplica a padronização
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================
# 9. Treinamento do modelo
# ========================

# Instancia o modelo Random Forest
model = RandomForestClassifier(random_state=42)

# Treina o modelo com os dados de treino
model.fit(X_train, y_train)

# ========================
# 10. Avaliação do modelo
# ========================

# Faz previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliação de acurácia (porcentagem de acertos)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Exibe métricas detalhadas: precisão, recall, f1-score
print(classification_report(y_test, y_pred))

# Matriz de confusão (visualiza erros e acertos)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Avaliação do Modelo de Churn')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# ========================
# 11. Importância das variáveis
# ========================

# Gera gráfico das 10 variáveis mais importantes
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Features Mais Importantes para o Churn')
plt.xlabel('Importância')
plt.show()
