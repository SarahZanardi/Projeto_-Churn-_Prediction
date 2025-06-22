# 🔍 Predição de Churn de Clientes com Machine Learning

Este projeto tem como objetivo prever se um cliente irá cancelar ou continuar utilizando os serviços de uma empresa de telecomunicações, utilizando técnicas de machine learning aplicadas ao dataset público da Telco (Kaggle).

## 📊 Objetivo

Desenvolver um modelo preditivo capaz de identificar clientes com maior propensão ao churn (cancelamento), auxiliando a empresa a tomar ações proativas de retenção.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Python 3**
- `pandas` – manipulação de dados
- `numpy` – operações numéricas
- `matplotlib` e `seaborn` – visualizações
- `scikit-learn` – modelagem preditiva

---

## 📁 Dataset

- Fonte: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Total de registros: 7.043 clientes
- Variável alvo: `Churn` (Sim/Não)

---

## 📌 Etapas do Projeto

1. **Importação dos dados**
2. **Limpeza e tratamento**
   - Conversão de colunas numéricas
   - Tratamento de valores ausentes
   - Codificação de variáveis categóricas
3. **Divisão treino/teste**
4. **Normalização**
5. **Treinamento com Random Forest**
6. **Avaliação do modelo**
   - Acurácia
   - Matriz de confusão
   - Principais variáveis preditoras

---

## 📈 Resultados

- **Acurácia do modelo:** ~79%
- **Principais variáveis preditoras:**
  - `MonthlyCharges`
  - `Contract_Two year`
  - `tenure`
  - `InternetService_Fiber optic`
  - entre outras...

---

## 🖼️ Visualizações

- Distribuição de clientes com/sem churn
- Matriz de confusão
- Gráfico de importância das features

---

## ✅ Conclusão

O modelo apresentou desempenho satisfatório, com boa capacidade de identificar clientes propensos ao cancelamento. Essa análise é fundamental para empresas que desejam aumentar a retenção e reduzir custos com aquisição de novos clientes.

---

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/SarahZanardi/Projeto_-Churn-_Prediction.git
   cd Projeto_-Churn-_Prediction
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o notebook principal ou o script de análise.

---

## 📬 Contato

Em caso de dúvidas, sugestões ou contribuições, fique à vontade para abrir uma issue ou entrar em contato pelo [GitHub](https://github.com/SarahZanardi).

---