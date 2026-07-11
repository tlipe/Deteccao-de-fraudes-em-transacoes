# Detecção de Fraudes em Transações Financeiras

## 📋 Visão Geral

Este projeto implementa um **Sistema Completo de Detecção de Fraudes em Transações Financeiras** utilizando Python, Machine Learning e boas práticas de engenharia de software. Desenvolvido para o curso de Python da DIO.

O sistema utiliza o dataset público de cartões de crédito disponível em: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## 🎯 Objetivos

- Detectar transações fraudulentas em dados de cartão de crédito
- Implementar pipeline completo de ML com pré-processamento, treinamento e avaliação
- Gerar visualizações gráficas para análise dos dados e resultados
- Utilizar logging e segurança para monitoramento e proteção de dados
- Documentar todo o processo para fins educacionais

## 🏗️ Arquitetura do Sistema

```
fraud_detection.py
├── SecurityManager         # Gerenciamento de segurança e integridade de dados
├── FraudDetectionSystem    # Sistema principal de detecção
│   ├── load_data_from_url  # Carregamento de dados da API pública
│   ├── explore_data        # Análise exploratória com gráficos
│   ├── preprocess_data     # Normalização e preparação
│   ├── train_model         # Treinamento do modelo (Random Forest)
│   ├── generate_visualizations  # Gráficos de métricas
│   ├── save_model          # Persistência do modelo
│   └── generate_report     # Relatório final
└── main                    # Pipeline completo
```

## 📦 Dependências

```bash
pip install pandas scikit-learn matplotlib seaborn numpy joblib
```

## 🚀 Como Executar

```bash
python3 fraud_detection.py
```

### Saída Esperada

O sistema executará as seguintes etapas:

1. **Carregamento de Dados** - Download do dataset da URL pública
2. **Análise Exploratória** - Geração de gráficos estatísticos
3. **Pré-processamento** - Normalização das features
4. **Treinamento** - Modelo Random Forest com balanceamento de classes
5. **Visualizações** - Gráficos de desempenho do modelo
6. **Salvamento** - Modelo, relatório e logs

## 📊 Resultados Gerados

### Diretório `output/`

| Arquivo | Descrição |
|---------|-----------|
| `class_distribution.png` | Distribuição das classes (legítimas vs fraudes) |
| `correlation_matrix.png` | Matriz de correlação entre features |
| `amount_distribution.png` | Distribuição dos valores das transações |
| `temporal_analysis.png` | Análise temporal das transações |
| `confusion_matrix.png` | Matriz de confusão do modelo |
| `roc_curve.png` | Curva ROC - AUC Score |
| `precision_recall_curve.png` | Curva Precision-Recall |
| `feature_importance.png` | Importância das features no modelo |
| `fraud_detection_report.txt` | Relatório completo em texto |
| `data_statistics.txt` | Estatísticas descritivas dos dados |

### Diretório `models/`

| Arquivo | Descrição |
|---------|-----------|
| `fraud_detection_model.joblib` | Modelo treinado |
| `scaler.joblib` | Normalizador StandardScaler |
| `feature_columns.joblib` | Colunas das features |
| `training_metrics.json` | Métricas do treinamento |

### Diretório `logs/`

Logs detalhados de cada execução com timestamps.

## 📈 Métricas do Modelo

O modelo Random Forest alcança excelentes resultados:

- **ROC-AUC Score**: ~0.97
- **Precision**: 0.58 (classe fraude)
- **Recall**: 0.83 (classe fraude) - detecta 83% das fraudes
- **Accuracy**: 99.9% (cuidado com datasets desbalanceados!)

### Matriz de Confusão Típica

```
                Predito
                Legítima    Fraude
Real Legítima      85207        88
Real Fraude           25       123
```

## 🔒 Recursos de Segurança

### SecurityManager

- **Hash de Dados**: Utiliza SHA-256 com salt para proteger informações sensíveis
- **Validação de Integridade**: Verifica null values, duplicatas e consistência dos dados
- **Logging Seguro**: Logs em arquivo com controle de acesso

## 📝 Logging

O sistema utiliza o módulo `logging` do Python com:

- **Handler de Arquivo**: Logs detalhados (DEBUG) salvos em `logs/`
- **Handler de Console**: Informações principais (INFO) em tempo real
- **Formato Padronizado**: Timestamp, nível, módulo e mensagem

Exemplo de log:
```
2026-07-11 22:10:07 - FraudDetectionSystem - INFO - Iniciando sistema de detecção de fraudes
2026-07-11 22:11:22 - FraudDetectionSystem - INFO - Treinamento concluído. ROC-AUC: 0.9738
```

## 🧠 Modelo de Machine Learning

### Random Forest Classifier

Parâmetros utilizados:

```python
RandomForestClassifier(
    n_estimators=50,      # Número de árvores
    max_depth=8,          # Profundidade máxima
    min_samples_split=10, # Mínimo de amostras para split
    min_samples_leaf=5,   # Mínimo de amostras por folha
    class_weight='balanced', # Balanceamento automático
    random_state=42,
    n_jobs=-1             # Paralelização total
)
```

### Por que Random Forest?

- ✅ Lida bem com datasets desbalanceados
- ✅ Resistente a overfitting
- ✅ Fornece feature importance
- ✅ Não requer normalização rigorosa
- ✅ Parallelizável

## 📉 Desafio do Dataset Desbalanceado

O dataset possui apenas **0.17% de transações fraudulentas**:

- Total: 284,807 transações
- Legítimas: 284,315 (99.83%)
- Fraudes: 492 (0.17%)

### Estratégias Utilizadas

1. **Class Weight Balanced**: Pesa automaticamente as classes
2. **Stratified Split**: Mantém proporção nas divisões treino/teste
3. **Métricas Adequadas**: ROC-AUC, Precision-Recall (não apenas accuracy)

## 🔍 Features do Dataset

O dataset contém 31 colunas:

- **Time**: Segundos desde a primeira transação
- **V1-V28**: Features anonimizadas (PCA)
- **Amount**: Valor da transação
- **Class**: Target (0 = legítima, 1 = fraude)

## 📚 Conceitos Demonstrados

### Python Avançado
- Type hints (`typing`)
- Classes e métodos
- Context managers
- Pathlib para manipulação de arquivos

### Data Science
- Pandas para manipulação de dados
- NumPy para operações numéricas
- Scikit-learn para ML
- Matplotlib/Seaborn para visualização

### Engenharia de Software
- Logging estruturado
- Tratamento de exceções
- Documentação de código
- Organização modular

### Segurança
- Hashing com hashlib
- Validação de integridade
- Proteção de dados sensíveis

## 🎓 Aprendizados do Projeto

1. **Importância do balanceamento** em problemas de classificação
2. **Escolha adequada de métricas** para datasets desbalanceados
3. **Valor do logging** para debugging e monitoramento
4. **Segurança de dados** mesmo em projetos educacionais
5. **Visualização** como ferramenta de análise

## 📄 Licença

Projeto educacional desenvolvido para o curso de Python da DIO.

## 👨‍💻 Autor

Sistema desenvolvido como parte do curso de Python da DIO (Digital Innovation One).

---

**🎉 Execute o sistema e explore os gráficos gerados na pasta `output/`!**