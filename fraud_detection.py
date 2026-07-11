#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detecção de Fraudes em Transações Financeiras
Curso Python DIO

Este módulo implementa um sistema completo de detecção de fraudes utilizando
machine learning, pandas para manipulação de dados, e bibliotecas de segurança
e logging para monitoramento.

Autor: Sistema DIO
Data: 2024
"""

import os
import sys
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Configuração do Logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Configura o sistema de logging com múltiplos handlers.
    
    Args:
        log_dir: Diretório para armazenar os logs
        
    Returns:
        Logger configurado
    """
    # Cria diretório de logs se não existir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Cria logger
    logger = logging.getLogger("FraudDetectionSystem")
    logger.setLevel(logging.DEBUG)
    
    # Formatação dos logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(
        f"{log_dir}/fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Adiciona handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class SecurityManager:
    """
    Gerenciador de segurança para proteção de dados sensíveis.
    """
    
    @staticmethod
    def hash_data(data: str, salt: str = "dio_fraud_detection") -> str:
        """
        Gera hash seguro para dados sensíveis.
        
        Args:
            data: Dados a serem hasheados
            salt: Salt para aumentar segurança
            
        Returns:
            Hash SHA-256 dos dados
        """
        salted_data = f"{salt}{data}{salt}"
        return hashlib.sha256(salted_data.encode()).hexdigest()
    
    @staticmethod
    def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a integridade dos dados.
        
        Args:
            df: DataFrame para validação
            
        Returns:
            Dicionário com informações de integridade
        """
        integrity_info = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "data_hash": SecurityManager.hash_data(str(df.shape))
        }
        return integrity_info


class FraudDetectionSystem:
    """
    Sistema principal de detecção de fraudes em transações financeiras.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa o sistema de detecção de fraudes.
        
        Args:
            logger: Logger para registro de eventos
        """
        self.logger = logger or setup_logging()
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: list = []
        self.training_metrics: Dict[str, Any] = {}
        self.security_manager = SecurityManager()
        
        self.logger.info("Sistema de Detecção de Fraudes inicializado")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Carrega os dados do CSV.
        
        Args:
            data_path: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados carregados
        """
        self.logger.info(f"Carregando dados de: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Dados carregados com sucesso: {df.shape}")
            
            # Valida integridade dos dados
            integrity = self.security_manager.validate_data_integrity(df)
            self.logger.debug(f"Integridade dos dados: {integrity}")
            
            return df
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def load_data_from_url(self, url: str) -> pd.DataFrame:
        """
        Carrega dados diretamente de uma URL.
        
        Args:
            url: URL do dataset
            
        Returns:
            DataFrame com os dados
        """
        self.logger.info(f"Carregando dados da URL: {url}")
        
        try:
            df = pd.read_csv(url)
            self.logger.info(f"Dados carregados com sucesso: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados da URL: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame, output_dir: str = "output") -> None:
        """
        Realiza análise exploratória dos dados e gera visualizações.
        
        Args:
            df: DataFrame para análise
            output_dir: Diretório para salvar gráficos
        """
        self.logger.info("Iniciando análise exploratória dos dados")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configura estilo dos gráficos
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Distribuição das classes
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de barras
        class_counts = df['Class'].value_counts()
        axes[0].bar(['Legítimas', 'Fraudes'], class_counts.values, 
                    color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Distribuição das Classes', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Quantidade')
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 500, str(v), ha='center', va='bottom', fontsize=12)
        
        # Gráfico de pizza
        axes[1].pie(class_counts.values, labels=['Legítimas', 'Fraudes'],
                   autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'],
                   explode=(0.05, 0.05))
        axes[1].set_title('Proporção das Classes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Gráfico de distribuição salvo em {output_dir}/class_distribution.png")
        
        # 2. Estatísticas descritivas
        stats_file = f"{output_dir}/data_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== ESTATÍSTICAS DESCRITIVAS ===\n\n")
            f.write(f"Total de transações: {len(df)}\n")
            f.write(f"Transações legítimas: {class_counts.get(0, 0)}\n")
            f.write(f"Transações fraudulentas: {class_counts.get(1, 0)}\n")
            f.write(f"Percentual de fraudes: {(class_counts.get(1, 0) / len(df) * 100):.4f}%\n\n")
            f.write("=== DESCRIÇÃO DOS DADOS ===\n\n")
            f.write(df.describe().to_string())
        self.logger.info(f"Estatísticas salvas em {stats_file}")
        
        # 3. Correlação entre features
        plt.figure(figsize=(16, 12))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Matriz de Correlação entre Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Matriz de correlação salva em {output_dir}/correlation_matrix.png")
        
        # 4. Distribuição do Amount por classe
        plt.figure(figsize=(12, 6))
        df_plot = df.copy()
        df_plot['Class_Label'] = df_plot['Class'].map({0: 'Legítima', 1: 'Fraude'})
        sns.boxplot(x='Class_Label', y='Amount', data=df_plot, 
                   palette=['#2ecc71', '#e74c3c'])
        plt.title('Distribuição do Valor das Transações por Classe', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Classe')
        plt.ylabel('Valor (Amount)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/amount_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Distribuição de Amount salva em {output_dir}/amount_distribution.png")
        
        # 5. Análise temporal (Time)
        plt.figure(figsize=(14, 6))
        for class_val in [0, 1]:
            subset = df[df['Class'] == class_val]
            plt.scatter(subset['Time'][::100], subset['Amount'][::100], 
                       alpha=0.3, label=f'Classe {class_val}',
                       c='#e74c3c' if class_val == 1 else '#2ecc71')
        plt.xlabel('Tempo (segundos)')
        plt.ylabel('Valor da Transação')
        plt.title('Distribuição Temporal das Transações', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Análise temporal salva em {output_dir}/temporal_analysis.png")
        
        self.logger.info("Análise exploratória concluída")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pré-processa os dados para treinamento.
        
        Args:
            df: DataFrame original
            
        Returns:
            Tupla com (features, target)
        """
        self.logger.info("Iniciando pré-processamento dos dados")
        
        # Separa features e target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Identifica colunas de features
        self.feature_columns = X.columns.tolist()
        
        # Normalização dos dados - apenas colunas numéricas exceto Time
        self.scaler = StandardScaler()
        
        # Converte para numpy arrays para economizar memória
        X_array = X.values.astype(np.float32)
        
        # Normaliza todas as colunas exceto Time (índice 0)
        cols_to_scale = [i for i, col in enumerate(self.feature_columns) if col != 'Time']
        X_array[:, cols_to_scale] = self.scaler.fit_transform(X_array[:, cols_to_scale])
        
        # Libera memória
        del X, df
        
        self.logger.info(f"Pré-processamento concluído. Features: {X_array.shape}")
        
        return X_array, y.values
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Treina o modelo de detecção de fraudes.
        
        Args:
            X: Features (numpy array)
            y: Target (numpy array)
            model_type: Tipo de modelo ('random_forest' ou 'logistic_regression')
            
        Returns:
            Dicionário com métricas do modelo
        """
        self.logger.info(f"Iniciando treinamento do modelo: {model_type}")
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.logger.info(f"Divisão: Treino={X_train.shape}, Teste={X_test.shape}")
        
        # Seleciona o modelo - usando Logistic Regression por ser mais leve
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
                C=0.1
            )
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")
        
        # Treina o modelo
        self.logger.info("Treinando modelo...")
        self.model.fit(X_train, y_train)
        
        # Libera memória
        del X_train, y_train
        
        # Faz predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcula métricas
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        
        metrics = {
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_curve": (fpr, tpr, thresholds_roc),
            "precision_recall": (precision, recall, thresholds_pr),
            "average_precision": average_precision_score(y_test, y_pred_proba)
        }
        
        # Armazena métricas
        self.training_metrics = {
            "model_type": model_type,
            "test_size": len(y_test),
            "train_size": X_test.shape[0] * 7 // 3,  # Aproximado
            "roc_auc_score": auc(fpr, tpr),
            "average_precision": metrics["average_precision"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Treinamento concluído. ROC-AUC: {self.training_metrics['roc_auc_score']:.4f}")
        self.logger.info(f"\n{metrics['classification_report']}")
        
        return metrics
    
    def generate_visualizations(self, metrics: Dict[str, Any], 
                               output_dir: str = "output") -> None:
        """
        Gera visualizações das métricas do modelo.
        
        Args:
            metrics: Métricas do modelo
            output_dir: Diretório para salvar gráficos
        """
        self.logger.info("Gerando visualizações do modelo")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Matriz de Confusão
        plt.figure(figsize=(10, 8))
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legítima', 'Fraude'],
                   yticklabels=['Legítima', 'Fraude'])
        plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Matriz de confusão salva em {output_dir}/confusion_matrix.png")
        
        # 2. Curva ROC
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = metrics["roc_curve"]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC - Detecção de Fraudes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Curva ROC salva em {output_dir}/roc_curve.png")
        
        # 3. Curva Precision-Recall
        plt.figure(figsize=(10, 8))
        precision, recall, _ = metrics["precision_recall"]
        avg_precision = metrics["average_precision"]
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Curva Precision-Recall salva em {output_dir}/precision_recall_curve.png")
        
        # 4. Feature Importance (se for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            feature_names = [self.feature_columns[i] for i in indices]
            plt.barh(range(len(indices)), importances[indices][::-1])
            plt.yticks(range(len(indices)), feature_names)
            plt.xlabel('Importância')
            plt.title('Top 15 Features Mais Importantes', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Feature importance salvo em {output_dir}/feature_importance.png")
        
        self.logger.info("Visualizações geradas com sucesso")
    
    def save_model(self, model_path: str = "models") -> None:
        """
        Salva o modelo treinado.
        
        Args:
            model_path: Diretório para salvar o modelo
        """
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Salva o modelo
        joblib.dump(self.model, f"{model_path}/fraud_detection_model.joblib")
        joblib.dump(self.scaler, f"{model_path}/scaler.joblib")
        joblib.dump(self.feature_columns, f"{model_path}/feature_columns.joblib")
        joblib.dump(self.training_metrics, f"{model_path}/training_metrics.json")
        
        self.logger.info(f"Modelo salvo em {model_path}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com o modelo treinado.
        
        Args:
            data: Dados para predição
            
        Returns:
            Array com predições (0 ou 1)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train_model primeiro.")
        
        # Aplica o mesmo pré-processamento
        cols_to_scale = [col for col in data.columns if col != 'Time']
        data_scaled = data.copy()
        data_scaled[cols_to_scale] = self.scaler.transform(data[cols_to_scale])
        
        predictions = self.model.predict(data_scaled)
        probabilities = self.model.predict_proba(data_scaled)[:, 1]
        
        return predictions, probabilities
    
    def generate_report(self, output_dir: str = "output") -> str:
        """
        Gera relatório completo do sistema.
        
        Args:
            output_dir: Diretório para salvar o relatório
            
        Returns:
            Caminho do relatório
        """
        report_path = f"{output_dir}/fraud_detection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DO SISTEMA DE DETECÇÃO DE FRAUDES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Tipo de Modelo: {self.training_metrics.get('model_type', 'N/A')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("MÉTRICAS DO MODELO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Tamanho do Conjunto de Treino: {self.training_metrics.get('train_size', 'N/A')}\n")
            f.write(f"Tamanho do Conjunto de Teste: {self.training_metrics.get('test_size', 'N/A')}\n")
            f.write(f"ROC-AUC Score: {self.training_metrics.get('roc_auc_score', 'N/A'):.4f}\n")
            f.write(f"Average Precision: {self.training_metrics.get('average_precision', 'N/A'):.4f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("MATRIZ DE CONFUSÃO\n")
            f.write("-" * 80 + "\n")
            cm = self.training_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            f.write(f"                Predito\n")
            f.write(f"                Legítima    Fraude\n")
            f.write(f"Real Legítima   {cm[0][0]:10d}  {cm[0][1]:10d}\n")
            f.write(f"Real Fraude     {cm[1][0]:10d}  {cm[1][1]:10d}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Relatório salvo em {report_path}")
        return report_path


def main():
    """
    Função principal que executa todo o pipeline de detecção de fraudes.
    """
    print("=" * 80)
    print("SISTEMA DE DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS")
    print("Curso Python DIO")
    print("=" * 80)
    print()
    
    # Inicializa o sistema
    logger = setup_logging()
    logger.info("Iniciando sistema de detecção de fraudes")
    
    # URL do dataset
    DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    
    # Inicializa o sistema
    fraud_system = FraudDetectionSystem(logger)
    
    try:
        # 1. Carrega os dados
        print("\n[1/6] Carregando dados...")
        df = fraud_system.load_data_from_url(DATA_URL)
        print(f"✓ Dados carregados: {df.shape[0]} transações, {df.shape[1]} features")
        
        # 2. Análise exploratória
        print("\n[2/6] Realizando análise exploratória...")
        fraud_system.explore_data(df, output_dir="output")
        print("✓ Análise exploratória concluída")
        
        # 3. Pré-processamento
        print("\n[3/6] Pré-processando dados...")
        X, y = fraud_system.preprocess_data(df)
        print(f"✓ Dados pré-processados: {X.shape}")
        
        # 4. Treinamento do modelo
        print("\n[4/6] Treinando modelo (Random Forest)...")
        metrics = fraud_system.train_model(X, y, model_type="random_forest")
        print(f"✓ Modelo treinado com ROC-AUC: {fraud_system.training_metrics['roc_auc_score']:.4f}")
        
        # 5. Gera visualizações
        print("\n[5/6] Gerando visualizações...")
        fraud_system.generate_visualizations(metrics, output_dir="output")
        print("✓ Visualizações geradas")
        
        # 6. Salva modelo e relatório
        print("\n[6/6] Salvando modelo e relatório...")
        fraud_system.save_model(model_path="models")
        fraud_system.generate_report(output_dir="output")
        print("✓ Modelo e relatório salvos")
        
        # Resumo final
        print("\n" + "=" * 80)
        print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("\nArquivos gerados:")
        print("  📊 output/class_distribution.png - Distribuição das classes")
        print("  📊 output/correlation_matrix.png - Matriz de correlação")
        print("  📊 output/amount_distribution.png - Distribuição de valores")
        print("  📊 output/temporal_analysis.png - Análise temporal")
        print("  📊 output/confusion_matrix.png - Matriz de confusão")
        print("  📊 output/roc_curve.png - Curva ROC")
        print("  📊 output/precision_recall_curve.png - Curva Precision-Recall")
        print("  📊 output/feature_importance.png - Importância das features")
        print("  📄 output/fraud_detection_report.txt - Relatório completo")
        print("  📁 models/ - Modelos treinados")
        print("  📁 logs/ - Logs do sistema")
        print()
        
        logger.info("Sistema executado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        print(f"\n❌ Erro: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
