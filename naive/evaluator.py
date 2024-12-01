from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from llm_interface import LLMInterface

class Evaluator:
    def __init__(self, model: LLMInterface, model_name: str = "unknown"):
        self.model = model
        self.model_name = model_name
        # Initialize the sentence transformer model
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Similarity threshold for considering answers as matching
        self.similarity_threshold = 0.8

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_df: DataFrame with columns ['Prompt', 'Answer', 'reasoning_types']
        
        Returns:
            Dictionary containing overall metrics and metrics by reasoning type
        """
        print(f"\nEvaluating {self.model_name}...")
        # Generate predictions
        predictions = self.model.batch_generate(test_df['Prompt'].tolist())
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(predictions, test_df['Answer'].tolist())
        
        # Calculate metrics by reasoning type
        type_metrics = self._evaluate_by_reasoning_type(test_df, predictions)
        
        return {
            'model': self.model_name,
            'overall': overall_metrics,
            'by_type': type_metrics
        }

    def _calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics using embedding similarity"""
        # Get embeddings for predictions and ground truth
        pred_embeddings = self.sentence_model.encode(predictions)
        truth_embeddings = self.sentence_model.encode(ground_truth)
        
        # Calculate cosine similarity between predictions and ground truth
        similarities = np.zeros(len(predictions))
        for i, (pred_emb, truth_emb) in enumerate(zip(pred_embeddings, truth_embeddings)):
            similarity = np.dot(pred_emb, truth_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(truth_emb))
            similarities[i] = similarity
        
        # Convert similarities to binary predictions based on threshold
        binary_preds = similarities >= self.similarity_threshold
        binary_truth = np.ones(len(ground_truth), dtype=bool)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(binary_truth, binary_preds),
            'precision': precision_score(binary_truth, binary_preds),
            'recall': recall_score(binary_truth, binary_preds),
            'f1': f1_score(binary_truth, binary_preds),
            'mean_similarity': np.mean(similarities)
        }
        
        return metrics

    def _evaluate_by_reasoning_type(self, df: pd.DataFrame, predictions: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by reasoning type"""
        metrics_by_type = {}
        
        for reasoning_type in df['reasoning_types'].unique():
            mask = df['reasoning_types'] == reasoning_type
            type_preds = [p for p, m in zip(predictions, mask) if m]
            type_truth = df[mask]['Answer'].tolist()
            
            if type_preds:
                metrics_by_type[reasoning_type] = self._calculate_metrics(type_preds, type_truth)
        
        return metrics_by_type
