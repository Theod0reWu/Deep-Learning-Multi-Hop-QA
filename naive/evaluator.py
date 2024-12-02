from typing import Dict, List
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from llm_interface import LLMInterface

import re

from rapidfuzz.fuzz import ratio

def preprocess(text):
    """Remove all non-alphanumeric characters and convert to lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', text.strip()).lower()

class Evaluator:
    def __init__(self, model: LLMInterface):
        self.model = model

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_df: DataFrame with columns ['prompt', 'answer', 'reasoning_types']
        
        Returns:
            Dictionary containing overall metrics and metrics by reasoning type
        """
        # Generate predictions
        predictions = self.model.batch_generate(test_df['Prompt'].tolist())
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(predictions, test_df['Answer'].tolist())
        
        # Calculate metrics by reasoning type
        # type_metrics = self._evaluate_by_reasoning_type(test_df, predictions)
        type_metrics = {}
        
        return {
            'overall': overall_metrics,
            'by_type': type_metrics
        }

    def _calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Convert string predictions to binary (correct/incorrect)
        threshold = 95
        binary_preds = [preprocess(truth) in preprocess(pred) or preprocess(pred) in preprocess(truth) or ratio(pred, truth) >= threshold
                       for pred, truth in zip(predictions, ground_truth)]
        binary_truth = [1] * len(ground_truth)
        
        return {
            'accuracy': accuracy_score(binary_truth, binary_preds),
            'f1': f1_score(binary_truth, binary_preds)
        }
    ## 50%
    '''
    Overall Metrics:
{
  "accuracy": 0.2545454545454545,
  "f1": 0.4057971014492754
}
    '''
    ## 80% 
    '''
    Overall Metrics:
    {
      "accuracy": 0.15757575757575756,
      "f1": 0.27225130890052357
    }
    '''
    # def _calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    #     """Calculate evaluation metrics"""
    #     # Convert string predictions to binary (correct/incorrect)
    #     binary_preds = [pred.strip().lower() == truth.strip().lower() 
    #                    for pred, truth in zip(predictions, ground_truth)]
    #     binary_truth = [1] * len(ground_truth)
        
    #     return {
    #         'accuracy': accuracy_score(binary_truth, binary_preds),
    #         'precision': precision_score(binary_truth, binary_preds),
    #         'recall': recall_score(binary_truth, binary_preds),
    #         'f1': f1_score(binary_truth, binary_preds)
    #     }

    def _evaluate_by_reasoning_type(
        self,
        df: pd.DataFrame,
        predictions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by reasoning type"""
        metrics_by_type = {}
        
        for reasoning_type in df['reasoning_types'].unique():
            mask = df['reasoning_types'] == reasoning_type
            type_preds = [p for p, m in zip(predictions, mask) if m]
            type_truth = df[mask]['Answer'].tolist()
            
            if type_preds:
                metrics_by_type[reasoning_type] = self._calculate_metrics(type_preds, type_truth)
        
        return metrics_by_type
