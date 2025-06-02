import pytest
from src.evaluator.rag_evaluator import RAGEvaluator
from src.evaluator.metrics_tester import test_metric_with_retries
from src.data.dataset_creator import create_test_dataset_complete

class TestRAGEvaluator:
    def setup_method(self):
        self.evaluator = RAGEvaluator()
        self.test_dataset = create_test_dataset_complete()
        self.validated_dataset = self.evaluator.validate_and_fix_dataset(self.test_dataset)

    def test_validate_and_fix_dataset(self):
        assert self.validated_dataset is not None, "Dataset validation failed"
        assert 'question' in self.validated_dataset[0], "Question field is missing"
        assert 'answer' in self.validated_dataset[0], "Answer field is missing"
        assert 'contexts' in self.validated_dataset[0], "Contexts field is missing"
        assert len(self.validated_dataset[0]['contexts']) > 0, "No valid contexts found"

    def test_metric_with_retries_success(self):
        metric_name = "test_metric"
        metric_obj = lambda x: x  # Dummy metric function for testing
        result = test_metric_with_retries(metric_name, metric_obj, self.validated_dataset, self.evaluator.create_ultra_compatible_llm(), self.evaluator.create_robust_embeddings())
        assert result['success'], f"Metric testing failed: {result['error']}"

    def test_metric_with_retries_failure(self):
        metric_name = "test_metric"
        metric_obj = lambda x: float('nan')  # Simulate a failure
        result = test_metric_with_retries(metric_name, metric_obj, self.validated_dataset, self.evaluator.create_ultra_compatible_llm(), self.evaluator.create_robust_embeddings())
        assert not result['success'], "Expected failure but succeeded"