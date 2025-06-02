import unittest
from src.evaluator.rag_evaluator import RAGEvaluator

class TestRAGEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = RAGEvaluator()

    def test_create_ultra_compatible_llm(self):
        llm = self.evaluator.create_ultra_compatible_llm()
        self.assertIsNotNone(llm)
        self.assertEqual(llm.model, "llama3.2")

    def test_create_robust_embeddings(self):
        embeddings = self.evaluator.create_robust_embeddings()
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.model, "nomic-embed-text")

    def test_validate_and_fix_dataset_empty(self):
        result = self.evaluator.validate_and_fix_dataset([])
        self.assertIsNone(result)

    def test_validate_and_fix_dataset_missing_question(self):
        dataset = [{'answer': 'This is an answer.', 'contexts': ['Context 1', 'Context 2']}]
        result = self.evaluator.validate_and_fix_dataset(dataset)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]['question'], "What information does this document provide?")

    def test_validate_and_fix_dataset_missing_answer(self):
        dataset = [{'question': 'What is this?', 'contexts': ['Context 1', 'Context 2']}]
        result = self.evaluator.validate_and_fix_dataset(dataset)
        self.assertIsNone(result)

    def test_validate_and_fix_dataset_missing_contexts(self):
        dataset = [{'question': 'What is this?', 'answer': 'This is an answer.'}]
        result = self.evaluator.validate_and_fix_dataset(dataset)
        self.assertIsNone(result)

    def test_test_metric_with_retries_success(self):
        # Mocking the evaluate function and its behavior would be necessary here
        pass

    def test_test_metric_with_retries_failure(self):
        # Mocking the evaluate function and its behavior would be necessary here
        pass

    def test_test_individual_metrics_enhanced(self):
        # Mocking the test of individual metrics would be necessary here
        pass

if __name__ == '__main__':
    unittest.main()