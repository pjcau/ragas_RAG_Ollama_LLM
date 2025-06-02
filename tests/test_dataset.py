import pytest
from src.evaluator.dataset_validator import RAGEvaluator

@pytest.fixture
def sample_dataset():
    return [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "contexts": [
                "Paris is the capital of France.",
                "France is located in Europe.",
                "The Eiffel Tower is in Paris."
            ],
            "ground_truth": "The capital of France is Paris."
        }
    ]

def test_validate_and_fix_dataset_valid(sample_dataset):
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is not None
    assert len(corrected_dataset) == 1
    assert corrected_dataset[0]['question'] == "What is the capital of France?"
    assert corrected_dataset[0]['answer'] == "The capital of France is Paris."
    assert len(corrected_dataset[0]['contexts']) <= 5

def test_validate_and_fix_dataset_empty_question(sample_dataset):
    sample_dataset[0]['question'] = ""
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is not None
    assert corrected_dataset[0]['question'] == "What information does this document provide?"

def test_validate_and_fix_dataset_missing_answer(sample_dataset):
    sample_dataset[0]['answer'] = ""
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is None

def test_validate_and_fix_dataset_missing_contexts(sample_dataset):
    sample_dataset[0]['contexts'] = []
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is None

def test_validate_and_fix_dataset_invalid_contexts(sample_dataset):
    sample_dataset[0]['contexts'] = ["Short", "Another short"]
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is None

def test_validate_and_fix_dataset_missing_ground_truth(sample_dataset):
    sample_dataset[0]['ground_truth'] = ""
    evaluator = RAGEvaluator()
    corrected_dataset = evaluator.validate_and_fix_dataset(sample_dataset)
    
    assert corrected_dataset is not None
    assert corrected_dataset[0]['ground_truth'] == "The capital of France is Paris."