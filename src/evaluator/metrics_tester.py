from evaluator.rag_evaluator import RAGEvaluator

def test_metrics(rag_evaluator: RAGEvaluator, test_dataset):
    """Test various metrics using the RAGEvaluator instance."""
    results = {}
    
    # Define metrics to test
    metrics_to_test = [
        ("faithfulness", rag_evaluator.test_faithfulness),
        ("answer_relevancy", rag_evaluator.test_answer_relevancy),
        ("context_precision", rag_evaluator.test_context_precision),
        ("context_recall", rag_evaluator.test_context_recall),
    ]
    
    for metric_name, metric_func in metrics_to_test:
        print(f"Testing {metric_name}...")
        result = rag_evaluator.test_metric_with_retries(metric_name, metric_func, test_dataset)
        results[metric_name] = result
    
    return results

def run_all_tests():
    """Run all metric tests and print results."""
    evaluator = RAGEvaluator()
    test_dataset = evaluator.create_test_dataset_complete()
    
    if not test_dataset:
        print("No valid test dataset available.")
        return
    
    results = test_metrics(evaluator, test_dataset)
    
    for metric_name, result in results.items():
        if result['success']:
            print(f"{metric_name}: Success with score {result['score']:.4f}")
        else:
            print(f"{metric_name}: Failed with error {result['error']}")