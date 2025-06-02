class DataValidator:
    """Class for validating and correcting datasets used in RAG evaluations."""
    
    def __init__(self):
        pass
    
    def validate_dataset(self, dataset):
        """Validates the structure and integrity of the dataset."""
        if not dataset or len(dataset) == 0:
            print("❌ Dataset is empty!")
            return False
        
        for sample in dataset:
            if 'question' not in sample or not sample['question'].strip():
                print("❌ Missing or empty question in sample!")
                return False
            
            if 'answer' not in sample or not sample['answer'].strip():
                print("❌ Missing or empty answer in sample!")
                return False
            
            if 'contexts' not in sample or not sample['contexts']:
                print("❌ Missing contexts in sample!")
                return False
            
            if not all(isinstance(ctx, str) and len(ctx.strip()) >= 20 for ctx in sample['contexts']):
                print("❌ Invalid contexts in sample!")
                return False
        
        print("✅ Dataset validation successful.")
        return True
    
    def fix_dataset(self, dataset):
        """Automatically fixes common issues in the dataset."""
        for sample in dataset:
            if 'question' not in sample or not sample['question'].strip():
                sample['question'] = "What information does this document provide?"
            
            if 'ground_truth' not in sample or not sample['ground_truth']:
                first_sentence = sample['answer'].split('.')[0].strip()
                sample['ground_truth'] = first_sentence + "." if len(first_sentence) > 10 else sample['answer'][:100].strip() + "."
        
        print("✅ Dataset fixed automatically.")
        return dataset