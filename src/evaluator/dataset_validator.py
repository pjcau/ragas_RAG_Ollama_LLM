# Validazione e creazione dataset per RAGAS
from ..config.settings import DATASET_CONFIG

# Import condizionali
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    # Mock Dataset class

    class MockDataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_list(cls, data_list):
            return cls(data_list)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

    Dataset = MockDataset


class DatasetValidator:

    @staticmethod
    def create_test_dataset_complete():
        test_data = [{
            'question': 'What is machine learning and how does it work?',
            'answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It works by using algorithms to identify patterns in data, training models on these patterns, and then using the trained models to make predictions or decisions on new, unseen data. The process typically involves data collection, preprocessing, feature selection, model training, validation, and deployment.',
            'contexts': [
                'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.',
                'The machine learning process involves several key steps: data collection and preparation, choosing an appropriate algorithm, training the model on a dataset, evaluating the model performance, and fine-tuning parameters to improve accuracy.',
                'Common machine learning algorithms include supervised learning (like linear regression and decision trees), unsupervised learning (like clustering and dimensionality reduction), and reinforcement learning (where agents learn through interaction with an environment).',
                'Machine learning applications are widespread, including recommendation systems, image recognition, natural language processing, fraud detection, and autonomous vehicles. The field continues to evolve with advances in deep learning and neural networks.'
            ],
            'ground_truth': 'Machine learning is a subset of AI that uses algorithms to learn patterns from data and make predictions without explicit programming.'
        }]

        return Dataset.from_list(test_data)

    @staticmethod
    def validate_dataset(dataset):
        if not dataset or len(dataset) == 0:
            return False

        required_fields = ['question', 'answer', 'contexts']

        for item in dataset:
            # Controlla campi obbligatori
            for field in required_fields:
                if field not in item:
                    print(f"❌ Missing field: {field}")
                    return False

                if not item[field]:
                    print(f"❌ Empty field: {field}")
                    return False

            # Valida specificamente i contexts
            if not isinstance(item['contexts'], list):
                print(f"❌ Contexts must be a list")
                return False

            if len(item['contexts']) == 0:
                print(f"❌ Empty contexts list")
                return False

            # Controlla che ogni context sia una stringa non vuota
            for i, ctx in enumerate(item['contexts']):
                if not isinstance(ctx, str) or len(ctx.strip()) < DATASET_CONFIG['min_context_length']:
                    print(f"❌ Context {i} invalid or too short")
                    return False

        return True

    @staticmethod
    def validate_and_fix_dataset(dataset):
        print("\n🔧 DATASET VALIDATION AND CORRECTION:")
        print("=" * 45)

        if not dataset or len(dataset) == 0:
            print("❌ Empty dataset!")
            return None

        sample = dataset[0]
        fixed = False

        # Fix question
        if 'question' not in sample or not sample['question'].strip():
            print("🔧 Fixing question...")
            sample['question'] = "What information does this document provide?"
            fixed = True

        # Fix answer
        if 'answer' not in sample or not sample['answer'].strip():
            print("❌ Missing or empty answer!")
            return None

        # Fix contexts
        if 'contexts' not in sample or not sample['contexts']:
            print("❌ Missing contexts!")
            return None

        # Pulisci e migliora contexts
        clean_contexts = []
        for ctx in sample['contexts']:
            if isinstance(ctx, str) and len(ctx.strip()) >= DATASET_CONFIG['min_context_length']:
                # Assicurati che termini con punteggiatura
                ctx = ctx.strip()
                if not ctx.endswith(('.', '!', '?')):
                    ctx += "."
                clean_contexts.append(ctx)

        if len(clean_contexts) == 0:
            print("❌ No valid contexts after cleaning!")
            return None

        sample['contexts'] = clean_contexts[:DATASET_CONFIG['max_contexts']]

        # Aggiungi ground_truth se mancante
        if 'ground_truth' not in sample or not sample['ground_truth']:
            # Crea ground truth dalla prima frase dell'answer
            first_sentence = sample['answer'].split('.')[0].strip()
            if len(first_sentence) > 10:
                sample['ground_truth'] = first_sentence + "."
            else:
                sample['ground_truth'] = sample['answer'][:100].strip() + "."
            fixed = True

        if fixed:
            print("✅ Dataset automatically corrected")

        # Ricrea dataset con dati corretti
        corrected_dataset = Dataset.from_list([sample])

        # Validazione finale
        if DatasetValidator.validate_dataset(corrected_dataset):
            return corrected_dataset
        else:
            return None
