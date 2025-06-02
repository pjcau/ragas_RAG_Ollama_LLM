class DatasetCreator:
    """Classe per la creazione di dataset per test e valutazione."""
    
    @staticmethod
    def create_sample_dataset(num_samples=10):
        """Crea un dataset di esempio per la valutazione."""
        dataset = []
        for i in range(num_samples):
            sample = {
                'question': f"What is the information in sample {i + 1}?",
                'answer': f"This is the answer for sample {i + 1}.",
                'contexts': [
                    f"Context for sample {i + 1} - part 1.",
                    f"Context for sample {i + 1} - part 2.",
                    f"Context for sample {i + 1} - part 3."
                ],
                'ground_truth': f"This is the ground truth for sample {i + 1}."
            }
            dataset.append(sample)
        return dataset

    @staticmethod
    def create_custom_dataset(questions, answers, contexts):
        """Crea un dataset personalizzato basato su domande, risposte e contesti forniti."""
        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError("Le liste di domande, risposte e contesti devono avere la stessa lunghezza.")
        
        dataset = []
        for i in range(len(questions)):
            sample = {
                'question': questions[i],
                'answer': answers[i],
                'contexts': contexts[i],
                'ground_truth': answers[i].split('.')[0] + "."
            }
            dataset.append(sample)
        return dataset

    @staticmethod
    def save_dataset_to_file(dataset, file_path):
        """Salva il dataset in un file JSON."""
        import json
        with open(file_path, 'w') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_dataset_from_file(file_path):
        """Carica un dataset da un file JSON."""
        import json
        with open(file_path, 'r') as f:
            return json.load(f)