class DatasetCreator:

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
        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError(
                "The lists of questions, answers, and contexts must have the same length.")

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
        import json
        with open(file_path, 'w') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_dataset_from_file(file_path):
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
