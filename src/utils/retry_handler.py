class RetryHandler:
    """Class to handle retries for metric evaluations."""
    
    def __init__(self, max_retries=3, backoff_factor=0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def execute_with_retries(self, func, *args, **kwargs):
        """Executes a function with retries on failure."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries exceeded for {func.__name__}. Last error: {e}")
                    raise e  # Re-raise the last exception after max retries

    def validate_and_fix_dataset(self, dataset):
        """Validates and fixes the dataset, retrying if necessary."""
        return self.execute_with_retries(self._validate_and_fix, dataset)

    def _validate_and_fix(self, dataset):
        """Internal method to validate and fix the dataset."""
        # Implement dataset validation and fixing logic here
        pass  # Placeholder for actual implementation