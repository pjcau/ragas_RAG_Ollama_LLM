class DatasetValidator:
    """Classe per la validazione e correzione dei dataset utilizzati nel processo di valutazione."""
    
    def __init__(self):
        pass
    
    def validate(self, dataset):
        """Valida il dataset e restituisce True se valido, False altrimenti."""
        if not dataset or len(dataset) == 0:
            print("❌ Dataset vuoto!")
            return False
        
        for sample in dataset:
            if 'question' not in sample or not sample['question'].strip():
                print("❌ Domanda mancante o vuota!")
                return False
            
            if 'answer' not in sample or not sample['answer'].strip():
                print("❌ Risposta mancante o vuota!")
                return False
            
            if 'contexts' not in sample or not sample['contexts']:
                print("❌ Contesti mancanti!")
                return False
            
            if not all(isinstance(ctx, str) and len(ctx.strip()) >= 20 for ctx in sample['contexts']):
                print("❌ Alcuni contesti non validi!")
                return False
        
        print("✅ Dataset valido")
        return True
    
    def fix(self, dataset):
        """Corregge automaticamente il dataset se necessario."""
        print("\n🔧 Correzione dataset:")
        print("=" * 45)
        
        fixed_dataset = []
        
        for sample in dataset:
            fixed = False
            
            if 'question' not in sample or not sample['question'].strip():
                print("🔧 Correggo domanda...")
                sample['question'] = "What information does this document provide?"
                fixed = True
            
            if 'answer' not in sample or not sample['answer'].strip():
                print("❌ Risposta mancante o vuota!")
                continue
            
            if 'contexts' not in sample or not sample['contexts']:
                print("❌ Contesti mancanti!")
                continue
            
            clean_contexts = []
            for ctx in sample['contexts']:
                if isinstance(ctx, str) and len(ctx.strip()) >= 20:
                    ctx = ctx.strip()
                    if not ctx.endswith(('.', '!', '?')):
                        ctx += "."
                    clean_contexts.append(ctx)
            
            if len(clean_contexts) == 0:
                print("❌ Nessun contesto valido dopo pulizia!")
                continue
            
            sample['contexts'] = clean_contexts[:5]  # Max 5 contexts
            
            if 'ground_truth' not in sample or not sample['ground_truth']:
                first_sentence = sample['answer'].split('.')[0].strip()
                if len(first_sentence) > 10:
                    sample['ground_truth'] = first_sentence + "."
                else:
                    sample['ground_truth'] = sample['answer'][:100].strip() + "."
                fixed = True
            
            if fixed:
                print("✅ Dataset corretto automaticamente")
            
            fixed_dataset.append(sample)
        
        return fixed_dataset