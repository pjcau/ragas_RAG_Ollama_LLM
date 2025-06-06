import time

# Import condizionali per le dipendenze
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy per isnan
    class MockNumpy:
        @staticmethod
        def isnan(value):
            try:
                return value != value
            except:
                return False
    np = MockNumpy()

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

# RAGAS imports con gestione errori
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAGAS non disponibile: {e}")
    RAGAS_AVAILABLE = False

# Import metriche opzionali
try:
    from ragas.metrics import answer_correctness
    ANSWER_CORRECTNESS_AVAILABLE = True
except ImportError:
    ANSWER_CORRECTNESS_AVAILABLE = False

try:
    from ragas.metrics import answer_similarity
    ANSWER_SIMILARITY_AVAILABLE = True
except ImportError:
    ANSWER_SIMILARITY_AVAILABLE = False

try:
    from ragas.metrics import context_entity_recall
    CONTEXT_ENTITY_RECALL_AVAILABLE = True
except ImportError:
    CONTEXT_ENTITY_RECALL_AVAILABLE = False

try:
    from ragas.metrics import coherence
    COHERENCE_AVAILABLE = True
except ImportError:
    COHERENCE_AVAILABLE = False

try:
    from ragas.metrics import fluency
    FLUENCY_AVAILABLE = True
except ImportError:
    FLUENCY_AVAILABLE = False

try:
    from ragas.metrics import conciseness
    CONCISENESS_AVAILABLE = True
except ImportError:
    CONCISENESS_AVAILABLE = False

from ..models.llm_factory import LLMFactory
from ..config.settings import RETRY_CONFIG
from .dataset_validator import DatasetValidator


class MetricsTester:
    """Classe per il test delle metriche RAGAS"""
    
    def __init__(self):
        self.working_metrics_cache = None
    
    def test_metric_with_retries(self, metric_name, metric_obj, test_dataset, test_llm, test_embeddings, max_retries=None, test_mode=False):
        """Testa una metrica con retry intelligenti"""
        
        if max_retries is None:
            max_retries = RETRY_CONFIG['max_retries']
        if test_mode == False:
            max_retries = 1

        for attempt in range(max_retries):
            try:
                print(f"  🔄 Tentativo {attempt + 1}/{max_retries} per {metric_name}...")
                
                start_time = time.time()
                
                result = evaluate(
                    dataset=test_dataset,
                    metrics=[metric_obj],
                    llm=test_llm,
                    embeddings=test_embeddings,
                    raise_exceptions=True
                )
                
                elapsed = time.time() - start_time
                
                if result.scores and len(result.scores) > 0:
                    # Cerca score con più varianti di nomi
                    score = None
                    score_key = None
                    
                    possible_keys = [
                        metric_name,
                        str(metric_obj),
                        metric_obj.__class__.__name__.lower(),
                        metric_obj.__class__.__name__,
                        f"{metric_name}_score",
                        "score",
                        "value"
                    ]
                    
                    # Per alcune metriche specifiche, aggiungi chiavi note
                    if metric_name == "answer_similarity":
                        possible_keys.extend(["semantic_similarity", "similarity"])
                    elif metric_name == "context_entity_recall":
                        possible_keys.extend(["entity_recall", "recall"])
                    
                    for key in possible_keys:
                        if key in result.scores[0]:
                            score = result.scores[0][key]
                            score_key = key
                            break
                    
                    if score is not None and not np.isnan(score):
                        return {
                            'success': True,
                            'score': score,
                            'time': elapsed,
                            'key': score_key,
                            'attempts': attempt + 1
                        }
                    else:
                        if attempt == max_retries - 1:
                            return {
                                'success': False,
                                'error': f"Score NaN. Available keys: {list(result.scores[0].keys())}",
                                'attempts': attempt + 1
                            }
                        else:
                            print(f"    ⚠️ Score NaN, retry...")
                            continue
                else:
                    if attempt == max_retries - 1:
                        return {
                            'success': False,
                            'error': "No scores returned",
                            'attempts': attempt + 1
                        }
                    else:
                        print(f"    ⚠️ No scores, retry...")
                        continue
                        
            except Exception as e:
                error_msg = str(e)
                
                # Se è un errore di parsing, prova con dataset semplificato
                if "output parser" in error_msg.lower() and attempt < max_retries - 1:
                    print(f"    🔧 Parser error, simplifying dataset...")
                    # Semplifica answer e contexts per il prossimo tentativo
                    simplified_data = []
                    for item in test_dataset:
                        simplified_item = item.copy()
                        # Accorcia answer
                        simplified_item['answer'] = item['answer'][:200] + "."
                        # Accorcia contexts
                        simplified_item['contexts'] = [ctx[:300] + "." for ctx in item['contexts'][:3]]
                        simplified_data.append(simplified_item)
                    
                    test_dataset = Dataset.from_list(simplified_data)
                    continue
                elif attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': error_msg[:150],
                        'attempts': attempt + 1
                    }
                else:
                    print(f"    ⚠️ Error: {error_msg[:50]}..., retry...")
                    time.sleep(RETRY_CONFIG['retry_delay'])
                    continue
        
        return {
            'success': False,
            'error': "Max retries exceeded",
            'attempts': max_retries
        }
    
    def test_individual_metrics_enhanced(self, test_mode=False):
        """Versione migliorata del test delle metriche"""
        print("\n🧪 TEST METRICHE INDIVIDUALI AVANZATO:")
        print("=" * 50)
        
        # LLM e embeddings ultra-compatibili
        test_llm = LLMFactory.create_ultra_compatible_llm()
        test_embeddings = LLMFactory.create_robust_embeddings()
        
        # Dataset di test molto robusto
        test_dataset = DatasetValidator.create_test_dataset_complete()
        test_dataset = DatasetValidator.validate_and_fix_dataset(test_dataset)
        
        if not test_dataset:
            print("❌ Impossibile creare dataset valido")
            return {}, {}
        
        # Lista completa di metriche con configurazioni specifiche
        metrics_to_test = []
        
        # Aggiungi solo metriche disponibili
        if RAGAS_AVAILABLE:
            if faithfulness is not None:
                metrics_to_test.append(("faithfulness", faithfulness, "Fedeltà al contesto"))
            if answer_relevancy is not None:
                metrics_to_test.append(("answer_relevancy", answer_relevancy, "Rilevanza risposta"))
            if context_precision is not None:
                metrics_to_test.append(("context_precision", context_precision, "Precisione contesto"))
            if context_recall is not None:
                metrics_to_test.append(("context_recall", context_recall, "Richiamo contesto"))
        
        # Aggiungi metriche opzionali con controlli specifici
        if ANSWER_CORRECTNESS_AVAILABLE and answer_correctness is not None:
            metrics_to_test.append(("answer_correctness", answer_correctness, "Correttezza"))
        if ANSWER_SIMILARITY_AVAILABLE and answer_similarity is not None:
            metrics_to_test.append(("answer_similarity", answer_similarity, "Similarità"))
        if CONTEXT_ENTITY_RECALL_AVAILABLE and context_entity_recall is not None:
            metrics_to_test.append(("context_entity_recall", context_entity_recall, "Entity Recall"))
        if COHERENCE_AVAILABLE and coherence is not None:
            metrics_to_test.append(("coherence", coherence, "Coerenza"))
        if FLUENCY_AVAILABLE and fluency is not None:
            metrics_to_test.append(("fluency", fluency, "Fluidità"))
        if CONCISENESS_AVAILABLE and conciseness is not None:
            metrics_to_test.append(("conciseness", conciseness, "Concisione"))
        
        if not metrics_to_test:
            print("❌ Nessuna metrica RAGAS disponibile")
            return {}, {}
        
        working_metrics = {}
        failed_metrics = {}
        
        for metric_name, metric_obj, description in metrics_to_test:
            print(f"\n🎯 Testing {metric_name} ({description})...")
            
            result = self.test_metric_with_retries(
                metric_name, metric_obj, test_dataset, test_llm, test_embeddings, test_mode
            )
            
            if result['success']:
                working_metrics[metric_name] = {
                    'metric': metric_obj,
                    'score': result['score'],
                    'time': result['time'],
                    'key': result['key'],
                    'attempts': result['attempts']
                }
                status = "🟢" if result['score'] > 0.7 else "✅" if result['score'] > 0.4 else "⚠️"
                print(f"  {status} {metric_name}: {result['score']:.4f} ({result['time']:.1f}s, {result['attempts']} attempts)")
            else:
                failed_metrics[metric_name] = result['error']
                print(f"  ❌ {metric_name}: {result['error'][:80]}...")
        
        # Cache dei risultati
        self.working_metrics_cache = working_metrics
        
        # Riassunto migliorato
        print(f"\n📊 RIASSUNTO TEST AVANZATO:")
        print(f"✅ Metriche funzionanti: {len(working_metrics)}/{len(metrics_to_test)}")
        
        if working_metrics:
            for name, info in working_metrics.items():
                print(f"  🎯 {name}: {info['score']:.4f} (key: {info['key']})")
        
        if failed_metrics:
            print(f"\n❌ Metriche fallite: {len(failed_metrics)}")
            for name, error in failed_metrics.items():
                print(f"  💥 {name}: {error[:60]}...")
        
        return working_metrics, failed_metrics
    
    def get_available_metrics_quick(self):
        """Ottiene tutte le metriche RAGAS disponibili senza test approfonditi"""
        print("🔍 Caricamento veloce metriche RAGAS disponibili...")
        
        if not RAGAS_AVAILABLE:
            print("❌ RAGAS non disponibile")
            return {}
        
        available_metrics = {}
        
        # Lista completa di metriche da controllare
        metrics_to_check = [
            # Metriche principali sempre disponibili
            ("faithfulness", faithfulness, "Fedeltà al contesto"),
            ("answer_relevancy", answer_relevancy, "Rilevanza della risposta"),
            ("context_precision", context_precision, "Precisione del contesto"),
            ("context_recall", context_recall, "Richiamo del contesto"),
        ]
        
        # Metriche opzionali con controlli di disponibilità
        if ANSWER_CORRECTNESS_AVAILABLE:
            try:
                from ragas.metrics import answer_correctness
                metrics_to_check.append(("answer_correctness", answer_correctness, "Correttezza della risposta"))
            except ImportError:
                pass
        
        if ANSWER_SIMILARITY_AVAILABLE:
            try:
                from ragas.metrics import answer_similarity
                metrics_to_check.append(("answer_similarity", answer_similarity, "Similarità della risposta"))
            except ImportError:
                pass
        
        if CONTEXT_ENTITY_RECALL_AVAILABLE:
            try:
                from ragas.metrics import context_entity_recall
                metrics_to_check.append(("context_entity_recall", context_entity_recall, "Richiamo entità del contesto"))
            except ImportError:
                pass
        
        if COHERENCE_AVAILABLE:
            try:
                from ragas.metrics import coherence
                metrics_to_check.append(("coherence", coherence, "Coerenza della risposta"))
            except ImportError:
                pass
        
        if FLUENCY_AVAILABLE:
            try:
                from ragas.metrics import fluency
                metrics_to_check.append(("fluency", fluency, "Fluidità della risposta"))
            except ImportError:
                pass
        
        if CONCISENESS_AVAILABLE:
            try:
                from ragas.metrics import conciseness
                metrics_to_check.append(("conciseness", conciseness, "Concisione della risposta"))
            except ImportError:
                pass
        
        # Metriche aggiuntive se disponibili
        try:
            from ragas.metrics import context_relevancy
            metrics_to_check.append(("context_relevancy", context_relevancy, "Rilevanza del contesto"))
        except ImportError:
            pass
        
        try:
            from ragas.metrics import summarization_score
            metrics_to_check.append(("summarization_score", summarization_score, "Score di riassunto"))
        except ImportError:
            pass
        
        try:
            from ragas.metrics import aspect_critique
            metrics_to_check.append(("aspect_critique", aspect_critique, "Critica per aspetti"))
        except ImportError:
            pass
        
        try:
            from ragas.metrics import maliciousness
            metrics_to_check.append(("maliciousness", maliciousness, "Rilevamento malizia"))
        except ImportError:
            pass
        
        try:
            from ragas.metrics import harmfulness
            metrics_to_check.append(("harmfulness", harmfulness, "Rilevamento dannosità"))
        except ImportError:
            pass
        
        # Verifica disponibilità di ogni metrica
        for metric_name, metric_obj, description in metrics_to_check:
            try:
                if metric_obj is not None:
                    available_metrics[metric_name] = {
                        'metric': metric_obj,
                        'key': metric_name,
                        'description': description
                    }
                    print(f"  ✅ {metric_name}: {description}")
                else:
                    print(f"  ⚠️ {metric_name}: oggetto nullo")
            except Exception as e:
                print(f"  ❌ {metric_name}: errore - {str(e)[:50]}")
        
        print(f"📋 Caricate {len(available_metrics)} metriche RAGAS disponibili")
        return available_metrics