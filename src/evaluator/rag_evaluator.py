import time

# Import condizionali per numpy
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

# Import condizionali per datasets
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
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAGAS non disponibile: {e}")
    RAGAS_AVAILABLE = False

# Import dei moduli refactorizzati con gestione errori
try:
    from .metrics_tester import MetricsTester
    METRICS_TESTER_AVAILABLE = True
except ImportError:
    METRICS_TESTER_AVAILABLE = False
    MetricsTester = None

try:
    from .custom_metrics import CustomMetrics
    CUSTOM_METRICS_AVAILABLE = True
except ImportError:
    CUSTOM_METRICS_AVAILABLE = False
    CustomMetrics = None

try:
    from .dataset_validator import DatasetValidator
    DATASET_VALIDATOR_AVAILABLE = True
except ImportError:
    DATASET_VALIDATOR_AVAILABLE = False
    DatasetValidator = None

try:
    from ..models.llm_factory import LLMFactory
    LLM_FACTORY_AVAILABLE = True
except ImportError:
    LLM_FACTORY_AVAILABLE = False
    LLMFactory = None

try:
    from ..utils.display_helper import DisplayHelper
    DISPLAY_HELPER_AVAILABLE = True
except ImportError:
    DISPLAY_HELPER_AVAILABLE = False
    DisplayHelper = None

class RAGEvaluator:
    """Classe principale per la valutazione completa di sistemi RAG"""

    def __init__(self):
        self.eval_llm = None
        self.eval_embeddings = None
        self.working_metrics_cache = None  # Cache per metriche funzionanti
        
        # Inizializza il metrics_tester solo se disponibile
        if METRICS_TESTER_AVAILABLE and MetricsTester is not None:
            self.metrics_tester = MetricsTester()
        else:
            print("⚠️ MetricsTester non disponibile")
            self.metrics_tester = None

    def create_ultra_compatible_llm(self):
        """Crea un LLM ultra-compatibile con RAGAS"""
        if LLM_FACTORY_AVAILABLE and LLMFactory is not None:
            return LLMFactory.create_ultra_compatible_llm()
        else:
            raise ImportError("LLMFactory non disponibile")

    def create_robust_embeddings(self):
        """Embeddings più robusti"""
        if LLM_FACTORY_AVAILABLE and LLMFactory is not None:
            return LLMFactory.create_robust_embeddings()
        else:
            raise ImportError("LLMFactory non disponibile")

    def create_test_dataset_complete(self):
        """Crea un dataset di test completo e robusto per testare le metriche"""
        if DATASET_VALIDATOR_AVAILABLE and DatasetValidator is not None:
            return DatasetValidator.create_test_dataset_complete()
        else:
            raise ImportError("DatasetValidator non disponibile")

    def validate_dataset(self, dataset):
        """Valida la struttura e il contenuto di un dataset"""
        if DATASET_VALIDATOR_AVAILABLE and DatasetValidator is not None:
            return DatasetValidator.validate_dataset(dataset)
        else:
            return False

    def get_status_emoji(self, score):
        """Restituisce emoji basato sul punteggio"""
        if DISPLAY_HELPER_AVAILABLE and DisplayHelper is not None:
            return DisplayHelper.get_status_emoji(score)
        else:
            # Fallback semplice
            if score >= 0.8:
                return "🟢"
            elif score >= 0.6:
                return "✅"
            elif score >= 0.4:
                return "⚠️"
            else:
                return "❌"

    def validate_and_fix_dataset(self, dataset):
        """Valida e corregge automaticamente il dataset"""
        if DATASET_VALIDATOR_AVAILABLE and DatasetValidator is not None:
            return DatasetValidator.validate_and_fix_dataset(dataset)
        else:
            print("⚠️ DatasetValidator non disponibile, usando fallback")
            return dataset

    def test_metric_with_retries(self, metric_name, metric_obj, test_dataset, test_llm, test_embeddings, max_retries=3):
        """Testa una metrica con retry intelligenti"""
        if self.metrics_tester is not None:
            return self.metrics_tester.test_metric_with_retries(
                metric_name, metric_obj, test_dataset, test_llm, test_embeddings, max_retries
            )
        else:
            return {'success': False, 'error': 'MetricsTester non disponibile'}

    def test_individual_metrics_enhanced(self, test_mode=False):
        """Versione migliorata del test delle metriche"""
        if self.metrics_tester is not None:
            working_metrics, failed_metrics = self.metrics_tester.test_individual_metrics_enhanced(test_mode)
            # Cache dei risultati
            self.working_metrics_cache = working_metrics
            return working_metrics, failed_metrics
        else:
            print("⚠️ MetricsTester non disponibile")
            return {}, {}

    def evaluate_all_working_metrics(self, query, answer, contexts, test_mode=False):
        """Valuta metriche funzionanti con accesso diretto ai valori
        
        Args:
            query: Query di ricerca
            answer: Risposta generata
            contexts: Contesti recuperati
            test_mode: Se True, ottimizza per ottenere working_metrics_cache velocemente
        """
        print("\n🚀 VALUTAZIONE CON METRICHE FUNZIONANTI:")
        print("=" * 60)

        # Prepara il dataset PRIMA di testare le metriche
        if isinstance(contexts, list) and all(isinstance(ctx, str) for ctx in contexts):
            contexts_list = contexts
        else:
            contexts_list = []
            for ctx in contexts:
                if hasattr(ctx, 'page_content'):
                    contexts_list.append(ctx.page_content)
                elif isinstance(ctx, str):
                    contexts_list.append(ctx)
                else:
                    contexts_list.append(str(ctx))

        print(f"📋 Input preparato:")
        print(f"  Query: {query[:50]}...")
        print(f"  Answer: {answer[:50]}...")
        print(f"  Contexts: {len(contexts_list)} items")
        print(f"  Test Mode: {test_mode}")

        # SEMPRE calcola le metriche custom PRIMA - non dipendono da RAGAS
        print("\n🔧 Calcolando metriche custom...")
        if CUSTOM_METRICS_AVAILABLE and CustomMetrics is not None:
            custom_results = CustomMetrics.calculate_custom_metrics(query, answer, contexts_list)
        else:
            print("⚠️ CustomMetrics non disponibile, usando fallback")
            custom_results = self._calculate_simple_custom_metrics(query, answer, contexts_list)
        print(f"📊 Custom results ottenuti: {len(custom_results)} metriche")

        # Inizializza risultati RAGAS vuoti
        ragas_results = {}

        # Gestione test_mode per ottimizzazione
        if test_mode:
            print("🧪 TEST MODE attivo - ottengo working_metrics_cache...")
            if not self.working_metrics_cache:
                working_metrics, failed_metrics = self.test_individual_metrics_enhanced()
                self.working_metrics_cache = working_metrics
                print(f"✅ Cache popolata con {len(working_metrics)} metriche funzionanti")
            else:
                print(f"✅ Cache già presente con {len(self.working_metrics_cache)} metriche")
            working_metrics = self.working_metrics_cache
        else:
            # Modalità normale - esegui test solo se necessario
            if not self.working_metrics_cache:
                print("🔍 Prima esecuzione - testing metriche RAGAS...")
                working_metrics, failed_metrics = self.test_individual_metrics_enhanced(test_mode)

                if not working_metrics:
                    print("❌ Nessuna metrica RAGAS funzionante trovata!")
                    print("✅ Ma abbiamo comunque le metriche custom!")
                else:
                    self.working_metrics_cache = working_metrics
                    print(f"✅ {len(working_metrics)} metriche RAGAS funzionanti trovate")
            else:
                working_metrics = self.working_metrics_cache
                print(f"✅ Utilizzando {len(working_metrics)} metriche RAGAS dalla cache")

        # Valuta metriche RAGAS solo se abbiamo metriche funzionanti
        if self.working_metrics_cache and len(self.working_metrics_cache) > 0:
            print("\n📊 Valutando metriche RAGAS...")

            # Crea dataset per RAGAS
            dataset_data = [{
                'question': query,
                'answer': answer,
                'contexts': contexts_list[:5],
                'ground_truth': answer
            }]

            try:
                dataset = Dataset.from_list(dataset_data)
                dataset = self.validate_and_fix_dataset(dataset)

                if dataset:
                    eval_llm = self.create_ultra_compatible_llm()
                    eval_embeddings = self.create_robust_embeddings()

                    for metric_name, metric_info in self.working_metrics_cache.items():
                        print(f"📊 Valutando {metric_name}...")

                        try:
                            start_time = time.time()

                            result = evaluate(
                                dataset=dataset,
                                metrics=[metric_info['metric']],
                                llm=eval_llm,
                                embeddings=eval_embeddings,
                                raise_exceptions=False
                            )

                            elapsed = time.time() - start_time

                            if result.scores and len(result.scores) > 0:
                                raw_output = result.scores[0]
                                print(f"  Raw Output: {raw_output}")  # DEBUG

                                score_key = metric_info.get('key', metric_name)
                                if score_key in raw_output:
                                    score = raw_output[score_key]
                                    
                                    # Esclusione automatica di score zero o NaN
                                    if np.isnan(score):
                                        print(f"  ⚠️ {metric_name}: Score NaN - ESCLUSO dal computo finale")
                                        continue
                                    elif score == 0:
                                        print(f"  ⚠️ {metric_name}: Score zero - ESCLUSO dal computo finale")
                                        continue
                                    else:
                                        ragas_results[metric_name] = score
                                        status = self.get_status_emoji(score)
                                        print(f"  {status} {metric_name}: {score:.4f} ({elapsed:.1f}s) - INCLUSO")
                                else:
                                    print(f"  ⚠️ {metric_name}: Chiave '{score_key}' non trovata - ESCLUSO")
                            else:
                                print(f"  ❌ {metric_name}: Nessun risultato - ESCLUSO")

                        except Exception as e:
                            print(f"  💥 {metric_name}: Errore - {str(e)[:50]}... - ESCLUSO")
                            # Rimuovi traceback per non intasare l'output
                else:
                    print("❌ Dataset RAGAS non valido")
            except Exception as e:
                print(f"❌ Errore nel setup RAGAS: {e}")
        else:
            print("⚠️ Nessuna metrica RAGAS da valutare, usando solo custom metrics")

        # Statistiche finali
        total_ragas = len(ragas_results)
        total_custom = len(custom_results)

        print(f"\n📈 RISULTATI FINALI:")
        print(f"✅ RAGAS metriche valutate: {total_ragas}")
        print(f"✅ Custom metriche calcolate: {total_custom}")
        print(f"📊 Totale metriche: {total_ragas + total_custom}")

        # DEBUG finale
        print(f"\n🔍 DEBUG FINALE:")
        print(f"  ragas_results: {ragas_results}")
        print(f"  custom_results: {custom_results}")

        return {
            'ragas': ragas_results,
            'custom': custom_results
        }

    def calculate_custom_metrics(self, query, answer, contexts):
        """Calcola metriche custom affidabili"""
        if CUSTOM_METRICS_AVAILABLE and CustomMetrics is not None:
            return CustomMetrics.calculate_custom_metrics(query, answer, contexts)
        else:
            return self._calculate_simple_custom_metrics(query, answer, contexts)

    def display_comprehensive_results(self, ragas_results, custom_results):
        """Visualizza risultati con debug esteso"""
        if DISPLAY_HELPER_AVAILABLE and DisplayHelper is not None:
            DisplayHelper.display_comprehensive_results(ragas_results, custom_results)
        else:
            self._simple_display_results(ragas_results, custom_results)

    def evaluate_complete(self, query, answer, contexts, test_mode=False):
        """Metodo di compatibilità per valutazione completa"""
        return self.evaluate_all_working_metrics(query, answer, contexts, test_mode)

    def test_custom_metrics_only(self, query, answer, contexts):
        """Test veloce solo delle metriche custom"""
        print("\n🧪 TEST SOLO METRICHE CUSTOM:")
        print("=" * 40)

        # Prepara contexts
        if isinstance(contexts, list) and all(isinstance(ctx, str) for ctx in contexts):
            contexts_list = contexts
        else:
            contexts_list = []
            for ctx in contexts:
                if hasattr(ctx, 'page_content'):
                    contexts_list.append(ctx.page_content)
                elif isinstance(ctx, str):
                    contexts_list.append(ctx)
                else:
                    contexts_list.append(str(ctx))

        if CUSTOM_METRICS_AVAILABLE and CustomMetrics is not None:
            custom_results = CustomMetrics.calculate_custom_metrics(query, answer, contexts_list)
        else:
            custom_results = self._calculate_simple_custom_metrics(query, answer, contexts_list)

        return {'ragas': {}, 'custom': custom_results}

    def _calculate_simple_custom_metrics(self, query, answer, contexts):
        """Calcolo semplificato delle metriche custom senza dipendenze esterne"""
        if not query or not answer or not contexts:
            return {}
        
        # Calcoli base senza numpy
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Context coverage semplice
        combined_context = " ".join(str(ctx) for ctx in contexts)
        context_words = set(combined_context.lower().split())
        
        custom_results = {}
        
        try:
            # Jaccard Similarity
            if query_words and answer_words:
                intersection = len(query_words.intersection(answer_words))
                union = len(query_words.union(answer_words))
                custom_results['jaccard_similarity'] = round(intersection / union if union > 0 else 0, 4)
            
            # Context Coverage
            if answer_words and context_words:
                answer_in_context = len(answer_words.intersection(context_words))
                custom_results['context_coverage'] = round(answer_in_context / len(answer_words) if answer_words else 0, 4)
            
            # Answer Length Score
            answer_length = len(answer.split())
            custom_results['answer_length_score'] = round(min(answer_length / 150, 1.0), 4)
            
        except Exception as e:
            print(f"⚠️ Errore nel calcolo metriche fallback: {e}")
        
        return custom_results

    def _simple_display_results(self, ragas_results, custom_results):
        """Visualizzazione semplificata dei risultati"""
        print("\n" + "="*60)
        print("📊 RISULTATI VALUTAZIONE RAG")
        print("="*60)
        
        if ragas_results:
            print("\n🎯 RAGAS Metrics:")
            for metric, value in ragas_results.items():
                status = self.get_status_emoji(value)
                print(f"  {status} {metric:20}: {value:.4f}")
        
        if custom_results:
            print("\n🔧 Custom Metrics:")
            for metric, value in custom_results.items():
                status = self.get_status_emoji(value)
                print(f"  {status} {metric:20}: {value:.4f}")
        
        # Statistiche semplici
        all_scores = list(ragas_results.values()) + list(custom_results.values())
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"\n📈 Score medio: {avg_score:.4f}")
            
            if avg_score >= 0.8:
                rating = "🏆 ECCELLENTE"
            elif avg_score >= 0.6:
                rating = "✅ BUONO"
            elif avg_score >= 0.4:
                rating = "⚠️ DISCRETO"
            else:
                rating = "❌ NECESSITA MIGLIORAMENTI"
            
            print(f"🎖️ Rating: {rating}")
        
        print("="*60)
