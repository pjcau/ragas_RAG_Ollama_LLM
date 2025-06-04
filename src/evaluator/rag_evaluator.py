import time
import numpy as np
from datasets import Dataset
import re
import json

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

from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = ".."  # Inserisci qui la tua chiave API OpenAI


class RAGEvaluator:
    """Classe per la valutazione completa di sistemi RAG"""

    def __init__(self):
        self.eval_llm = None
        self.eval_embeddings = None
        self.working_metrics_cache = None  # Cache per metriche funzionanti

    def create_ultra_compatible_llm(self):
        """Crea un LLM ultra-compatibile con RAGAS"""

        # return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        return ChatOllama(
            model="deepseek-r1",
            temperature=0.0,
            top_p=0.1,
            num_predict=100,
            format="json",
        )

    def create_robust_embeddings(self):
        """Embeddings più robusti"""
        # return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        return OllamaEmbeddings(
            model="mxbai-embed-large"
        )

    def create_test_dataset_complete(self):
        """Crea un dataset di test completo e robusto per testare le metriche"""

        # Dataset di test con esempi diversificati
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

    def validate_dataset(self, dataset):
        """Valida la struttura e il contenuto di un dataset"""

        if not dataset or len(dataset) == 0:
            return False

        required_fields = ['question', 'answer', 'contexts']

        for item in dataset:
            # Controlla campi obbligatori
            for field in required_fields:
                if field not in item:
                    print(f"❌ Campo mancante: {field}")
                    return False

                if not item[field]:
                    print(f"❌ Campo vuoto: {field}")
                    return False

            # Valida specificamente i contexts
            if not isinstance(item['contexts'], list):
                print(f"❌ Contexts deve essere una lista")
                return False

            if len(item['contexts']) == 0:
                print(f"❌ Lista contexts vuota")
                return False

            # Controlla che ogni context sia una stringa non vuota
            for i, ctx in enumerate(item['contexts']):
                if not isinstance(ctx, str) or len(ctx.strip()) < 10:
                    print(f"❌ Context {i} non valido o troppo corto")
                    return False

        return True

    def get_status_emoji(self, score):
        """Restituisce emoji basato sul punteggio"""
        if score >= 0.8:
            return "🟢"  # Eccellente
        elif score >= 0.6:
            return "✅"  # Buono
        elif score >= 0.4:
            return "⚠️"  # Discreto
        else:
            return "❌"  # Scarso

    def validate_and_fix_dataset(self, dataset):
        """Valida e corregge automaticamente il dataset"""
        print("\n🔧 VALIDAZIONE E CORREZIONE DATASET:")
        print("=" * 45)

        if not dataset or len(dataset) == 0:
            print("❌ Dataset vuoto!")
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
            print("❌ Answer mancante o vuoto!")
            return None

        # Fix contexts
        if 'contexts' not in sample or not sample['contexts']:
            print("❌ Contexts mancanti!")
            return None

        # Pulisci e migliora contexts
        clean_contexts = []
        for ctx in sample['contexts']:
            if isinstance(ctx, str) and len(ctx.strip()) >= 20:
                # Assicurati che termini con punteggiatura
                ctx = ctx.strip()
                if not ctx.endswith(('.', '!', '?')):
                    ctx += "."
                clean_contexts.append(ctx)

        if len(clean_contexts) == 0:
            print("❌ Nessun context valido dopo pulizia!")
            return None

        sample['contexts'] = clean_contexts[:5]  # Max 5 contexts

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
            print("✅ Dataset corretto automaticamente")

        # Ricrea dataset con dati corretti
        corrected_dataset = Dataset.from_list([sample])

        # Validazione finale
        if self.validate_dataset(corrected_dataset):
            return corrected_dataset
        else:
            return None

    def test_metric_with_retries(self, metric_name, metric_obj, test_dataset, test_llm, test_embeddings, max_retries=3):
        """Testa una metrica con retry intelligenti"""

        for attempt in range(max_retries):
            try:
                print(
                    f"  🔄 Tentativo {attempt + 1}/{max_retries} per {metric_name}...")

                start_time = time.time()

                result = evaluate(
                    dataset=test_dataset,
                    metrics=[metric_obj],
                    llm=test_llm,
                    embeddings=test_embeddings,
                    raise_exceptions=True
                    # RIMUOVI: max_workers=1  <-- Questo causa l'errore!
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
                        possible_keys.extend(
                            ["semantic_similarity", "similarity"])
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
                        simplified_item['contexts'] = [
                            ctx[:300] + "." for ctx in item['contexts'][:3]]
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
                    time.sleep(1)  # Breve pausa tra tentativi
                    continue

        return {
            'success': False,
            'error': "Max retries exceeded",
            'attempts': max_retries
        }

    def test_individual_metrics_enhanced(self):
        """Versione migliorata del test delle metriche"""
        print("\n🧪 TEST METRICHE INDIVIDUALI AVANZATO:")
        print("=" * 50)

        # LLM e embeddings ultra-compatibili
        test_llm = self.create_ultra_compatible_llm()
        test_embeddings = self.create_robust_embeddings()

        # Dataset di test molto robusto
        test_dataset = self.create_test_dataset_complete()
        test_dataset = self.validate_and_fix_dataset(test_dataset)

        if not test_dataset:
            print("❌ Impossibile creare dataset valido")
            return {}, {}

        # Lista completa di metriche con configurazioni specifiche
        metrics_to_test = [
            ("faithfulness", faithfulness, "Fedeltà al contesto"),
            ("answer_relevancy", answer_relevancy, "Rilevanza risposta"),
            ("context_precision", context_precision, "Precisione contesto"),
            ("context_recall", context_recall, "Richiamo contesto"),
        ]

        # Aggiungi metriche opzionali con controlli specifici
        if ANSWER_CORRECTNESS_AVAILABLE:
            metrics_to_test.append(
                ("answer_correctness", answer_correctness, "Correttezza"))
        if ANSWER_SIMILARITY_AVAILABLE:
            metrics_to_test.append(
                ("answer_similarity", answer_similarity, "Similarità"))
        if CONTEXT_ENTITY_RECALL_AVAILABLE:
            metrics_to_test.append(
                ("context_entity_recall", context_entity_recall, "Entity Recall"))
        if COHERENCE_AVAILABLE:
            metrics_to_test.append(("coherence", coherence, "Coerenza"))
        if FLUENCY_AVAILABLE:
            metrics_to_test.append(("fluency", fluency, "Fluidità"))
        if CONCISENESS_AVAILABLE:
            metrics_to_test.append(("conciseness", conciseness, "Concisione"))

        working_metrics = {}
        failed_metrics = {}

        for metric_name, metric_obj, description in metrics_to_test:
            print(f"\n🎯 Testing {metric_name} ({description})...")

            result = self.test_metric_with_retries(
                metric_name, metric_obj, test_dataset, test_llm, test_embeddings
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
                print(
                    f"  {status} {metric_name}: {result['score']:.4f} ({result['time']:.1f}s, {result['attempts']} attempts)")
            else:
                failed_metrics[metric_name] = result['error']
                print(f"  ❌ {metric_name}: {result['error'][:80]}...")

        # Cache dei risultati
        self.working_metrics_cache = working_metrics

        # Riassunto migliorato
        print(f"\n📊 RIASSUNTO TEST AVANZATO:")
        print(
            f"✅ Metriche funzionanti: {len(working_metrics)}/{len(metrics_to_test)}")

        if working_metrics:
            for name, info in working_metrics.items():
                print(f"  🎯 {name}: {info['score']:.4f} (key: {info['key']})")

        if failed_metrics:
            print(f"\n❌ Metriche fallite: {len(failed_metrics)}")
            for name, error in failed_metrics.items():
                print(f"  💥 {name}: {error[:60]}...")

        return working_metrics, failed_metrics

    def evaluate_all_working_metrics(self, query, answer, contexts, isTest=False):
        """Valuta metriche funzionanti con filtro automatico per production"""
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
        print(f"  Modalità Test: {isTest}")

        # SEMPRE calcola le metriche custom PRIMA
        print("\n🔧 Calcolando metriche custom...")
        custom_results = self.calculate_custom_metrics(
            query, answer, contexts_list)
        print(f"📊 Custom results ottenuti: {len(custom_results)} metriche")

        # Inizializza risultati RAGAS vuoti
        ragas_results = {}

        if isTest:
            # MODALITÀ TEST: Come prima - testa e cachea
            if not self.working_metrics_cache:
                print("🔍 MODALITÀ TEST - Testing metriche RAGAS...")
                working_metrics, failed_metrics = self.test_individual_metrics_enhanced()

                if not working_metrics:
                    print("❌ Nessuna metrica RAGAS funzionante trovata!")
                    print("✅ Ma abbiamo comunque le metriche custom!")
                else:
                    self.working_metrics_cache = working_metrics
                    print(
                        f"✅ {len(working_metrics)} metriche RAGAS funzionanti trovate")
            else:
                working_metrics = self.working_metrics_cache
                print(
                    f"✅ Utilizzando {len(working_metrics)} metriche RAGAS dalla cache")

            # Valuta con cache esistente
            if self.working_metrics_cache and len(self.working_metrics_cache) > 0:
                ragas_results = self._evaluate_cached_metrics(
                    query, answer, contexts_list)
        else:
            # MODALITÀ PRODUCTION: Valuta tutte le metriche e filtra automaticamente
            print("🎯 MODALITÀ PRODUCTION - Valutazione e filtro automatico...")
            ragas_results = self._evaluate_and_filter_metrics(
                query, answer, contexts_list)

        # Statistiche finali
        total_ragas = len(ragas_results)
        total_custom = len(custom_results)

        print(f"\n📈 RISULTATI FINALI:")
        print(f"✅ RAGAS metriche valutate: {total_ragas}")
        print(f"✅ Custom metriche calcolate: {total_custom}")
        print(f"📊 Totale metriche: {total_ragas + total_custom}")

        return {
            'ragas': ragas_results,
            'custom': custom_results
        }

    def _evaluate_cached_metrics(self, query, answer, contexts_list):
        """Valuta usando le metriche in cache (modalità test)"""
        ragas_results = {}

        print("\n📊 Valutando metriche RAGAS dalla cache...")

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
                            score_key = metric_info.get('key', metric_name)
                            if score_key in raw_output:
                                score = raw_output[score_key]
                                if not np.isnan(score):
                                    ragas_results[metric_name] = score
                                    status = self.get_status_emoji(score)
                                    print(
                                        f"  {status} {metric_name}: {score:.4f} ({elapsed:.1f}s)")
                                else:
                                    print(f"  ⚠️ {metric_name}: Score NaN")
                            else:
                                print(
                                    f"  ⚠️ {metric_name}: Chiave '{score_key}' non trovata")
                        else:
                            print(f"  ❌ {metric_name}: Nessun risultato")

                    except Exception as e:
                        print(f"  💥 {metric_name}: Errore - {str(e)[:50]}...")
            else:
                print("❌ Dataset RAGAS non valido")
        except Exception as e:
            print(f"❌ Errore nel setup RAGAS: {e}")

        return ragas_results

    def _evaluate_and_filter_metrics(self, query, answer, contexts_list):
        """Valuta tutte le metriche e filtra automaticamente quelle valide (modalità production)"""
        ragas_results = {}

        print("\n🎯 Valutando e filtrando metriche RAGAS automaticamente...")

        # Lista di tutte le metriche disponibili
        all_metrics = [
            ("faithfulness", faithfulness, "Fedeltà al contesto"),
            ("answer_relevancy", answer_relevancy, "Rilevanza risposta"),
            ("context_precision", context_precision, "Precisione contesto"),
            ("context_recall", context_recall, "Richiamo contesto"),
        ]

        # Aggiungi metriche opzionali
        if ANSWER_CORRECTNESS_AVAILABLE:
            all_metrics.append(
                ("answer_correctness", answer_correctness, "Correttezza"))
        if ANSWER_SIMILARITY_AVAILABLE:
            all_metrics.append(
                ("answer_similarity", answer_similarity, "Similarità"))
        if CONTEXT_ENTITY_RECALL_AVAILABLE:
            all_metrics.append(
                ("context_entity_recall", context_entity_recall, "Entity Recall"))
        if COHERENCE_AVAILABLE:
            all_metrics.append(("coherence", coherence, "Coerenza"))
        if FLUENCY_AVAILABLE:
            all_metrics.append(("fluency", fluency, "Fluidità"))
        if CONCISENESS_AVAILABLE:
            all_metrics.append(("conciseness", conciseness, "Concisione"))

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

            if not dataset:
                print("❌ Dataset RAGAS non valido")
                return ragas_results

            eval_llm = self.create_ultra_compatible_llm()
            eval_embeddings = self.create_robust_embeddings()

            valid_metrics = 0
            failed_metrics = 0

            for metric_name, metric_obj, description in all_metrics:
                print(f"🔍 Testando {metric_name} ({description})...")

                try:
                    start_time = time.time()

                    result = evaluate(
                        dataset=dataset,
                        metrics=[metric_obj],
                        llm=eval_llm,
                        embeddings=eval_embeddings,
                        raise_exceptions=False
                    )

                    elapsed = time.time() - start_time

                    if result.scores and len(result.scores) > 0:
                        raw_output = result.scores[0]

                        # Cerca la chiave del punteggio
                        possible_keys = [
                            metric_name,
                            str(metric_obj),
                            metric_obj.__class__.__name__.lower(),
                            metric_obj.__class__.__name__,
                            f"{metric_name}_score",
                            "score",
                            "value"
                        ]

                        score = None
                        score_key = None

                        for key in possible_keys:
                            if key in raw_output:
                                potential_score = raw_output[key]
                                if not np.isnan(potential_score) and potential_score > 0:
                                    score = potential_score
                                    score_key = key
                                    break

                        if score is not None:
                            ragas_results[metric_name] = score
                            status = self.get_status_emoji(score)
                            print(
                                f"  ✅ {status} {metric_name}: {score:.4f} ({elapsed:.1f}s)")
                            valid_metrics += 1
                        else:
                            print(
                                f"  ❌ {metric_name}: Score <= 0 o NaN - FILTRATO")
                            failed_metrics += 1
                    else:
                        print(
                            f"  ❌ {metric_name}: Nessun risultato - FILTRATO")
                        failed_metrics += 1

                except Exception as e:
                    print(
                        f"  💥 {metric_name}: Errore - {str(e)[:50]}... - FILTRATO")
                    failed_metrics += 1

            print(f"\n📊 RISULTATI FILTRO AUTOMATICO:")
            print(f"  ✅ Metriche valide (score > 0): {valid_metrics}")
            print(f"  ❌ Metriche filtrate: {failed_metrics}")
            print(
                f"  📈 Tasso successo: {valid_metrics/(valid_metrics+failed_metrics)*100:.1f}%")

        except Exception as e:
            print(f"❌ Errore generale nel filtro automatico: {e}")

        return ragas_results

    def get_comprehensive_results_string(self, results):
        """Genera una string completa con tutti i risultati di valutazione e statistiche"""

        if not results or (not results.get('ragas') and not results.get('custom')):
            return "❌ Nessun risultato disponibile per la formattazione"

        output = []
        output.append("=" * 80)
        output.append("📊 REPORT COMPLETO VALUTAZIONE RAG")
        output.append("=" * 80)

        # Sezione RAGAS
        ragas_results = results.get('ragas', {})
        if ragas_results:
            output.append("\n🔍 METRICHE RAGAS:")
            output.append("-" * 40)

            ragas_scores = []
            for metric_name, score in ragas_results.items():
                status = self.get_status_emoji(score)
                output.append(
                    f"  {status} {metric_name.capitalize()}: {score:.4f}")
                ragas_scores.append(score)

            if ragas_scores:
                ragas_avg = np.mean(ragas_scores)
                ragas_std = np.std(ragas_scores)
                ragas_min = min(ragas_scores)
                ragas_max = max(ragas_scores)

                output.append(f"\n📈 STATISTICHE RAGAS:")
                output.append(f"  🎯 Media: {ragas_avg:.4f}")
                output.append(f"  📊 Deviazione std: {ragas_std:.4f}")
                output.append(f"  ⬇️ Minimo: {ragas_min:.4f}")
                output.append(f"  ⬆️ Massimo: {ragas_max:.4f}")
                output.append(f"  📋 Totale metriche: {len(ragas_scores)}")
        else:
            output.append("\n🔍 METRICHE RAGAS:")
            output.append("-" * 40)
            output.append("  ⚠️ Nessuna metrica RAGAS disponibile")

        # Sezione Custom
        custom_results = results.get('custom', {})
        if custom_results:
            output.append("\n🛠️ METRICHE CUSTOM:")
            output.append("-" * 40)

            custom_scores = []
            for metric_name, score in custom_results.items():
                if isinstance(score, (int, float)):
                    status = self.get_status_emoji(score)
                    output.append(
                        f"  {status} {metric_name.replace('_', ' ').title()}: {score:.4f}")
                    custom_scores.append(score)
                else:
                    output.append(
                        f"  📊 {metric_name.replace('_', ' ').title()}: {score}")

            if custom_scores:
                custom_avg = np.mean(custom_scores)
                custom_std = np.std(custom_scores)
                custom_min = min(custom_scores)
                custom_max = max(custom_scores)

                output.append(f"\n📈 STATISTICHE CUSTOM:")
                output.append(f"  🎯 Media: {custom_avg:.4f}")
                output.append(f"  📊 Deviazione std: {custom_std:.4f}")
                output.append(f"  ⬇️ Minimo: {custom_min:.4f}")
                output.append(f"  ⬆️ Massimo: {custom_max:.4f}")
                output.append(f"  📋 Totale metriche: {len(custom_scores)}")
        else:
            output.append("\n🛠️ METRICHE CUSTOM:")
            output.append("-" * 40)
            output.append("  ⚠️ Nessuna metrica custom disponibile")

        # Sezione riassunto globale
        all_scores = []
        ragas_scores = [score for score in ragas_results.values(
        ) if isinstance(score, (int, float))]
        custom_scores = [score for score in custom_results.values(
        ) if isinstance(score, (int, float))]

        all_scores.extend(ragas_scores)
        all_scores.extend(custom_scores)

        if all_scores:
            overall_avg = np.mean(all_scores)
            overall_std = np.std(all_scores)
            overall_min = min(all_scores)
            overall_max = max(all_scores)

            output.append("\n🌟 RIASSUNTO GLOBALE:")
            output.append("-" * 40)
            output.append(f"  🎯 Media generale: {overall_avg:.4f}")
            output.append(f"  📊 Deviazione std generale: {overall_std:.4f}")
            output.append(f"  ⬇️ Punteggio minimo: {overall_min:.4f}")
            output.append(f"  ⬆️ Punteggio massimo: {overall_max:.4f}")
            output.append(f"  📋 Totale metriche valutate: {len(all_scores)}")

            # Interpretazione qualitativa
            if overall_avg >= 0.8:
                interpretation = "🟢 ECCELLENTE - Sistema RAG molto performante"
            elif overall_avg >= 0.6:
                interpretation = "✅ BUONO - Sistema RAG ben funzionante"
            elif overall_avg >= 0.4:
                interpretation = "⚠️ DISCRETO - Sistema RAG con margini di miglioramento"
            else:
                interpretation = "❌ SCARSO - Sistema RAG necessita ottimizzazioni"

            output.append(f"  🏆 Valutazione: {interpretation}")

            # Distribuzione dei punteggi
            excellent_count = sum(1 for score in all_scores if score >= 0.8)
            good_count = sum(1 for score in all_scores if 0.6 <= score < 0.8)
            fair_count = sum(1 for score in all_scores if 0.4 <= score < 0.6)
            poor_count = sum(1 for score in all_scores if score < 0.4)

            output.append(f"\n📊 DISTRIBUZIONE PUNTEGGI:")
            output.append(
                f"  🟢 Eccellenti (≥0.8): {excellent_count}/{len(all_scores)} ({excellent_count/len(all_scores)*100:.1f}%)")
            output.append(
                f"  ✅ Buoni (0.6-0.8): {good_count}/{len(all_scores)} ({good_count/len(all_scores)*100:.1f}%)")
            output.append(
                f"  ⚠️ Discreti (0.4-0.6): {fair_count}/{len(all_scores)} ({fair_count/len(all_scores)*100:.1f}%)")
            output.append(
                f"  ❌ Scarsi (<0.4): {poor_count}/{len(all_scores)} ({poor_count/len(all_scores)*100:.1f}%)")

        output.append("\n" + "=" * 80)
        output.append("🔚 FINE REPORT")
        output.append("=" * 80)

        return "\n".join(output)

    def calculate_custom_metrics(self, query, answer, contexts_list):
        """Placeholder per metriche custom - implementare secondo necessità"""
        # Questo è un placeholder - dovresti implementare le tue metriche custom qui
        return {
            'response_length': len(answer),
            'context_count': len(contexts_list),
            'query_length': len(query)
        }
