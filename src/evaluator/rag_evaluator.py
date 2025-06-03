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

    def evaluate_all_working_metrics(self, query, answer, contexts):
        """Valuta metriche funzionanti con accesso diretto ai valori"""
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

        # SEMPRE calcola le metriche custom PRIMA - non dipendono da RAGAS
        print("\n🔧 Calcolando metriche custom...")
        custom_results = self.calculate_custom_metrics(
            query, answer, contexts_list)
        print(f"📊 Custom results ottenuti: {len(custom_results)} metriche")

        # Inizializza risultati RAGAS vuoti
        ragas_results = {}

        # Ora testa le metriche RAGAS solo se necessario
        if not self.working_metrics_cache:
            print("🔍 Prima esecuzione - testing metriche RAGAS...")
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
                                print(f"  Raw Output: {raw_output}")  #
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
                            print(
                                f"  💥 {metric_name}: Errore - {str(e)[:50]}...")
                            import traceback
                            traceback.print_exc()  # Stampa stack trace completo
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
        """Calcola metriche custom affidabili con debug migliorato"""

        print(f"🔧 DEBUG Custom Metrics:")
        print(f"  Query: {query[:50] if query else 'None'}...")
        print(f"  Answer: {answer[:50] if answer else 'None'}...")
        print(f"  Contexts: {len(contexts) if contexts else 0} items")

        if not query or not answer:
            print("❌ Query o Answer mancanti")
            return {}

        if not contexts or len(contexts) == 0:
            print("❌ Contexts mancanti")
            return {}

        # Prepara testi per analisi
        query_lower = query.lower().strip()
        answer_lower = answer.lower().strip()

        query_words = set([w.strip()
                          for w in query_lower.split() if w.strip()])
        answer_words = set([w.strip()
                           for w in answer_lower.split() if w.strip()])

        print(f"  Query words: {len(query_words)}")
        print(f"  Answer words: {len(answer_words)}")

        # Combina tutti i contexts
        combined_context = ""
        for ctx in contexts:
            if isinstance(ctx, str):
                combined_context += " " + ctx
            elif hasattr(ctx, 'page_content'):
                combined_context += " " + ctx.page_content
            else:
                combined_context += " " + str(ctx)

        context_words = set(
            [w.strip() for w in combined_context.lower().split() if w.strip()])
        print(f"  Context words: {len(context_words)}")

        custom_results = {}

        try:
            # 1. Jaccard Similarity (Query-Answer)
            if query_words and answer_words:
                intersection = len(query_words.intersection(answer_words))
                union = len(query_words.union(answer_words))
                jaccard_sim = intersection / union if union > 0 else 0
                custom_results['jaccard_similarity'] = round(jaccard_sim, 4)
                print(f"  ✅ Jaccard similarity: {jaccard_sim:.4f}")
            else:
                print("  ⚠️ Jaccard: query_words o answer_words vuoti")

            # 2. Context Coverage (Answer utilizza Context)
            if answer_words and context_words:
                answer_in_context = len(
                    answer_words.intersection(context_words))
                context_coverage = answer_in_context / \
                    len(answer_words) if answer_words else 0
                custom_results['context_coverage'] = round(context_coverage, 4)
                print(f"  ✅ Context coverage: {context_coverage:.4f}")
            else:
                print("  ⚠️ Context coverage: answer_words o context_words vuoti")

            # 3. Information Density
            if answer and answer.strip():
                words = answer.split()
                unique_words = set([w.lower().strip()
                                   for w in words if w.strip()])
                info_density = len(unique_words) / len(words) if words else 0
                custom_results['information_density'] = round(info_density, 4)
                print(f"  ✅ Information density: {info_density:.4f}")
            else:
                print("  ⚠️ Information density: answer vuoto")

            # 4. Query Coverage (Answer copre Query)
            if query_words and answer_words:
                query_covered = len(query_words.intersection(answer_words))
                query_coverage = query_covered / \
                    len(query_words) if query_words else 0
                custom_results['query_coverage'] = round(query_coverage, 4)
                print(f"  ✅ Query coverage: {query_coverage:.4f}")
            else:
                print("  ⚠️ Query coverage: query_words o answer_words vuoti")

            # 5. Answer Length Score (normalizzato)
            if answer and answer.strip():
                answer_length = len(answer.split())
                optimal_length = 150  # Lunghezza ottimale
                length_score = min(answer_length / optimal_length,
                                   1.0) if optimal_length > 0 else 0
                custom_results['answer_length_score'] = round(length_score, 4)
                print(
                    f"  ✅ Answer length score: {length_score:.4f} (length: {answer_length})")
            else:
                print("  ⚠️ Answer length: answer vuoto")

            # 6. Technical Terms Density
            if answer and answer.strip():
                words = answer.split()
                # Parole "tecniche"
                long_words = [w for w in words if len(w) > 6]
                tech_density = len(long_words) / len(words) if words else 0
                custom_results['technical_density'] = round(tech_density, 4)
                print(f"  ✅ Technical density: {tech_density:.4f}")
            else:
                print("  ⚠️ Technical density: answer vuoto")

            # 7. Context Relevance (Query vs Context)
            if query_words and context_words:
                query_in_context = len(query_words.intersection(context_words))
                context_relevance = query_in_context / \
                    len(query_words) if query_words else 0
                custom_results['context_relevance'] = round(
                    context_relevance, 4)
                print(f"  ✅ Context relevance: {context_relevance:.4f}")
            else:
                print("  ⚠️ Context relevance: query_words o context_words vuoti")

            # 8. Answer Completeness
            if answer and query:
                # Semplice euristica: risposta più lunga = più completa
                answer_completeness = min(
                    len(answer) / 200, 1.0)  # Cap a 200 caratteri
                custom_results['answer_completeness'] = round(
                    answer_completeness, 4)
                print(f"  ✅ Answer completeness: {answer_completeness:.4f}")

        except Exception as e:
            print(f"❌ Errore nel calcolo metriche custom: {e}")
            import traceback
            traceback.print_exc()

        print(f"📊 Metriche custom calcolate: {len(custom_results)}")
        for name, value in custom_results.items():
            print(f"  • {name}: {value}")

        return custom_results

    def get_comprehensive_results_string(self, ragas_results, custom_results):
        """Restituisce una stringa formattata con i risultati della valutazione RAG"""

        result_lines = []
        result_lines.append("=" * 80)
        result_lines.append("📊 RISULTATI VALUTAZIONE RAG COMPLETA")
        result_lines.append("=" * 80)

        # DEBUG: Verifica input
        result_lines.append(f"🔍 DEBUG display_comprehensive_results:")
        result_lines.append(f"  RAGAS results type: {type(ragas_results)}")
        result_lines.append(f"  RAGAS results: {ragas_results}")
        result_lines.append(f"  Custom results type: {type(custom_results)}")
        result_lines.append(f"  Custom results: {custom_results}")
        result_lines.append(
            f"  Custom results length: {len(custom_results) if custom_results else 0}")

        # Aggiungi questo blocco per debug RAGAS
        if ragas_results:
            for metric, value in ragas_results.items():
                result_lines.append(
                    f"  RAGAS Metric: {metric}, Value: {value}, Type: {type(value)}")
        else:
            result_lines.append("  Nessun risultato RAGAS")

        # Aggiungi questo blocco per debug Custom
        if custom_results:
            for metric, value in custom_results.items():
                result_lines.append(
                    f"  Custom Metric: {metric}, Value: {value}, Type: {type(value)}")
        else:
            result_lines.append("  Nessun risultato Custom")

        # Categorizza metriche RAGAS
        metric_categories = {
            "🎯 Core RAG Metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            "📝 Answer Quality": ["answer_correctness", "answer_similarity"],
            "📄 Context Analysis": ["context_entity_recall"],
            "🗣️ Language Quality": ["coherence", "fluency", "conciseness"],
            "🛡️ Safety & Ethics": ["harmfulness", "maliciousness"],
            "🔬 Specialized": ["summarization_score", "aspect_critique"]
        }

        # Mostra metriche RAGAS per categoria
        ragas_scores = []
        zero_scores = []

        result_lines.append(f"\n📊 SEZIONE RAGAS METRICS:")
        if ragas_results and len(ragas_results) > 0:
            for category_name, metrics in metric_categories.items():
                category_metrics = {k: v for k,
                                    v in ragas_results.items() if k in metrics}
                if category_metrics:
                    result_lines.append(f"\n{category_name}:")
                    for metric, value in category_metrics.items():
                        if not np.isnan(value):
                            status = self.get_status_emoji(value)
                            result_lines.append(
                                f"  {status} {metric:20}: {value:.4f}")
                            ragas_scores.append(value)
                            if value == 0:
                                zero_scores.append(metric)
                        else:
                            result_lines.append(f"  🔶 {metric:20}: NaN")
        else:
            result_lines.append("  ❌ Nessuna metrica RAGAS disponibile")

        # Mostra metriche custom con debug esteso
        result_lines.append(f"\n🔧 SEZIONE CUSTOM METRICS:")
        result_lines.append(f"📋 Debug custom_results:")
        result_lines.append(f"  Type: {type(custom_results)}")
        result_lines.append(f"  Content: {custom_results}")
        result_lines.append(f"  Is dict: {isinstance(custom_results, dict)}")
        result_lines.append(
            f"  Length: {len(custom_results) if custom_results else 0}")

        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            result_lines.append(
                f"✅ Mostrando {len(custom_results)} metriche custom:")
            for metric, value in custom_results.items():
                result_lines.append(
                    f"  Processing: {metric} = {value} (type: {type(value)})")
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        status = self.get_status_emoji(value)
                        result_lines.append(
                            f"  {status} {metric:25}: {value:.4f}")
                        custom_scores.append(value)
                    else:
                        result_lines.append(
                            f"  ⚠️ {metric:25}: {value} (invalid)")
                except Exception as e:
                    result_lines.append(
                        f"  ❌ Errore processando {metric}: {e}")
        else:
            result_lines.append("  ❌ Nessuna metrica custom disponibile")
            result_lines.append(f"     custom_results: {custom_results}")

        # Statistiche globali
        all_scores = ragas_scores + custom_scores

        result_lines.append(f"\n📈 STATISTICHE GLOBALI:")
        result_lines.append(f"  📊 RAGAS scores: {len(ragas_scores)}")
        result_lines.append(f"  🔧 Custom scores: {len(custom_scores)}")
        result_lines.append(f"  📋 Total scores: {len(all_scores)}")

        if all_scores:
            avg_score = np.mean(all_scores)
            median_score = np.median(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)

            result_lines.append(f"  📊 Score medio:      {avg_score:.4f}")
            result_lines.append(f"  🎯 Score mediano:    {median_score:.4f}")
            result_lines.append(f"  🏆 Score massimo:    {max_score:.4f}")
            result_lines.append(f"  ⚠️ Score minimo:     {min_score:.4f}")

            # Rating globale
            if avg_score >= 0.8:
                rating = "🏆 ECCELLENTE"
            elif avg_score >= 0.6:
                rating = "✅ BUONO"
            elif avg_score >= 0.4:
                rating = "⚠️ DISCRETO"
            else:
                rating = "❌ NECESSITA MIGLIORAMENTI"

            result_lines.append(f"  🎖️ Rating globale:   {rating}")

            # Aree di miglioramento
            if zero_scores:
                result_lines.append(f"\n🔧 AREE DI MIGLIORAMENTO:")
                for metric in zero_scores:
                    result_lines.append(f"  ❌ {metric}")
        else:
            result_lines.append(
                "  ❌ Nessun punteggio disponibile per le statistiche")

        result_lines.append("=" * 80)

        return "\n".join(result_lines)

    def display_comprehensive_results(self, ragas_results, custom_results):
        """Visualizza risultati con debug esteso"""
        print("\n" + "="*80)
        print("📊 RISULTATI VALUTAZIONE RAG COMPLETA")
        print("="*80)

        # DEBUG: Verifica input
        print(f"🔍 DEBUG display_comprehensive_results:")
        print(f"  RAGAS results type: {type(ragas_results)}")
        print(f"  RAGAS results: {ragas_results}")
        print(f"  Custom results type: {type(custom_results)}")
        print(f"  Custom results: {custom_results}")
        print(
            f"  Custom results length: {len(custom_results) if custom_results else 0}")

        # Aggiungi questo blocco per debug RAGAS
        if ragas_results:
            for metric, value in ragas_results.items():
                print(
                    f"  RAGAS Metric: {metric}, Value: {value}, Type: {type(value)}")
        else:
            print("  Nessun risultato RAGAS")

        # Aggiungi questo blocco per debug Custom
        if custom_results:
            for metric, value in custom_results.items():
                print(
                    f"  Custom Metric: {metric}, Value: {value}, Type: {type(value)}")
        else:
            print("  Nessun risultato Custom")

        # Categorizza metriche RAGAS
        metric_categories = {
            "🎯 Core RAG Metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            "📝 Answer Quality": ["answer_correctness", "answer_similarity"],
            "📄 Context Analysis": ["context_entity_recall"],
            "🗣️ Language Quality": ["coherence", "fluency", "conciseness"],
            "🛡️ Safety & Ethics": ["harmfulness", "maliciousness"],
            "🔬 Specialized": ["summarization_score", "aspect_critique"]
        }

        # Mostra metriche RAGAS per categoria
        ragas_scores = []
        zero_scores = []

        print(f"\n📊 SEZIONE RAGAS METRICS:")
        if ragas_results and len(ragas_results) > 0:
            for category_name, metrics in metric_categories.items():
                category_metrics = {k: v for k,
                                    v in ragas_results.items() if k in metrics}
                if category_metrics:
                    print(f"\n{category_name}:")
                    for metric, value in category_metrics.items():
                        if not np.isnan(value):
                            status = self.get_status_emoji(value)
                            print(f"  {status} {metric:20}: {value:.4f}")
                            ragas_scores.append(value)
                            if value == 0:
                                zero_scores.append(metric)
                        else:
                            print(f"  🔶 {metric:20}: NaN")
        else:
            print("  ❌ Nessuna metrica RAGAS disponibile")

        # Mostra metriche custom con debug esteso
        print(f"\n🔧 SEZIONE CUSTOM METRICS:")
        print(f"📋 Debug custom_results:")
        print(f"  Type: {type(custom_results)}")
        print(f"  Content: {custom_results}")
        print(f"  Is dict: {isinstance(custom_results, dict)}")
        print(f"  Length: {len(custom_results) if custom_results else 0}")

        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            print(f"✅ Mostrando {len(custom_results)} metriche custom:")
            for metric, value in custom_results.items():
                print(
                    f"  Processing: {metric} = {value} (type: {type(value)})")
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        status = self.get_status_emoji(value)
                        print(f"  {status} {metric:25}: {value:.4f}")
                        custom_scores.append(value)
                    else:
                        print(f"  ⚠️ {metric:25}: {value} (invalid)")
                except Exception as e:
                    print(f"  ❌ Errore processando {metric}: {e}")
        else:
            print("  ❌ Nessuna metrica custom disponibile")
            print(f"     custom_results: {custom_results}")

        # Statistiche globali
        all_scores = ragas_scores + custom_scores

        print(f"\n📈 STATISTICHE GLOBALI:")
        print(f"  📊 RAGAS scores: {len(ragas_scores)}")
        print(f"  🔧 Custom scores: {len(custom_scores)}")
        print(f"  📋 Total scores: {len(all_scores)}")

        if all_scores:
            avg_score = np.mean(all_scores)
            median_score = np.median(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)

            print(f"  📊 Score medio:      {avg_score:.4f}")
            print(f"  🎯 Score mediano:    {median_score:.4f}")
            print(f"  🏆 Score massimo:    {max_score:.4f}")
            print(f"  ⚠️ Score minimo:     {min_score:.4f}")

            # Rating globale
            if avg_score >= 0.8:
                rating = "🏆 ECCELLENTE"
            elif avg_score >= 0.6:
                rating = "✅ BUONO"
            elif avg_score >= 0.4:
                rating = "⚠️ DISCRETO"
            else:
                rating = "❌ NECESSITA MIGLIORAMENTI"

            print(f"  🎖️ Rating globale:   {rating}")

            # Aree di miglioramento
            if zero_scores:
                print(f"\n🔧 AREE DI MIGLIORAMENTO:")
                for metric in zero_scores:
                    print(f"  ❌ {metric}")
        else:
            print("  ❌ Nessun punteggio disponibile per le statistiche")

        print("\n" + "="*80)

    def evaluate_complete(self, query, answer, contexts):
        """Metodo di compatibilità per valutazione completa"""
        return self.evaluate_all_working_metrics(query, answer, contexts)

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

        custom_results = self.calculate_custom_metrics(
            query, answer, contexts_list)

        return {'ragas': {}, 'custom': custom_results}


def extract_json(text):
    """Estrae JSON da stringhe con virgolette"""
    try:
        # Rimuovi virgolette esterne se presenti
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        # Trova il primo oggetto JSON valido
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            json_string = match.group(0)
            return json.loads(json_string)
        else:
            return None
    except json.JSONDecodeError:
        return None


def extract_score_from_error(error_message):
    """Estrae lo score da un messaggio di errore"""
    # Cerca pattern come "score: 0.8" o "value=0.5"
    match = re.search(
        r"(score|value)[:=]\s*([0-9.]+)", error_message, re.IGNORECASE)
    if match:
        try:
            return float(match.group(2))
        except ValueError:
            return None
    return None
