# Utility per la visualizzazione dei risultati

# Import condizionali
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock per numpy functions

    class MockNumpy:
        @staticmethod
        def isnan(value):
            try:
                return value != value  # NaN check senza numpy
            except:
                return False

        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0

        @staticmethod
        def median(values):
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n == 0:
                return 0
            elif n % 2 == 0:
                return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
            else:
                return sorted_values[n//2]

        @staticmethod
        def max(values):
            return max(values) if values else 0

        @staticmethod
        def min(values):
            return min(values) if values else 0

    np = MockNumpy()


class DisplayHelper:
    """Classe per la visualizzazione dei risultati"""

    @staticmethod
    def get_status_emoji(score):
        """Restituisce emoji basato sul punteggio"""
        if score >= 0.8:
            return "🟢"  # Eccellente
        elif score >= 0.6:
            return "✅"  # Buono
        elif score >= 0.4:
            return "⚠️"  # Discreto
        else:
            return "❌"  # Scarso

    @staticmethod
    def display_comprehensive_results(ragas_results, custom_results):
        """Visualizza risultati con debug esteso"""
        print("\n" + "="*80)
        print("📊 RISULTATI VALUTAZIONE RAG COMPLETA")
        print("="*80)

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
                            status = DisplayHelper.get_status_emoji(value)
                            print(f"  {status} {metric:20}: {value:.4f}")
                            ragas_scores.append(value)
                            if value == 0:
                                zero_scores.append(metric)
                        else:
                            print(f"  🔶 {metric:20}: NaN")
        else:
            print("  ❌ Nessuna metrica RAGAS disponibile")

        # Mostra metriche custom
        print(f"\n🔧 SEZIONE CUSTOM METRICS:")
        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            print(f"✅ Mostrando {len(custom_results)} metriche custom:")
            for metric, value in custom_results.items():
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        status = DisplayHelper.get_status_emoji(value)
                        print(f"  {status} {metric:25}: {value:.4f}")
                        custom_scores.append(value)
                    else:
                        print(f"  ⚠️ {metric:25}: {value} (invalid)")
                except Exception as e:
                    print(f"  ❌ Errore processando {metric}: {e}")
        else:
            print("  ❌ Nessuna metrica custom disponibile")

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

    @staticmethod
    def format_comprehensive_results(ragas_results, custom_results):
        """Formatta risultati con debug esteso e restituisce una stringa"""
        output = []

        output.append("=" * 80)
        output.append("📊 RISULTATI VALUTAZIONE RAG COMPLETA")
        output.append("=" * 80)

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

        output.append("\n📊 SEZIONE RAGAS METRICS:")
        if ragas_results and len(ragas_results) > 0:
            for category_name, metrics in metric_categories.items():
                category_metrics = {
                    k: v for k, v in ragas_results.items() if k in metrics}
                if category_metrics:
                    output.append(f"\n{category_name}:")
                    for metric, value in category_metrics.items():
                        if not np.isnan(value):
                            status = DisplayHelper.get_status_emoji(value)
                            output.append(
                                f"  {status} {metric:20}: {value:.4f}")
                            ragas_scores.append(value)
                            if value == 0:
                                zero_scores.append(metric)
                        else:
                            output.append(f"  🔶 {metric:20}: NaN")
        else:
            output.append("  ❌ Nessuna metrica RAGAS disponibile")

        # Mostra metriche custom
        output.append(f"\n🔧 SEZIONE CUSTOM METRICS:")
        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            output.append(
                f"✅ Mostrando {len(custom_results)} metriche custom:")
            for metric, value in custom_results.items():
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        status = DisplayHelper.get_status_emoji(value)
                        output.append(
                            f"  {status} {metric:25}: {value:.4f}")
                        custom_scores.append(value)
                    else:
                        output.append(
                            f"  ⚠️ {metric:25}: {value} (invalid)")
                except Exception as e:
                    output.append(f"  ❌ Errore processando {metric}: {e}")
        else:
            output.append("  ❌ Nessuna metrica custom disponibile")

        # Statistiche globali
        all_scores = ragas_scores + custom_scores

        output.append(f"\n📈 STATISTICHE GLOBALI:")
        output.append(f"  📊 RAGAS scores: {len(ragas_scores)}")
        output.append(f"  🔧 Custom scores: {len(custom_scores)}")
        output.append(f"  📋 Total scores: {len(all_scores)}")

        if all_scores:
            avg_score = np.mean(all_scores)
            median_score = np.median(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)

            output.append(f"  📊 Score medio:      {avg_score:.4f}")
            output.append(f"  🎯 Score mediano:    {median_score:.4f}")
            output.append(f"  🏆 Score massimo:    {max_score:.4f}")
            output.append(f"  ⚠️ Score minimo:     {min_score:.4f}")

            # Rating globale
            if avg_score >= 0.8:
                rating = "🏆 ECCELLENTE"
            elif avg_score >= 0.6:
                rating = "✅ BUONO"
            elif avg_score >= 0.4:
                rating = "⚠️ DISCRETO"
            else:
                rating = "❌ NECESSITA MIGLIORAMENTI"

            output.append(f"  🎖️ Rating globale:   {rating}")

            # Aree di miglioramento
            if zero_scores:
                output.append(f"\n🔧 AREE DI MIGLIORAMENTO:")
                for metric in zero_scores:
                    output.append(f"  ❌ {metric}")
        else:
            output.append(
                "  ❌ Nessun punteggio disponibile per le statistiche")

        output.append("\n" + "=" * 80)

        return "\n".join(output)

    @staticmethod
    def format_comprehensive_results_simple(ragas_results, custom_results):
        """Versione semplificata che restituisce solo le statistiche principali"""
        ragas_scores = [v for v in ragas_results.values() if isinstance(
            v, (int, float)) and not np.isnan(v)] if ragas_results else []
        custom_scores = [v for v in custom_results.values() if isinstance(
            v, (int, float)) and not np.isnan(v)] if custom_results else []
        all_scores = ragas_scores + custom_scores

        if not all_scores:
            return "❌ Nessun punteggio disponibile"

        avg_score = np.mean(all_scores)

        # Rating
        if avg_score >= 0.8:
            rating = "🏆 ECCELLENTE"
        elif avg_score >= 0.6:
            rating = "✅ BUONO"
        elif avg_score >= 0.4:
            rating = "⚠️ DISCRETO"
        else:
            rating = "❌ NECESSITA MIGLIORAMENTI"

        return f"📊 Score: {avg_score:.4f} | {rating} | RAGAS: {len(ragas_scores)} | Custom: {len(custom_scores)}"
