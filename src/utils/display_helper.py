# Utility for displaying results

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock for numpy functions

    class MockNumpy:
        @staticmethod
        def isnan(value):
            try:
                return value != value  # NaN check without numpy
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
    """Class for displaying results"""

    @staticmethod
    def get_status_emoji(score):
        """Returns emoji based on score"""
        if score >= 0.8:
            return "🟢"  # Excellent
        elif score >= 0.6:
            return "✅"  # Good
        elif score >= 0.4:
            return "⚠️"  # Fair
        else:
            return "❌"  # Poor

    @staticmethod
    def display_comprehensive_results(ragas_results, custom_results):
        """Displays results with extended debug"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE RAG EVALUATION RESULTS")
        print("="*80)

        # Categorize RAGAS metrics
        metric_categories = {
            "🎯 Core RAG Metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            "📝 Answer Quality": ["answer_correctness", "answer_similarity"],
            "📄 Context Analysis": ["context_entity_recall"],
            "🗣️ Language Quality": ["coherence", "fluency", "conciseness"],
            "🛡️ Safety & Ethics": ["harmfulness", "maliciousness"],
            "🔬 Specialized": ["summarization_score", "aspect_critique"]
        }

        # Show RAGAS metrics by category
        ragas_scores = []
        zero_scores = []

        print(f"\n📊 RAGAS METRICS SECTION:")
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
            print("  ❌ No RAGAS metrics available")

        # Show custom metrics
        print(f"\n🔧 CUSTOM METRICS SECTION:")
        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            print(f"✅ Showing {len(custom_results)} custom metrics:")
            for metric, value in custom_results.items():
                try:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        status = DisplayHelper.get_status_emoji(value)
                        print(f"  {status} {metric:25}: {value:.4f}")
                        custom_scores.append(value)
                    else:
                        print(f"  ⚠️ {metric:25}: {value} (invalid)")
                except Exception as e:
                    print(f"  ❌ Error processing {metric}: {e}")
        else:
            print("  ❌ No custom metrics available")

        # Global statistics
        all_scores = ragas_scores + custom_scores

        print(f"\n📈 GLOBAL STATISTICS:")
        print(f"  📊 RAGAS scores: {len(ragas_scores)}")
        print(f"  🔧 Custom scores: {len(custom_scores)}")
        print(f"  📋 Total scores: {len(all_scores)}")

        if all_scores:
            avg_score = np.mean(all_scores)
            median_score = np.median(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)

            print(f"  📊 Average score:    {avg_score:.4f}")
            print(f"  🎯 Median score:     {median_score:.4f}")
            print(f"  🏆 Maximum score:    {max_score:.4f}")
            print(f"  ⚠️ Minimum score:    {min_score:.4f}")

            # Global rating
            if avg_score >= 0.8:
                rating = "🏆 EXCELLENT"
            elif avg_score >= 0.6:
                rating = "✅ GOOD"
            elif avg_score >= 0.4:
                rating = "⚠️ FAIR"
            else:
                rating = "❌ NEEDS IMPROVEMENT"

            print(f"  🎖️ Global rating:    {rating}")

            # Areas for improvement
            if zero_scores:
                print(f"\n🔧 AREAS FOR IMPROVEMENT:")
                for metric in zero_scores:
                    print(f"  ❌ {metric}")
        else:
            print("  ❌ No scores available for statistics")

        print("\n" + "="*80)

    @staticmethod
    def format_comprehensive_results(ragas_results, custom_results):
        """Formats results with extended debug and returns a string"""
        output = []

        output.append("=" * 80)
        output.append("📊 COMPREHENSIVE RAG EVALUATION RESULTS")
        output.append("=" * 80)

        # Categorize RAGAS metrics
        metric_categories = {
            "🎯 Core RAG Metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            "📝 Answer Quality": ["answer_correctness", "answer_similarity"],
            "📄 Context Analysis": ["context_entity_recall"],
            "🗣️ Language Quality": ["coherence", "fluency", "conciseness"],
            "🛡️ Safety & Ethics": ["harmfulness", "maliciousness"],
            "🔬 Specialized": ["summarization_score", "aspect_critique"]
        }

        # Show RAGAS metrics by category
        ragas_scores = []
        zero_scores = []

        output.append("\n📊 RAGAS METRICS SECTION:")
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
            output.append("  ❌ No RAGAS metrics available")

        # Show custom metrics
        output.append(f"\n🔧 CUSTOM METRICS SECTION:")
        custom_scores = []
        if custom_results and isinstance(custom_results, dict) and len(custom_results) > 0:
            output.append(
                f"✅ Showing {len(custom_results)} custom metrics:")
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
                    output.append(f"  ❌ Error processing {metric}: {e}")
        else:
            output.append("  ❌ No custom metrics available")

        # Global statistics
        all_scores = ragas_scores + custom_scores

        output.append(f"\n📈 GLOBAL STATISTICS:")
        output.append(f"  📊 RAGAS scores: {len(ragas_scores)}")
        output.append(f"  🔧 Custom scores: {len(custom_scores)}")
        output.append(f"  📋 Total scores: {len(all_scores)}")

        if all_scores:
            avg_score = np.mean(all_scores)
            median_score = np.median(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)

            output.append(f"  📊 Average score:    {avg_score:.4f}")
            output.append(f"  🎯 Median score:     {median_score:.4f}")
            output.append(f"  🏆 Maximum score:    {max_score:.4f}")
            output.append(f"  ⚠️ Minimum score:    {min_score:.4f}")

            # Global rating
            if avg_score >= 0.8:
                rating = "🏆 EXCELLENT"
            elif avg_score >= 0.6:
                rating = "✅ GOOD"
            elif avg_score >= 0.4:
                rating = "⚠️ FAIR"
            else:
                rating = "❌ NEEDS IMPROVEMENT"

            output.append(f"  🎖️ Global rating:    {rating}")

            # Areas for improvement
            if zero_scores:
                output.append(f"\n🔧 AREAS FOR IMPROVEMENT:")
                for metric in zero_scores:
                    output.append(f"  ❌ {metric}")
        else:
            output.append(
                "  ❌ No scores available for statistics")

        output.append("\n" + "=" * 80)

        return "\n".join(output)

    @staticmethod
    def format_comprehensive_results_simple(ragas_results, custom_results):
        """Simplified version that returns only main statistics"""
        ragas_scores = [v for v in ragas_results.values() if isinstance(
            v, (int, float)) and not np.isnan(v)] if ragas_results else []
        custom_scores = [v for v in custom_results.values() if isinstance(
            v, (int, float)) and not np.isnan(v)] if custom_results else []
        all_scores = ragas_scores + custom_scores

        if not all_scores:
            return "❌ No scores available"

        avg_score = np.mean(all_scores)

        # Rating
        if avg_score >= 0.8:
            rating = "🏆 EXCELLENT"
        elif avg_score >= 0.6:
            rating = "✅ GOOD"
        elif avg_score >= 0.4:
            rating = "⚠️ FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"

        return f"📊 Score: {avg_score:.4f} | {rating} | RAGAS: {len(ragas_scores)} | Custom: {len(custom_scores)}"
