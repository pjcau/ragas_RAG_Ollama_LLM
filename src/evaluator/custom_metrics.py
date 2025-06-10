from ..config.settings import CUSTOM_METRICS_CONFIG


class CustomMetrics:

    @staticmethod
    def calculate_custom_metrics(query, answer, contexts):

        if not query or not answer:
            return {}

        if not contexts or len(contexts) == 0:
            return {}

        # Prepara testi per analisi
        query_lower = query.lower().strip()
        answer_lower = answer.lower().strip()

        query_words = set([w.strip()
                          for w in query_lower.split() if w.strip()])
        answer_words = set([w.strip()
                           for w in answer_lower.split() if w.strip()])

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

        custom_results = {}

        try:
            # 1. Jaccard Similarity (Query-Answer)
            if query_words and answer_words:
                intersection = len(query_words.intersection(answer_words))
                union = len(query_words.union(answer_words))
                jaccard_sim = intersection / union if union > 0 else 0
                custom_results['jaccard_similarity'] = round(jaccard_sim, 4)

            # 2. Context Coverage (Answer utilizza Context)
            if answer_words and context_words:
                answer_in_context = len(
                    answer_words.intersection(context_words))
                context_coverage = answer_in_context / \
                    len(answer_words) if answer_words else 0
                custom_results['context_coverage'] = round(context_coverage, 4)

            # 3. Information Density
            if answer and answer.strip():
                words = answer.split()
                unique_words = set([w.lower().strip()
                                   for w in words if w.strip()])
                info_density = len(unique_words) / len(words) if words else 0
                custom_results['information_density'] = round(info_density, 4)

            # 4. Query Coverage (Answer copre Query)
            if query_words and answer_words:
                query_covered = len(query_words.intersection(answer_words))
                query_coverage = query_covered / \
                    len(query_words) if query_words else 0
                custom_results['query_coverage'] = round(query_coverage, 4)

            # 5. Answer Length Score (normalizzato)
            if answer and answer.strip():
                answer_length = len(answer.split())
                optimal_length = CUSTOM_METRICS_CONFIG['optimal_length']
                length_score = min(answer_length / optimal_length,
                                   1.0) if optimal_length > 0 else 0
                custom_results['answer_length_score'] = round(length_score, 4)

            # 6. Technical Terms Density
            if answer and answer.strip():
                words = answer.split()
                long_words = [w for w in words if len(
                    w) > CUSTOM_METRICS_CONFIG['technical_word_min_length']]
                tech_density = len(long_words) / len(words) if words else 0
                custom_results['technical_density'] = round(tech_density, 4)

            # 7. Context Relevance (Query vs Context)
            if query_words and context_words:
                query_in_context = len(query_words.intersection(context_words))
                context_relevance = query_in_context / \
                    len(query_words) if query_words else 0
                custom_results['context_relevance'] = round(
                    context_relevance, 4)

            # 8. Answer Completeness
            if answer and query:
                answer_completeness = min(
                    len(answer) / CUSTOM_METRICS_CONFIG['answer_completeness_cap'], 1.0)
                custom_results['answer_completeness'] = round(
                    answer_completeness, 4)

        except Exception as e:

        return custom_results
