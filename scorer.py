import re
import json
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

SENTENCE_TRANSFORMER_REPO = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
ROUGE_L_THRESHOLD = 0.2


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (for ROUGE-L scoring)."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


@scorer(metrics=[mean(), stderr()])
def rouge_l_scorer() -> Scorer:
    """ROUGE-L scorer for the SEVENLLM benchmark.

    This scorer calculates the ROUGE-L score to evaluate the similarity between
    the model-generated output and the reference text on a per-sentence basis.
    The ROUGE-L metric focuses on the longest common subsequence (LCS) between
    the sentences, which captures the structure and fluency of the text. 

    The routine:
    - Splits the target reference and model-generated text into sentences.
    - Tokenizes each sentence (with support for Chinese tokenization).
    - Computes the ROUGE-L similarity score for each pair of sentences.
    - Aggregates the scores, with sentences scoring above a defined threshold
      (default: 0.2) contributing to the final score proportionally.

    The total score reflects the percentage of sentences in the reference text
    that have sufficiently similar counterparts in the generated output,
    scaled to 100%.

    Returns:
        Scorer: A scoring function that computes the aggregated ROUGE-L score
        and provides the result alongside the model-generated output and a
        detailed explanation.
    """
    # Import rouge here to avoid loading it in the global scope
    from rouge import Rouge  # type: ignore

    rouge = Rouge()

    async def score(state: TaskState, target: Target) -> Score:
        test_sentences = split_sentences(target.text)
        generated_sentences = split_sentences(state.output.completion)
        num_sentences = len(test_sentences)
        per_sentence_score = 100.0 / num_sentences  # Adjust scaling as needed
        total_score = 0.0

        for test_sentence in test_sentences:
            max_similarity = 0.0
            for gen_sentence in generated_sentences:
                test_sentence_tok = test_sentence
                gen_sentence_tok = gen_sentence
                similarity = 0.0  # Default similarity
                TRUNCATION_LIMIT = 1000 # Max characters for truncated sentences

                try:
                    # Ensure sentences are not empty before calling.
                    if not test_sentence_tok.strip() or not gen_sentence_tok.strip():
                        similarity = 0.0
                    else:
                        scores = rouge.get_scores(test_sentence_tok, gen_sentence_tok)
                        similarity = scores[0]["rouge-l"]["f"]
                except RecursionError:
                    print(f"WARNING: RecursionError occurred in ROUGE-L calculation. Attempting with truncated sentences.")
                    print(f"  Original Target sentence (first 200 chars): \'{test_sentence_tok[:200]}...\'")
                    print(f"  Original Generated sentence (first 200 chars): \'{gen_sentence_tok[:200]}...\'")

                    truncated_test_sentence = test_sentence_tok[:TRUNCATION_LIMIT]
                    truncated_gen_sentence = gen_sentence_tok[:TRUNCATION_LIMIT]

                    if not truncated_test_sentence.strip() or not truncated_gen_sentence.strip():
                        similarity = 0.0
                        print(f"  INFO: One or both sentences became empty after truncation. Assigning similarity 0.0.")
                    else:
                        try:
                            scores_truncated = rouge.get_scores(truncated_test_sentence, truncated_gen_sentence)
                            similarity = scores_truncated[0]["rouge-l"]["f"]
                            print(f"  INFO: Successfully calculated ROUGE-L with truncated sentences. Score: {similarity:.4f}")
                        except RecursionError:
                            similarity = 0.0
                            print(f"  ERROR: RecursionError persisted even with truncated sentences (limit: {TRUNCATION_LIMIT} chars). Assigning similarity 0.0.")
                        except Exception as e_truncated:
                            similarity = 0.0
                            print(f"  ERROR: An unexpected error occurred with truncated ROUGE-L calculation: {str(e_truncated)}. Assigning similarity 0.0.")
                
                except Exception as e:
                    # Catch other potential errors from rouge.get_scores for robustness
                    similarity = 0.0
                    print(f"WARNING: An unexpected error occurred in ROUGE-L calculation: {str(e)}")
                    print(f"  Target sentence (first 200 chars): \'{test_sentence_tok[:200]}...\'")
                    print(f"  Generated sentence (first 200 chars): \'{gen_sentence_tok[:200]}...\'")
                    print(f"  Assigning similarity 0.0 for this pair and continuing.")

                if similarity > max_similarity:
                    max_similarity = similarity
            if max_similarity >= ROUGE_L_THRESHOLD:
                total_score += per_sentence_score

        return Score(
            value=total_score,
            answer=state.output.completion,
            explanation=f"Accumulated ROUGE-L score: {total_score:.4f}",
        )

    return score


@scorer(metrics=[mean(), stderr()])
def semantic_similarity_scorer() -> Scorer:
    """Semantic similarity scorer for the SEVENLLM benchmark.

    This scorer evaluates the semantic similarity between the model-generated output
    and the reference target text using a pre-trained SentenceTransformer model.
    It computes cosine similarity between the embeddings of the two texts, providing
    a quantitative measure of their alignment in meaning.

    The routine:
    - Encodes both the model's output and the reference target text into embeddings
      using a SentenceTransformer.
    - Computes the cosine similarity between the two embeddings as the similarity score.
    - Returns the score along with the model output and an explanation string.

    The score is aggregated using mean and standard error (stderr) metrics to evaluate
    model performance across the benchmark.

    Returns:
        Scorer: A scoring function that computes semantic similarity and provides
        both the similarity score and a detailed explanation of the result.
    """
    # Import SentenceTransformer here to avoid loading it in the global scope
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    model = SentenceTransformer(SENTENCE_TRANSFORMER_REPO)

    async def score(state: TaskState, target: Target) -> Score:
        model_output = state.output.completion
        target_text = target.text
        embeddings = model.encode([model_output, target_text], convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        similarity_score = cosine_similarity.item()

        return Score(
            value=similarity_score,
            answer=model_output,
            explanation=f"Semantic similarity score: {similarity_score:.4f}",
        )

    return score


@scorer(metrics=[mean(), stderr()])
def affected_components_scorer() -> Scorer:
    """
    Scores based on the percentage of 'affectedComponents' from the target JSON
    that are found as substrings in the model output string.

    The primary score is the fraction of unique affected components from the target
    that are present in the model's output. If the target lists no components,
    the score is 1.0 (100%).

    The routine:
    - Parses the target text as a JSON list of threat objects.
    - Extracts all unique 'affectedComponents' strings from the target.
    - Searches for each unique target component as a substring within the raw model output string.
    - Calculates the score as (count of found target components) / (total unique target components).
      If total unique target components is 0, the score is 1.0.
    - The explanation provides the percentage score, counts (TP, FN), lists of components,
      and attempts to parse the model output to provide contextual False Positives.
    """
    import json # Standard library
    from typing import Any # For typing the helper

    def _extract_components_from_parsed_json(data_source: list[Any], source_name_for_error: str) -> set[str]:
        """Helper to extract unique, non-empty affected components from a parsed JSON list."""
        components: set[str] = set()
        if not isinstance(data_source, list):
            # This case should be handled by the caller, but as a safeguard:
            return components

        for threat_idx, threat_obj in enumerate(data_source):
            if not isinstance(threat_obj, dict):
                # Silently skip non-dictionary items in the list.
                # Alternatively, could log: f"Item at index {threat_idx} in {source_name_for_error} is not a dictionary."
                continue
            
            affected_list = threat_obj.get("affectedComponents")
            if affected_list is None: # Key might be missing
                continue
            if not isinstance(affected_list, list):
                # Silently skip if 'affectedComponents' is not a list.
                # Alternatively, could log: f"'affectedComponents' in threat {threat_obj.get('threatID', 'N/A')} from {source_name_for_error} is not a list."
                continue

            for component_item in affected_list:
                if isinstance(component_item, str):
                    stripped_component = component_item.strip()
                    if stripped_component: # Add only non-empty strings
                        components.add(stripped_component)
                # else:
                    # Silently skip non-string items in affected_list.
                    # Alternatively, could log: f"Invalid component item '{component_item}' in {source_name_for_error}."
        return components

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion

        try:
            target_data = json.loads(target.text)
        except json.JSONDecodeError as e:
            return Score(value=0.0, answer=state.output.completion, explanation=f"Error parsing target text as JSON: {e}. Cannot determine percentage of true positives.")
        
        if not isinstance(target_data, list):
            return Score(value=0.0, answer=state.output.completion, explanation="Target text JSON is not a list of threats. Cannot determine percentage of true positives.")

        target_components = _extract_components_from_parsed_json(target_data, "target text")
        num_target_components = len(target_components)

        # Calculate True Positives by searching target components in model_output_str
        true_positives_set = set()
        if target_components: # Only search if there are target components
            for component in target_components:
                if component in model_output_str: # Substring search
                    true_positives_set.add(component)
        
        tp_count = len(true_positives_set)
        
        score_value: float
        score_explanation_prefix: str

        if num_target_components == 0:
            score_value = 1.0 # All (zero) target components were 'found'.
            score_explanation_prefix = f"Target listed no affected components. Score: 1.0 (100%) as all zero components were 'found'."
        else:
            score_value = float(tp_count) / num_target_components
            score_explanation_prefix = f"Percentage of target components found in model output string: {score_value*100:.2f}% ({tp_count}/{num_target_components})."

        # For explanation: try to parse model output to get its components for a fuller picture
        model_components_for_explanation = set()
        model_parse_error_for_explanation = ""
        try:
            model_data_for_expl = json.loads(state.output.completion)
            if isinstance(model_data_for_expl, list):
                model_components_for_explanation = _extract_components_from_parsed_json(model_data_for_expl, "model output (for explanation)")
            else:
                model_parse_error_for_explanation = "Model output was not a JSON list (for explanation)."
        except json.JSONDecodeError as e:
            model_parse_error_for_explanation = f"Model output was not valid JSON (for explanation): {e}"

        # False Negatives (target components not found in model output string)
        false_negatives_set = set()
        if target_components: # This will be true if num_target_components > 0
            false_negatives_set = target_components.difference(true_positives_set) 
        fn_count = len(false_negatives_set)

        # False Positives for explanation 
        # (components in parsable model output but not in parsable target)
        false_positives_set_for_expl = set()
        if model_components_for_explanation and target_components:
             false_positives_set_for_expl = model_components_for_explanation.difference(target_components)
        fp_count_expl = len(false_positives_set_for_expl)
        
        explanation_parts = [score_explanation_prefix]

        explanation_parts.append(f"Target unique affected components (from parsed target JSON): {num_target_components}.")
        if model_parse_error_for_explanation:
            explanation_parts.append(f"Model unique affected components (for context, from model output string): Not determined due to model parsing issue ({model_parse_error_for_explanation}).")
        else:
            explanation_parts.append(f"Model unique affected components (for context, if parsable as JSON): {len(model_components_for_explanation)}.")

        explanation_parts.append(f"Matched target components found in model output string (True Positives): {tp_count}.")
        if true_positives_set:
            explanation_parts.append(f"  Matched in string: {', '.join(sorted(list(true_positives_set)))}.")
        
        explanation_parts.append(f"Extra components in (parsable) model output compared to (parsable) target (Contextual False Positives): {fp_count_expl}.")
        if false_positives_set_for_expl:
            explanation_parts.append(f"  Contextual FPs: {', '.join(sorted(list(false_positives_set_for_expl)))}.")

        explanation_parts.append(f"Missing target components from model output string (Contextual False Negatives): {fn_count}.")
        if false_negatives_set:
             explanation_parts.append(f"  Contextual FNs: {', '.join(sorted(list(false_negatives_set)))}.")
        elif num_target_components > 0: # Target had components, but all were found
            explanation_parts.append("  Contextual FNs: None (all target components found in model string).")
        # If num_target_components == 0, FNs are 0, covered by tp_count being 0.

        final_explanation = " ".join(explanation_parts)
        
        # The primary score is the percentage of target components found.
        return Score(
            value=score_value,
            answer=state.output.completion,
            explanation=final_explanation
        )

    return score


@scorer(metrics=[mean(), stderr()])
def affected_components_fp_scorer() -> Scorer:
    """
    Scores based on the percentage of 'affectedComponents' declared by the model
    (if parsable) that are NOT found as substrings in the target text (False Positives).

    The primary score is (Count of FP components) / (Total components declared by model).
    If the model output is unparsable or declares no components, the FP percentage is 0%.

    The routine:
    - Attempts to parse model output to extract 'model_components'. This can be from a list of
      threat objects or a direct list of component strings.
    - For each extracted model component, checks if it exists as a substring in the raw target text.
    - FP Count = number of model components NOT found in target text.
    - Score = FP Count / Total Model Components (if Total > 0, else 0.0).
    - Explanation provides the percentage, counts, and details.
    """
    import json
    from typing import Any

    def _extract_components_from_parsed_json(data_source: list[Any], source_name_for_error: str) -> set[str]:
        """Helper to extract unique, non-empty affected components from a parsed JSON list."""
        components: set[str] = set()
        if not isinstance(data_source, list):
            return components

        for threat_obj in data_source:
            if not isinstance(threat_obj, dict):
                continue
            
            affected_list = threat_obj.get("affectedComponents")
            if affected_list is None or not isinstance(affected_list, list):
                continue

            for component_item in affected_list:
                if isinstance(component_item, str):
                    stripped_component = component_item.strip()
                    if stripped_component:
                        components.add(stripped_component)
        return components

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion
        target_text_str = target.text

        model_components_for_fp_calc: set[str] = set()
        parsed_model_successfully = False # Initialize to False
        model_parsing_explanation = "" 

        try:
            model_data = json.loads(model_output_str)
            if isinstance(model_data, list):
                if all(isinstance(item, str) for item in model_data):
                    model_components_for_fp_calc = {s.strip() for s in model_data if s.strip()}
                    parsed_model_successfully = True
                    model_parsing_explanation = "Model output parsed as a JSON list of strings (treated as affected components)."
                elif all(isinstance(item, dict) for item in model_data):
                    model_components_for_fp_calc = _extract_components_from_parsed_json(model_data, "model output")
                    parsed_model_successfully = True # Successfully parsed and extracted or attempted extraction
                    model_parsing_explanation = "Model output parsed as a JSON list of threat objects."
                    if not model_components_for_fp_calc:
                         model_parsing_explanation += " No 'affectedComponents' found or structure within objects not as expected."
                    # Even if no components are found from valid structure, consider it a successful parse attempt
                else:
                    # parsed_model_successfully remains False by default
                    model_parsing_explanation = "Model output is a JSON list but not uniformly strings or threat objects. Cannot reliably extract components for FP."
            else:
                # parsed_model_successfully remains False
                model_parsing_explanation = "Model output is valid JSON but not a list. Cannot extract components for FP."
        except json.JSONDecodeError as e:
            # parsed_model_successfully remains False
            model_parsing_explanation = f"Model output not valid JSON: {e}. Cannot extract structured components for FP."
        
        num_model_components = len(model_components_for_fp_calc)
        false_positives_set = set()
        if model_components_for_fp_calc:
            for component in model_components_for_fp_calc:
                if component not in target_text_str:
                    false_positives_set.add(component)
        
        fp_count = len(false_positives_set)
        
        score_value: float
        if num_model_components == 0:
            score_value = 0.0 # 0 FPs out of 0 declared = 0% FP rate
            score_explanation_prefix = f"Model declared 0 components ({model_parsing_explanation}). False Positive component percentage: 0.00% (0/0)."
        else:
            score_value = float(fp_count) / num_model_components
            score_explanation_prefix = f"Percentage of model-declared components that are False Positives: {score_value*100:.2f}% ({fp_count}/{num_model_components})."

        # For explanation context: try parse target and calc TPs/FNs (target_comp in model_output_str)
        target_components_for_expl: set[str] = set()
        parsed_target_successfully_for_expl = False
        target_parse_error_msg_for_expl = ""
        try:
            target_data_for_expl = json.loads(target_text_str)
            if isinstance(target_data_for_expl, list):
                # Attempt to extract from list of dicts or list of strings for target explanation
                if all(isinstance(item, str) for item in target_data_for_expl):
                    target_components_for_expl = {s.strip() for s in target_data_for_expl if s.strip()}
                    parsed_target_successfully_for_expl = True
                elif all(isinstance(item, dict) for item in target_data_for_expl):
                    target_components_for_expl = _extract_components_from_parsed_json(target_data_for_expl, "target text (for explanation)")
                    parsed_target_successfully_for_expl = True
                if parsed_target_successfully_for_expl and not target_components_for_expl:
                    target_parse_error_msg_for_expl = "Target parsed as JSON list, but no components extracted (e.g. empty list, or list of dicts without components)."
            else:
                target_parse_error_msg_for_expl = "Target text JSON is valid but not a list (for explanation context)."
        except json.JSONDecodeError as e:
            target_parse_error_msg_for_expl = f"Error parsing target text as JSON (for explanation context): {e}."

        true_positives_set_for_expl = set()
        if parsed_target_successfully_for_expl and target_components_for_expl:
            for tc in target_components_for_expl:
                if tc in model_output_str: 
                    true_positives_set_for_expl.add(tc)
        tp_count_for_expl = len(true_positives_set_for_expl)

        false_negatives_set_for_expl = set()
        if parsed_target_successfully_for_expl and target_components_for_expl:
            false_negatives_set_for_expl = target_components_for_expl.difference(true_positives_set_for_expl)
        fn_count_for_expl = len(false_negatives_set_for_expl)

        explanation_parts = [
            score_explanation_prefix, # Updated to include the percentage and ratio
            f"Model parsing details: {model_parsing_explanation}",
            f"Model unique affected components (extracted for FP calc): {num_model_components} {sorted(list(model_components_for_fp_calc)) if model_components_for_fp_calc else '[]'}."
        ]
        
        explanation_parts.append(f"Target unique affected components (from parsed target, for context): {len(target_components_for_expl)} {sorted(list(target_components_for_expl)) if target_components_for_expl else '[]'}.")
        if target_parse_error_msg_for_expl:
            explanation_parts.append(f"(Target parsing for context: {target_parse_error_msg_for_expl})")

        explanation_parts.append(f"Contextual TPs (target components in model string): {tp_count_for_expl}.")
        if true_positives_set_for_expl:
            explanation_parts.append(f"  TPs details: {', '.join(sorted(list(true_positives_set_for_expl)))}.")
        
        if false_positives_set: 
            explanation_parts.append(f"  FPs details (contributing to score): {', '.join(sorted(list(false_positives_set)))}.")
        elif parsed_model_successfully and not model_components_for_fp_calc : 
             explanation_parts.append("  FPs details: None (Model parsed but yielded no components for FP calculation).")
        elif parsed_model_successfully and model_components_for_fp_calc:
             explanation_parts.append("  FPs details: None (All extracted model components were found in target text string).")
        elif not parsed_model_successfully:
             explanation_parts.append("  FPs details: None (Model output not parsable into structured components).")

        explanation_parts.append(f"Contextual FNs (target components not in model string): {fn_count_for_expl}.")
        if false_negatives_set_for_expl:
             explanation_parts.append(f"  FNs details: {', '.join(sorted(list(false_negatives_set_for_expl)))}.")
        elif parsed_target_successfully_for_expl and target_components_for_expl:
             explanation_parts.append("  FNs details: None (All parsed target components found in model string).")
        elif parsed_target_successfully_for_expl and not target_components_for_expl:
            explanation_parts.append("  FNs details: None (Target parsed but yielded no components).")

        final_explanation = " ".join(explanation_parts)
        
        return Score(
            value=score_value, # Use the calculated percentage
            answer=state.output.completion,
            explanation=final_explanation
        )

    return score


# Helper function for STRIDE scorers
def _extract_stride_categories(data_source: list[Any], source_name_for_error: str) -> set[str]:
    """Extracts unique, normalized STRIDE categories from a parsed JSON list of threats."""
    stride_categories: set[str] = set()
    if not isinstance(data_source, list):
        # This situation should ideally be caught by the caller before passing to helper.
        # However, as a safeguard, return empty set.
        # print(f"Warning: {source_name_for_error} data is not a list in _extract_stride_categories.")
        return stride_categories

    for threat_obj in data_source:
        if not isinstance(threat_obj, dict):
            # print(f"Warning: Item in {source_name_for_error} is not a dictionary.")
            continue
        
        frameworks = threat_obj.get("frameworks")
        if not isinstance(frameworks, dict):
            # print(f"Warning: 'frameworks' in {source_name_for_error} threat {threat_obj.get('threatID', '')} is not a dictionary.")
            continue
        
        stride_list = frameworks.get("stride")
        if not isinstance(stride_list, list):
            # print(f"Warning: 'stride' in {source_name_for_error} threat {threat_obj.get('threatID', '')} is not a list.")
            continue

        for item in stride_list:
            if isinstance(item, str):
                normalized_item = item.strip().lower() # Normalize: lowercase and strip whitespace
                if normalized_item:
                    stride_categories.add(normalized_item)
            # else:
                # print(f"Warning: Non-string item '{item}' in stride list for {source_name_for_error} threat {threat_obj.get('threatID', '')}.")
    return stride_categories


@scorer(metrics=[mean(), stderr()])
def stride_fraction_identified_scorer() -> Scorer:
    """
    Calculates the percentage of unique STRIDE categories from the target
    that are also identified as substrings in the (normalized) model output.
    Score = (Common Unique STRIDE Categories) / (Total Unique STRIDE Categories in Target).
    Handles cases where the target has no STRIDE categories (score 1.0 or 0.0 based on model).
    Handles cases where target is unparsable (score 1.0 or 0.0 based on model).
    """
    import json
    from typing import Any # Required by the helper, though not directly used in this function signature

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion
        target_text_str = target.text

        target_strides: set[str] = set()
        target_data_parsed_list: list[Any] | None = None # To check if parsing led to a list
        target_parse_error_msg = ""

        # 1. Attempt to parse Target
        try:
            parsed_json_target = json.loads(target_text_str)
            if isinstance(parsed_json_target, list):
                target_data_parsed_list = parsed_json_target
                target_strides = _extract_stride_categories(target_data_parsed_list, "target text")
                if not target_strides and not target_data_parsed_list: # e.g. empty list from json.loads
                    pass # target_strides is already empty
                elif not target_strides: # Parsed as list, but helper found no strides
                    target_parse_error_msg = "Target parsed as JSON list, but no STRIDE categories extracted (e.g., empty objects or wrong structure)."
            else:
                target_parse_error_msg = "Target text JSON is valid but not a list."
        except json.JSONDecodeError as e:
            target_parse_error_msg = f"Error parsing target text as JSON: {e}."

        # 2. Handle cases where target is unusable (parse error, or not a list)
        if target_data_parsed_list is None: # True if JSONDecodeError or not a list initially
            model_strides_for_context: set[str] = set()
            model_context_parse_details = "(for context) successfully parsed and yielded no STRIDE categories"
            try:
                model_data_context = json.loads(model_output_str)
                if isinstance(model_data_context, list):
                    model_strides_for_context = _extract_stride_categories(model_data_context, "model output (context for bad target)")
                    if model_strides_for_context:
                        model_context_parse_details = f"(for context) successfully parsed and yielded {len(model_strides_for_context)} STRIDE categories"
                else: # valid JSON but not a list
                    model_context_parse_details = "(for context) was valid JSON but not a list"
            except json.JSONDecodeError as e:
                model_context_parse_details = f"(for context) could not be parsed as JSON: {e}"

            final_score_value = 0.0
            explanation = f"{target_parse_error_msg} Cannot determine target STRIDE categories. "
            
            if not model_strides_for_context: # Model also effectively empty/unusable
                final_score_value = 1.0
                explanation += f"Model output {model_context_parse_details}. Score implies 0/0 = 100% due to unusable target and model also appearing empty/unusable."
            else: # Model has strides, target is bad
                final_score_value = 0.0
                explanation += f"Model output {model_context_parse_details}: {sorted(list(model_strides_for_context))}. Score 0.0% as target is unusable."
            
            return Score(value=final_score_value, answer=model_output_str, explanation=explanation)

        # Target was parsed successfully as a list, and target_strides are extracted (possibly empty).
        num_target_strides = len(target_strides)

        # 3. Identify common strides by substring search in (normalized) model_output_str
        # _extract_stride_categories normalizes strides to .strip().lower()
        # So, t_stride is already normalized. Search in model_output_str.lower().
        normalized_model_output_str = model_output_str.lower()
        identified_strides_in_model_string: set[str] = set()
        if target_strides: # Only search if there are target strides to look for
            for t_stride in target_strides:
                if t_stride in normalized_model_output_str:
                    identified_strides_in_model_string.add(t_stride)
        
        num_identified = len(identified_strides_in_model_string)

        # 4. Calculate score value based on num_target_strides and num_identified
        current_score_value: float
        explanation_main_part: str

        if num_target_strides == 0:
            # Target parsed successfully but had no STRIDE categories.
            # Check model output (by parsing it) for the 0/0=1.0 or N/0=0.0 case scenario.
            model_strides_for_00_context: set[str] = set()
            model_00_context_parse_details = "(for 0/0 context) successfully parsed and yielded no STRIDE categories"
            try:
                model_data_00_context = json.loads(model_output_str) 
                if isinstance(model_data_00_context, list):
                    model_strides_for_00_context = _extract_stride_categories(model_data_00_context, "model output (for 0/0 context)")
                    if model_strides_for_00_context:
                        model_00_context_parse_details = f"(for 0/0 context) successfully parsed and yielded {len(model_strides_for_00_context)} categories"
                else: # valid JSON but not a list
                    model_00_context_parse_details = "(for 0/0 context) was valid JSON but not a list"
            except json.JSONDecodeError as e: 
                model_00_context_parse_details = f"(for 0/0 context) could not be parsed as JSON: {e}"
               
            if not model_strides_for_00_context: # Model also effectively has no strides (empty or unparsable)
                current_score_value = 1.0
                explanation_main_part = (f"Target (parsed) listed no STRIDE categories {target_parse_error_msg if target_parse_error_msg else ''}. "
                                         f"Model output {model_00_context_parse_details}. Percentage identified: {current_score_value*100:.2f}% (0/0 case).")
            else: # Target has 0, model (parsed for context) has some
                current_score_value = 0.0
                explanation_main_part = (f"Target (parsed) listed no STRIDE categories {target_parse_error_msg if target_parse_error_msg else ''}. "
                                         f"Model output {model_00_context_parse_details}: {sorted(list(model_strides_for_00_context))}. Percentage identified: {current_score_value*100:.2f}% ({len(model_strides_for_00_context)} found in model, 0 expected).")
        else: # num_target_strides > 0
            current_score_value = num_identified / num_target_strides
            explanation_main_part = (f"Percentage of target STRIDE categories ({num_target_strides}) identified by substring search in normalized model output: "
                                     f"{current_score_value*100:.2f}% ({num_identified}/{num_target_strides}).")

        # 5. Explanation details (FP, FN using parsed model for fuller context)
        model_strides_for_explanation: set[str] = set()
        model_explanation_parse_error = ""
        try:
            model_data_for_expl = json.loads(model_output_str)
            if isinstance(model_data_for_expl, list):
                model_strides_for_explanation = _extract_stride_categories(model_data_for_expl, "model output (for explanation)")
            else:
                model_explanation_parse_error = "Model output (for explanation context) was valid JSON but not a list."
        except json.JSONDecodeError as e:
            model_explanation_parse_error = f"Model output (for explanation context) could not be parsed as JSON: {e}."

        # FP (for explanation): strides in (parsed) model output but not in (parsed) target
        fp_strides_expl: set[str] = model_strides_for_explanation.difference(target_strides)
        
        # FN (for explanation): strides in (parsed) target but not found by substring search in normalized model string
        fn_strides_expl: set[str] = target_strides.difference(identified_strides_in_model_string)

        explanation_parts = [explanation_main_part]
        explanation_parts.append(f"Target unique STRIDE categories (parsed from target.text, normalized): {len(target_strides)} {sorted(list(target_strides)) if target_strides else '[]'}.")
        if not target_strides and not target_parse_error_msg and target_data_parsed_list is not None: 
            explanation_parts.append("(Target parsed as list but yielded no STRIDE categories e.g. empty list or items without STRIDE data).")
        elif target_parse_error_msg and num_target_strides == 0 : # For clarity if target parsing failed but led to 0 strides (e.g. not a list)
             explanation_parts.append(f"(Target parsing issue: {target_parse_error_msg}).")

        if model_explanation_parse_error:
            explanation_parts.append(f"Contextual model STRIDE categories: Not determined due to parsing issue ({model_explanation_parse_error}).")
        else:
            explanation_parts.append(f"Contextual model STRIDE categories (parsed from model output, normalized for context): {len(model_strides_for_explanation)} {sorted(list(model_strides_for_explanation)) if model_strides_for_explanation else '[]'}.\"")

        explanation_parts.append(f"Identified in normalized model string (True Positives by string search): {num_identified} {sorted(list(identified_strides_in_model_string)) if identified_strides_in_model_string else '[]'}.\"")
        explanation_parts.append(f"Contextual False Positives (parsed model vs parsed target): {len(fp_strides_expl)} {sorted(list(fp_strides_expl)) if fp_strides_expl else '[]'}.\"")
        explanation_parts.append(f"Contextual False Negatives (parsed target vs string search in model): {len(fn_strides_expl)} {sorted(list(fn_strides_expl)) if fn_strides_expl else '[]'}.\"")
        
        final_explanation = " ".join(explanation_parts)

        return Score(
            value=current_score_value,
            answer=model_output_str, 
            explanation=final_explanation
        )
    return score


@scorer(metrics=[mean(), stderr()])
def stride_false_positive_scorer() -> Scorer:
    """
    Calculates the percentage of unique STRIDE categories present in the model output
    (if parsable) that are NOT found in the target text (False Positives).
    Score = (FP STRIDE Count) / (Total STRIDE categories declared by model).
    If model output is unparsable or declares no STRIDE, FP percentage is 0%.
    """
    import json
    from typing import Any # Required by the helper

    async def score(state: TaskState, target: Target) -> Score:
        model_strides: set[str] = set()
        model_parse_error = "" # For detailed explanation
        parsed_model_successfully = False

        try:
            model_data = json.loads(state.output.completion)
            if isinstance(model_data, list):
                model_strides = _extract_stride_categories(model_data, "model output")
                parsed_model_successfully = True
                model_parse_error = "Model output parsed as JSON list."
                if not model_strides:
                    model_parse_error += " However, no STRIDE categories were extracted (e.g., empty items or wrong structure)."
            else:
                model_parse_error = "Model output is valid JSON but not a list. Cannot extract STRIDE categories."
        except json.JSONDecodeError as e:
            model_parse_error = f"Error parsing model output as JSON: {e}. Cannot extract STRIDE categories."

        # Target parsing for comparison
        target_strides: set[str] = set()
        target_parse_error = ""
        # explanation_suffix = "" # This was from old logic, replaced by more direct error messages
        try:
            target_data = json.loads(target.text)
            if isinstance(target_data, list):
                target_strides = _extract_stride_categories(target_data, "target text")
                if not target_strides:
                    target_parse_error = "Target parsed as JSON list, but no STRIDE categories extracted."
            else:
                # Target is malformed (not a list): all model STRIDEs are FPs against an empty target_strides set.
                target_parse_error = "Target text JSON is valid but not a list; treated as having zero target STRIDE categories for FP calculation."
        except json.JSONDecodeError as e:
            # If target is malformed, all model STRIDE categories are effectively FPs against an empty target_strides set.
            target_parse_error = f"Error parsing target text as JSON: {e}; treated as having zero target STRIDE categories for FP calculation."

        num_model_strides = len(model_strides)
        false_positive_strides = model_strides.difference(target_strides)
        fp_count = len(false_positive_strides)

        score_value: float
        score_explanation_prefix: str

        if num_model_strides == 0:
            score_value = 0.0 # 0 FPs out of 0 declared model strides
            score_explanation_prefix = f"Model declared 0 STRIDE categories ({model_parse_error if model_parse_error else 'Model output yielded no STRIDE categories'}). False Positive STRIDE percentage: 0.00% (0/0)."
        else:
            score_value = float(fp_count) / num_model_strides
            score_explanation_prefix = f"Percentage of model-declared STRIDE categories that are False Positives: {score_value*100:.2f}% ({fp_count}/{num_model_strides})."

        # For context in explanation
        identified_strides = model_strides.intersection(target_strides)
        # fn_strides = target_strides.difference(model_strides) # FN based on parsed model vs parsed target

        explanation_parts = [
            score_explanation_prefix,
            # f"Count of false positive STRIDE categories: {fp_count}. (This is the primary score)",
            f"Model parsing details: {model_parse_error if model_parse_error else 'Model parsed successfully and yielded STRIDE categories'}.",
            f"Model unique STRIDE (extracted for FP calc): {num_model_strides} {sorted(list(model_strides)) if model_strides else '[]'}.",
            f"Target parsing details: {target_parse_error if target_parse_error else 'Target parsed successfully and yielded STRIDE categories'}.",
            f"Target unique STRIDE (for context): {len(target_strides)} {sorted(list(target_strides)) if target_strides else '[]'}.",
            f"Contextual TPs (model vs target, both parsed): {len(identified_strides)} {sorted(list(identified_strides)) if identified_strides else '[]'}.",
            f"FPs details (contributing to score - model strides not in target strides): {fp_count} {sorted(list(false_positive_strides)) if false_positive_strides else '[]'}.",
            # f"Missing from model (FN): {len(fn_strides)} {sorted(list(fn_strides)) if fn_strides else '[]'}."
        ]
        # if explanation_suffix: # old logic
        #     explanation_parts.append(explanation_suffix)
        
        final_explanation = " ".join(explanation_parts)

        return Score(
            value=score_value, # Use the calculated percentage
            answer=state.output.completion,
            explanation=final_explanation
        )
    return score


# Helper functions for normalizing and extracting codes
def _normalize_cwe_id(cwe_str: Any) -> str | None:
    if not isinstance(cwe_str, str):
        return None
    normalized = cwe_str.strip().lower()
    # Basic check for "cwe-" prefix and digits, can be more stringent
    if re.fullmatch(r"cwe-\d+", normalized):
        return normalized
    # Allow just number to be prefixed with cwe-
    if re.fullmatch(r"\d+", normalized):
        return f"cwe-{normalized}"
    return None 

def _extract_owasp_code(owasp_item: Any) -> str | None:
    if not isinstance(owasp_item, str):
        return None
    # Regex to find AXX:YYYY pattern, case insensitive for input, then lowercase
    match = re.search(r"(A\d{2}:\d{4})", owasp_item, re.IGNORECASE)
    return match.group(1).lower() if match else None

def _extract_mitre_code(mitre_item: Any) -> str | None:
    if not isinstance(mitre_item, str):
        return None
    # Regex to find TXXXX or TXXXX.YYY pattern, case insensitive for input, then lowercase
    match = re.search(r"(T\d{4}(\.\d{3})?)", mitre_item, re.IGNORECASE)
    return match.group(1).lower() if match else None


@scorer(metrics=[mean(), stderr()])
def cwe_recall_scorer() -> Scorer:
    """
    Scores the percentage of unique target CWE IDs (e.g., "cwe-123") found as
    substrings in the model's output string (case-insensitive).

    The routine:
    - Parses target text as JSON list of threats, extracts and normalizes CWE IDs.
    - Normalizes model output to lowercase for case-insensitive substring search.
    - Calculates score: (Target CWEs found in model string) / (Total unique target CWEs).
    - Handles 0 target CWEs: 100% if model also appears empty of CWEs, else 0%.
    - Handles unparsable target/model for context.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion
        target_text_str = target.text
        
        target_cwes: set[str] = set()
        target_parse_error = ""
        raw_target_threats = None # Flag to distinguish parse failure vs. empty list

        try:
            parsed_json_target = json.loads(target_text_str)
            if not isinstance(parsed_json_target, list):
                target_parse_error = "Target text is valid JSON but not a list of threats."
                # raw_target_threats remains None, handled by 'target_parse_error and raw_target_threats is None'
            else:
                raw_target_threats = parsed_json_target # Mark that we had a list
                for threat in raw_target_threats:
                    if isinstance(threat, dict):
                        cwe_val = threat.get("cwe")
                        normalized_cwe = _normalize_cwe_id(cwe_val)
                        if normalized_cwe:
                            target_cwes.add(normalized_cwe)
                if not target_cwes and raw_target_threats is not None and not target_parse_error:
                    target_parse_error = "Target parsed as list, but no valid CWEs extracted (e.g. items lacked 'cwe' or format was incorrect)."

        except json.JSONDecodeError as e:
            target_parse_error = f"Error parsing target text as JSON: {e}"
            # raw_target_threats remains None

        num_target_cwes = len(target_cwes)
        normalized_model_output = model_output_str.lower()
        identified_cwes_in_model_string: set[str] = set()

        if num_target_cwes > 0:
            for cwe_id in target_cwes:
                if cwe_id in normalized_model_output: # cwe_id is already normalized
                    identified_cwes_in_model_string.add(cwe_id)
        
        num_identified = len(identified_cwes_in_model_string)
        score_value: float
        explanation_prefix: str

        # Case 1: Target was unparsable or not a list
        if raw_target_threats is None and target_parse_error:
            model_cwes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                    for th in m_threats:
                        if isinstance(th, dict):
                            norm_cwe = _normalize_cwe_id(th.get("cwe"))
                            if norm_cwe: model_cwes_context.add(norm_cwe)
            except: pass 
            score_value = 1.0 if not model_cwes_context else 0.0
            explanation_prefix = f"Target unparsable or not a list ({target_parse_error}). Model (contextually) had {len(model_cwes_context)} CWEs. Score: {score_value*100:.2f}%."
        # Case 2: Target parsed as list, but yielded 0 CWEs
        elif num_target_cwes == 0:
            model_cwes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                    for th in m_threats:
                        if isinstance(th, dict):
                            norm_cwe = _normalize_cwe_id(th.get("cwe"))
                            if norm_cwe: model_cwes_context.add(norm_cwe)
            except: pass
            score_value = 1.0 if not model_cwes_context else 0.0
            explanation_prefix = f"Target listed 0 CWEs ({target_parse_error if target_parse_error else 'successfully parsed, no CWEs found'}). Model (contextually) had {len(model_cwes_context)} CWEs. Score: {score_value*100:.2f}%."
        # Case 3: Target has CWEs
        else:
            score_value = num_identified / num_target_cwes
            explanation_prefix = f"Percentage of target CWEs found in model string: {score_value*100:.2f}% ({num_identified}/{num_target_cwes})."

        # Contextual analysis of model output for FPs
        model_cwes_for_fp_context = set()
        model_parse_error_context = ""
        try:
            model_threats_context = json.loads(model_output_str)
            if isinstance(model_threats_context, list):
                for m_threat in model_threats_context:
                    if isinstance(m_threat, dict):
                        norm_cwe = _normalize_cwe_id(m_threat.get("cwe"))
                        if norm_cwe: model_cwes_for_fp_context.add(norm_cwe)
                if not model_cwes_for_fp_context and model_threats_context:
                     model_parse_error_context = "Model parsed as list, but no valid CWEs extracted for FP context."
            else: 
                model_parse_error_context = "Model output valid JSON but not a list (for FP context)."
        except json.JSONDecodeError as e:
            model_parse_error_context = f"Model output not valid JSON (for FP context): {e}."
            
        contextual_fp_cwes = model_cwes_for_fp_context.difference(target_cwes)
        contextual_fn_cwes = target_cwes.difference(identified_cwes_in_model_string)

        explanation_parts = [
            explanation_prefix,
            f"Target unique CWEs (attempted extraction): {num_target_cwes} {sorted(list(target_cwes)) if target_cwes else '[]'}.",
            f"Identified in model string (TPs): {num_identified} {sorted(list(identified_cwes_in_model_string)) if identified_cwes_in_model_string else '[]'}.",
            f"Missing from model string (FNs based on string search): {len(contextual_fn_cwes)} {sorted(list(contextual_fn_cwes)) if contextual_fn_cwes else '[]'}."
        ]
        if target_parse_error: 
            explanation_parts.append(f"Target parsing/extraction note: {target_parse_error}.")
        
        explanation_parts.extend([
            f"Model unique CWEs (for FP context - {'parsed' if not model_parse_error_context else 'issue'}): {len(model_cwes_for_fp_context)} {sorted(list(model_cwes_for_fp_context)) if model_cwes_for_fp_context else '[]'}.",
            f"Contextual FPs (model CWEs not in target CWEs): {len(contextual_fp_cwes)} {sorted(list(contextual_fp_cwes)) if contextual_fp_cwes else '[]'}."
        ])
        if model_parse_error_context: 
            explanation_parts.append(f"Model parsing/extraction for FP context note: {model_parse_error_context}.")

        return Score(value=score_value, answer=model_output_str, explanation=" ".join(explanation_parts))
    return score


@scorer(metrics=[mean(), stderr()])
def owasp_recall_scorer() -> Scorer:
    """
    Scores the percentage of unique target OWASP Top 10 codes (e.g., "a01:2021")
    found as substrings in the model's output string (case-insensitive).

    The routine:
    - Parses target JSON, extracts OWASP items, normalizes to codes (e.g. "a01:2021").
    - Normalizes model output to lowercase for case-insensitive substring search.
    - Calculates score: (Target OWASP codes found) / (Total unique target OWASP codes).
    - Handles 0 target codes: 100% if model also appears empty of codes, else 0%.
    - Handles unparsable target/model for context.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion
        target_text_str = target.text
        
        target_owasp_codes: set[str] = set()
        target_parse_error = ""
        raw_target_threats = None # Flag to distinguish parse failure vs. empty list

        try:
            parsed_json_target = json.loads(target_text_str)
            if not isinstance(parsed_json_target, list):
                target_parse_error = "Target text is valid JSON but not a list of threats."
            else:
                raw_target_threats = parsed_json_target # Mark that we had a list
                for threat in raw_target_threats:
                    if isinstance(threat, dict):
                        frameworks = threat.get("frameworks")
                        if isinstance(frameworks, dict):
                            owasp_list = frameworks.get("owaspTop10")
                            if isinstance(owasp_list, list):
                                for item in owasp_list:
                                    code = _extract_owasp_code(item)
                                    if code: target_owasp_codes.add(code)
                if not target_owasp_codes and raw_target_threats is not None and not target_parse_error:
                    target_parse_error = "Target parsed as list, but no valid OWASP codes extracted."
        except json.JSONDecodeError as e:
            target_parse_error = f"Error parsing target text as JSON: {e}"

        num_target_codes = len(target_owasp_codes)
        normalized_model_output = model_output_str.lower()
        identified_codes_in_model_string: set[str] = set()

        if num_target_codes > 0:
            for code_id in target_owasp_codes: # Already normalized by _extract_owasp_code
                if code_id in normalized_model_output:
                    identified_codes_in_model_string.add(code_id)
        
        num_identified = len(identified_codes_in_model_string)
        score_value: float
        explanation_prefix: str

        if raw_target_threats is None and target_parse_error: # Target unparsable or not a list
            model_codes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                    for th in m_threats:
                        if isinstance(th,dict) and isinstance(th.get("frameworks"),dict):
                            o_list = th["frameworks"].get("owaspTop10")
                            if isinstance(o_list, list):
                                for item in o_list:
                                    c = _extract_owasp_code(item)
                                    if c: model_codes_context.add(c)
            except: pass
            score_value = 1.0 if not model_codes_context else 0.0
            explanation_prefix = f"Target unparsable or not a list ({target_parse_error}). Model (contextually) had {len(model_codes_context)} OWASP codes. Score: {score_value*100:.2f}%."
        elif num_target_codes == 0: # Target parsed, but 0 OWASP codes found
            model_codes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                     for th in m_threats:
                        if isinstance(th,dict) and isinstance(th.get("frameworks"),dict):
                            o_list = th["frameworks"].get("owaspTop10")
                            if isinstance(o_list, list):
                                for item in o_list:
                                    c = _extract_owasp_code(item)
                                    if c: model_codes_context.add(c)
            except: pass
            score_value = 1.0 if not model_codes_context else 0.0
            explanation_prefix = f"Target listed 0 OWASP codes ({target_parse_error if target_parse_error else 'successfully parsed, no codes found'}). Model (contextually) had {len(model_codes_context)} codes. Score: {score_value*100:.2f}%."
        else: # Target has OWASP codes
            score_value = num_identified / num_target_codes
            explanation_prefix = f"Percentage of target OWASP codes found in model string: {score_value*100:.2f}% ({num_identified}/{num_target_codes})."

        model_codes_for_fp_context = set()
        model_parse_error_context = ""
        try:
            model_threats_context = json.loads(model_output_str)
            if isinstance(model_threats_context, list):
                for m_threat in model_threats_context:
                    if isinstance(m_threat, dict) and isinstance(m_threat.get("frameworks"),dict):
                        o_list = m_threat["frameworks"].get("owaspTop10")
                        if isinstance(o_list, list):
                            for item in o_list:
                                c = _extract_owasp_code(item)
                                if c: model_codes_for_fp_context.add(c)
                if not model_codes_for_fp_context and model_threats_context:
                    model_parse_error_context = "Model parsed as list, but no OWASP codes extracted for FP context."
            else: 
                model_parse_error_context = "Model output valid JSON but not a list (for FP context)."
        except json.JSONDecodeError as e:
            model_parse_error_context = f"Model output not valid JSON (for FP context): {e}."

        contextual_fp_codes = model_codes_for_fp_context.difference(target_owasp_codes)
        contextual_fn_codes = target_owasp_codes.difference(identified_codes_in_model_string)

        explanation_parts = [
            explanation_prefix,
            f"Target unique OWASP codes (attempted extraction): {num_target_codes} {sorted(list(target_owasp_codes)) if target_owasp_codes else '[]'}.",
            f"Identified in model string (TPs): {num_identified} {sorted(list(identified_codes_in_model_string)) if identified_codes_in_model_string else '[]'}.",
            f"Missing from model string (FNs based on string search): {len(contextual_fn_codes)} {sorted(list(contextual_fn_codes)) if contextual_fn_codes else '[]'}."
        ]
        if target_parse_error: 
            explanation_parts.append(f"Target parsing/extraction note: {target_parse_error}.")

        explanation_parts.extend([
            f"Model unique OWASP codes (for FP context - {'parsed' if not model_parse_error_context else 'issue'}): {len(model_codes_for_fp_context)} {sorted(list(model_codes_for_fp_context)) if model_codes_for_fp_context else '[]'}.",
            f"Contextual FPs (model OWASP codes not in target OWASP codes): {len(contextual_fp_codes)} {sorted(list(contextual_fp_codes)) if contextual_fp_codes else '[]'}."
        ])
        if model_parse_error_context: 
            explanation_parts.append(f"Model parsing/extraction for FP context note: {model_parse_error_context}.")
        
        return Score(value=score_value, answer=model_output_str, explanation=" ".join(explanation_parts))
    return score


@scorer(metrics=[mean(), stderr()])
def mitre_recall_scorer() -> Scorer:
    """
    Scores the percentage of unique target MITRE ATT&CK codes (e.g., "t1078.004")
    found as substrings in the model's output string (case-insensitive).

    The routine:
    - Parses target JSON, extracts MITRE items, normalizes to codes (e.g. "t1234.001").
    - Normalizes model output to lowercase for case-insensitive substring search.
    - Calculates score: (Target MITRE codes found) / (Total unique target MITRE codes).
    - Handles 0 target codes: 100% if model also appears empty of codes, else 0%.
    - Handles unparsable target/model for context.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        model_output_str = state.output.completion
        target_text_str = target.text
        
        target_mitre_codes: set[str] = set()
        target_parse_error = ""
        raw_target_threats = None # Flag to distinguish parse failure vs. empty list

        try:
            parsed_json_target = json.loads(target_text_str)
            if not isinstance(parsed_json_target, list):
                target_parse_error = "Target text is valid JSON but not a list of threats."
            else:
                raw_target_threats = parsed_json_target # Mark that we had a list
                for threat in raw_target_threats:
                    if isinstance(threat, dict):
                        frameworks = threat.get("frameworks")
                        if isinstance(frameworks, dict):
                            mitre_list = frameworks.get("mitreATT&CK")
                            if isinstance(mitre_list, list):
                                for item in mitre_list:
                                    code = _extract_mitre_code(item)
                                    if code: target_mitre_codes.add(code)
                if not target_mitre_codes and raw_target_threats is not None and not target_parse_error:
                    target_parse_error = "Target parsed as list, but no valid MITRE codes extracted."
        except json.JSONDecodeError as e:
            target_parse_error = f"Error parsing target text as JSON: {e}"

        num_target_codes = len(target_mitre_codes)
        normalized_model_output = model_output_str.lower()
        identified_codes_in_model_string: set[str] = set()

        if num_target_codes > 0:
            for code_id in target_mitre_codes: # Already normalized by _extract_mitre_code
                if code_id in normalized_model_output:
                    identified_codes_in_model_string.add(code_id)
        
        num_identified = len(identified_codes_in_model_string)
        score_value: float
        explanation_prefix: str
        
        if raw_target_threats is None and target_parse_error: # Target unparsable or not a list
            model_codes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                    for th in m_threats:
                        if isinstance(th,dict) and isinstance(th.get("frameworks"),dict):
                            m_list = th["frameworks"].get("mitreATT&CK")
                            if isinstance(m_list, list):
                                for item in m_list:
                                    c = _extract_mitre_code(item)
                                    if c: model_codes_context.add(c)
            except: pass
            score_value = 1.0 if not model_codes_context else 0.0
            explanation_prefix = f"Target unparsable or not a list ({target_parse_error}). Model (contextually) had {len(model_codes_context)} MITRE codes. Score: {score_value*100:.2f}%."
        elif num_target_codes == 0: # Target parsed, but 0 MITRE codes found
            model_codes_context = set()
            try:
                m_threats = json.loads(model_output_str)
                if isinstance(m_threats, list):
                    for th in m_threats:
                        if isinstance(th,dict) and isinstance(th.get("frameworks"),dict):
                            m_list = th["frameworks"].get("mitreATT&CK")
                            if isinstance(m_list, list):
                                for item in m_list:
                                    c = _extract_mitre_code(item)
                                    if c: model_codes_context.add(c)
            except: pass
            score_value = 1.0 if not model_codes_context else 0.0
            explanation_prefix = f"Target listed 0 MITRE codes ({target_parse_error if target_parse_error else 'successfully parsed, no codes found'}). Model (contextually) had {len(model_codes_context)} codes. Score: {score_value*100:.2f}%."
        else: # Target has MITRE codes
            score_value = num_identified / num_target_codes
            explanation_prefix = f"Percentage of target MITRE codes found in model string: {score_value*100:.2f}% ({num_identified}/{num_target_codes})."

        model_codes_for_fp_context = set()
        model_parse_error_context = ""
        try:
            model_threats_context = json.loads(model_output_str)
            if isinstance(model_threats_context, list):
                for m_threat in model_threats_context:
                    if isinstance(m_threat, dict) and isinstance(m_threat.get("frameworks"),dict):
                        m_list = m_threat["frameworks"].get("mitreATT&CK")
                        if isinstance(m_list, list):
                            for item in m_list:
                                c = _extract_mitre_code(item)
                                if c: model_codes_for_fp_context.add(c)
                if not model_codes_for_fp_context and model_threats_context:
                    model_parse_error_context = "Model parsed as list, but no MITRE codes extracted for FP context."
            else: 
                model_parse_error_context = "Model output valid JSON but not a list (for FP context)."
        except json.JSONDecodeError as e:
            model_parse_error_context = f"Model output not valid JSON (for FP context): {e}."

        contextual_fp_codes = model_codes_for_fp_context.difference(target_mitre_codes)
        contextual_fn_codes = target_mitre_codes.difference(identified_codes_in_model_string)

        explanation_parts = [
            explanation_prefix,
            f"Target unique MITRE codes (attempted extraction): {num_target_codes} {sorted(list(target_mitre_codes)) if target_mitre_codes else '[]'}.",
            f"Identified in model string (TPs): {num_identified} {sorted(list(identified_codes_in_model_string)) if identified_codes_in_model_string else '[]'}.",
            f"Missing from model string (FNs based on string search): {len(contextual_fn_codes)} {sorted(list(contextual_fn_codes)) if contextual_fn_codes else '[]'}."
        ]
        if target_parse_error: 
            explanation_parts.append(f"Target parsing/extraction note: {target_parse_error}.")

        explanation_parts.extend([
            f"Model unique MITRE codes (for FP context - {'parsed' if not model_parse_error_context else 'issue'}): {len(model_codes_for_fp_context)} {sorted(list(model_codes_for_fp_context)) if model_codes_for_fp_context else '[]'}.",
            f"Contextual FPs (model MITRE codes not in target MITRE codes): {len(contextual_fp_codes)} {sorted(list(contextual_fp_codes)) if contextual_fp_codes else '[]'}."
        ])
        if model_parse_error_context: 
            explanation_parts.append(f"Model parsing/extraction for FP context note: {model_parse_error_context}.")

        return Score(value=score_value, answer=model_output_str, explanation=" ".join(explanation_parts))
    return score


@scorer(metrics=[mean(), stderr()])
def high_likelihood_threat_count_scorer() -> Scorer:
    """
    Counts the number of threats in the model output with likelihood 'High'.

    The routine:
    - Parses model output as a JSON list of threat objects.
    - Iterates through the threats and counts how many have 'likelihood' set to 'High' (case-insensitive).
    - The final score is the total count.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        try:
            model_threats = json.loads(state.output.completion)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=state.output.completion,
                explanation=f"Error parsing JSON: {e}",
            )

        if not isinstance(model_threats, list):
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Model output is not a list of threats.",
            )

        high_likelihood_count = 0
        for threat in model_threats:
            if isinstance(threat, dict):
                likelihood = threat.get("likelihood")
                # Case-insensitive check for 'High'
                if isinstance(likelihood, str) and likelihood.lower() == "high":
                    high_likelihood_count += 1

        explanation_str = f"Found {high_likelihood_count} threat(s) with likelihood 'High'."

        return Score(
            value=float(high_likelihood_count),
            answer=state.output.completion,
            explanation=explanation_str,
        )

    return score


@scorer(metrics=[mean(), stderr()])
def medium_likelihood_threat_count_scorer() -> Scorer:
    """
    Counts the number of threats in the model output with likelihood 'Medium'.

    The routine:
    - Parses model output as a JSON list of threat objects.
    - Iterates through the threats and counts how many have 'likelihood' set to 'Medium' (case-insensitive).
    - The final score is the total count.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        try:
            model_threats = json.loads(state.output.completion)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=state.output.completion,
                explanation=f"Error parsing JSON: {e}",
            )

        if not isinstance(model_threats, list):
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Model output is not a list of threats.",
            )

        medium_likelihood_count = 0
        for threat in model_threats:
            if isinstance(threat, dict):
                likelihood = threat.get("likelihood")
                # Case-insensitive check for 'Medium'
                if isinstance(likelihood, str) and likelihood.lower() == "medium":
                    medium_likelihood_count += 1

        explanation_str = f"Found {medium_likelihood_count} threat(s) with likelihood 'Medium'."

        return Score(
            value=float(medium_likelihood_count),
            answer=state.output.completion,
            explanation=explanation_str,
        )

    return score


@scorer(metrics=[mean(), stderr()])
def low_likelihood_threat_count_scorer() -> Scorer:
    """
    Counts the number of threats in the model output with likelihood 'Low'.

    The routine:
    - Parses model output as a JSON list of threat objects.
    - Iterates through the threats and counts how many have 'likelihood' set to 'Low' (case-insensitive).
    - The final score is the total count.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        try:
            model_threats = json.loads(state.output.completion)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=state.output.completion,
                explanation=f"Error parsing JSON: {e}",
            )

        if not isinstance(model_threats, list):
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Model output is not a list of threats.",
            )

        low_likelihood_count = 0
        for threat in model_threats:
            if isinstance(threat, dict):
                likelihood = threat.get("likelihood")
                # Case-insensitive check for 'Low'
                if isinstance(likelihood, str) and likelihood.lower() == "low":
                    low_likelihood_count += 1

        explanation_str = f"Found {low_likelihood_count} threat(s) with likelihood 'Low'."

        return Score(
            value=float(low_likelihood_count),
            answer=state.output.completion,
            explanation=explanation_str,
        )

    return score


@scorer(metrics=[mean(), stderr()])
def false_positive_threat_count_scorer() -> Scorer:
    """
    Counts threats in model output not found in target output based on CWE, OWASP, or MITRE IDs.

    A model threat is a false positive if none of its specified identifiers (CWE,
    OWASP Top 10 items, MITRE ATT&CK items) match any corresponding identifiers
    found in the entire set of target threats.

    The routine:
    - Parses model output and target text as JSON lists of threat objects.
    - Collects all unique CWE, OWASP Top 10, and MITRE ATT&CK identifiers from target threats.
    - Iterates through model output threats.
    - For each model threat, checks if it matches any target threat based on CWE, any OWASP ID, or any MITRE ID.
    - If no such match is found, the model threat is counted as a false positive.
    - The final score is the total count of false positive threats.
    """
    import json

    async def score(state: TaskState, target: Target) -> Score:
        try:
            model_threats = json.loads(state.output.completion)
            target_threats = json.loads(target.text)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=state.output.completion,
                explanation=f"Error parsing JSON: {e}",
            )

        if not isinstance(model_threats, list):
            # If model output is not a list, it's hard to count FPs meaningfully.
            # Depending on strictness, this could be all threats if target is non-empty, or 0.
            # For now, returning 0 if model output is malformed.
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Model output is not a list of threats, cannot determine false positives.",
            )
        
        if not isinstance(target_threats, list):
            # If target is not a list, all model threats could be considered FPs.
            # For now, if target is malformed, count all model items as FP if model_threats is a list.
            # If model_threats is also not a list, the above check handles it.
            fp_count = len(model_threats) if isinstance(model_threats, list) else 0
            return Score(
                value=float(fp_count),
                answer=state.output.completion,
                explanation=f"Target output is not a list of threats. All {fp_count} model threats counted as false positives."
            )

        # Collect all unique identifiers from target threats
        all_target_cwes = set()
        all_target_owasp_ids = set()
        all_target_mitre_ids = set()

        for t_threat in target_threats:
            if not isinstance(t_threat, dict):
                continue
            if t_threat.get("cwe"):
                all_target_cwes.add(t_threat["cwe"])
            frameworks = t_threat.get("frameworks")
            if isinstance(frameworks, dict):
                owasp_list = frameworks.get("owaspTop10")
                if isinstance(owasp_list, list):
                    for item in owasp_list:
                        if item: all_target_owasp_ids.add(item)
                mitre_list = frameworks.get("mitreATT&CK")
                if isinstance(mitre_list, list):
                    for item in mitre_list:
                        if item: all_target_mitre_ids.add(item)

        false_positive_count = 0
        false_positive_details = []

        for m_threat in model_threats:
            if not isinstance(m_threat, dict):
                # Consider non-dict items in model output as FPs if we expect dicts
                false_positive_count +=1
                false_positive_details.append("Non-dictionary item in model output")
                continue

            matched_to_target = False

            # Check CWE
            model_cwe = m_threat.get("cwe")
            if model_cwe and model_cwe in all_target_cwes:
                matched_to_target = True

            # Check OWASP if not already matched
            if not matched_to_target:
                model_frameworks = m_threat.get("frameworks")
                if isinstance(model_frameworks, dict):
                    model_owasp_list = model_frameworks.get("owaspTop10")
                    if isinstance(model_owasp_list, list):
                        for item in model_owasp_list:
                            if item and item in all_target_owasp_ids:
                                matched_to_target = True
                                break
            
            # Check MITRE if not already matched
            if not matched_to_target:
                model_frameworks = m_threat.get("frameworks") # Re-get in case only mitre exists
                if isinstance(model_frameworks, dict):
                    model_mitre_list = model_frameworks.get("mitreATT&CK")
                    if isinstance(model_mitre_list, list):
                        for item in model_mitre_list:
                            if item and item in all_target_mitre_ids:
                                matched_to_target = True
                                break
            
            if not matched_to_target:
                false_positive_count += 1
                threat_name = m_threat.get("threatName", "Unnamed Threat")
                # Gather identifiers of the FP model threat for clarity
                fp_identifiers = []
                if model_cwe: fp_identifiers.append(f"CWE: {model_cwe}")
                # Simplified collection for OWASP/MITRE for brevity in this part
                if isinstance(m_threat.get("frameworks", {}).get("owaspTop10"), list) and m_threat["frameworks"]["owaspTop10"]:
                    fp_identifiers.append(f"OWASP: {m_threat['frameworks']['owaspTop10']}")
                if isinstance(m_threat.get("frameworks", {}).get("mitreATT&CK"), list) and m_threat["frameworks"]["mitreATT&CK"]:
                    fp_identifiers.append(f"MITRE: {m_threat['frameworks']['mitreATT&CK']}")
                
                detail_str = f"'{threat_name}'"
                if fp_identifiers:
                    detail_str += f" (IDs: {', '.join(fp_identifiers)})"
                elif not model_cwe: # if no identifiers at all
                     detail_str += " (No standard identifiers provided)"
                false_positive_details.append(detail_str)

        explanation_str = f"Found {false_positive_count} false positive threat(s) in model output."
        if false_positive_details:
            explanation_str += " Details: " + "; ".join(false_positive_details)

        return Score(
            value=float(false_positive_count),
            answer=state.output.completion,
            explanation=explanation_str,
        )

    return score

