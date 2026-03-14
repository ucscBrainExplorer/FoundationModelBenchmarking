#!/usr/bin/env python3
"""
Map readable cell type names to CL ontology term IDs using an OBO file,
with LLM-assisted fallback for unmatched names.

Enables biologists to feed free-text predictions into evaluate.py.

Usage:
  python3 annotate_cl_terms.py \
    --obo reference_data/cl.obo \
    --input biologist_predictions.tsv \
    --name_col cell_type \
    --output annotated_predictions.tsv
"""

import argparse
import os
import re
import sys
import time
import warnings
import pandas as pd


class FatalAPIError(Exception):
    """Raised when an API call fails with an unrecoverable error (e.g. no credits, bad key)."""
    pass


def _is_retryable(err_str):
    """Return True if the error is transient and worth retrying (e.g. overload, rate limit)."""
    return any(k in err_str for k in ('529', 'overloaded', 'rate_limit', '429', 'timeout', 'service_unavailable'))

from obo_parser import parse_obo_names


def build_parser():
    parser = argparse.ArgumentParser(
        description="Map readable cell type names to CL ontology term IDs"
    )
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--input", type=str, required=True,
                        help="Input TSV with readable cell type names")
    parser.add_argument("--output", type=str, required=True,
                        help="Output TSV (input + added cell_type_ontology_term_id column)")
    parser.add_argument("--name_col", type=str, default="cell_type",
                        help="Column containing readable cell type names (default: cell_type)")
    return parser


def parse_obo_synonyms(obo_path):
    """Parse OBO file and return {lowercase_synonym: CL_id} dict."""
    synonym_map = {}
    current_id = None
    in_term = False

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
            elif line.startswith('[') and line.endswith(']'):
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('synonym:') and current_id:
                    # Parse synonym line: synonym: "text" EXACT|RELATED|... []
                    match = re.match(r'synonym:\s*"([^"]+)"', line)
                    if match:
                        syn_text = match.group(1).lower()
                        # Don't overwrite if already mapped (prefer first occurrence)
                        if syn_text not in synonym_map:
                            synonym_map[syn_text] = current_id

    return synonym_map


def fuzzy_normalize(name):
    """Normalize a cell type name for fuzzy matching.

    Strips whitespace, removes hyphens/underscores, strips trailing 's' (plurals),
    and lowercases.
    """
    name = name.lower().strip()
    name = name.replace('-', '').replace('_', '')
    # Strip trailing 's' for simple plural handling (but not if the name is very short)
    if len(name) > 3 and name.endswith('s'):
        name = name[:-1]
    return name


def query_llm_mapping(name, cl_terms_subset, api="claude", paper_context=None):
    """Query an LLM to map a cell type name to a CL term.

    Args:
        name: The unmatched cell type name
        cl_terms_subset: Dict of {CL_id: canonical_name} (candidate terms)
        api: 'claude' or 'gemini'
        paper_context: Optional paper title/abstract to help identify novel cell types.

    Returns:
        CL term ID string, or None if the LLM couldn't determine a match
    """
    # Build a condensed list of candidates
    candidates = "\n".join(f"  {cl_id}: {cl_name}"
                           for cl_id, cl_name in sorted(cl_terms_subset.items()))

    context_line = (
        f"This label comes from the following study: {paper_context}\n"
        if paper_context else ""
    )
    prompt = (
        f"Given the cell type name \"{name}\", which Cell Ontology (CL) term "
        f"is the best match from this list?\n\n{candidates}\n\n"
        f"{context_line}"
        f"Rules:\n"
        f"- HARD RULE: If the label contains a marker gene (e.g. VIP, SST, PV, SNCG, "
        f"LAMP5, PVALB, CCK, NPY, CR, CB), you MUST choose the term that includes that "
        f"marker gene name. Do NOT choose a broader lineage or origin term (e.g. 'caudal "
        f"ganglionic eminence derived interneuron') when a marker-specific term (e.g. "
        f"'VIP GABAergic interneuron', 'sst GABAergic interneuron') is available in the "
        f"candidate list. The marker gene takes priority over lineage origin.\n"
        f"- HARD RULE: If the label or description contains a distinctive biological word "
        f"(e.g. 'tripotential', 'corticothalamic', 'intratelencephalic') that appears "
        f"verbatim in a candidate term's name, strongly prefer that term over a generic "
        f"alternative (e.g. 'neural progenitor cell', 'neuron').\n"
        f"- Prefer the most specific term that captures ALL meaningful parts of the label "
        f"(e.g. cell class, brain region, maturity state, origin).\n"
        f"- Do NOT choose a generic term (e.g. 'neuron', 'immature neuron') if the label "
        f"encodes additional specificity that the generic term ignores.\n"
        f"- Do NOT choose a term that adds specificity absent from the label. "
        f"For example, if the label says 'IT neuron' without mentioning a layer, "
        f"do NOT pick 'L2/3 IT neuron' or 'L5 IT neuron' — reply NONE instead and let the "
        f"ancestor fallback find the right parent term.\n"
        f"- Only select a term if you are confident it is biologically correct.\n"
        f"- Novel or recently described cell types may not have a CL term — reply NONE.\n"
        f"- If no term captures the full specificity of the label, reply NONE.\n"
        f"Reply with ONLY the CL term ID (e.g. CL:0000540) or NONE, nothing else."
    )

    try:
        if api == "claude":
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text.strip()
        elif api == "gemini":
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            answer = ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
        else:
            return None

        # Extract CL ID from response
        cl_match = re.search(r'CL:\d+', answer)
        if cl_match and cl_match.group() in cl_terms_subset:
            return cl_match.group()
        if answer.upper() == "NONE":
            return "NONE"  # explicit veto — distinguishable from API failure (None)
        return None

    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ('credit', 'billing', 'unauthorized', 'invalid_api_key', 'permission')):
            raise FatalAPIError(f"{api} API fatal error: {e}")
        if _is_retryable(err_str):
            for attempt in range(2):
                wait = 5 * (attempt + 1)
                print(f"    Warning: {api} overloaded, retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                try:
                    if api == "claude":
                        import anthropic
                        client = anthropic.Anthropic()
                        response = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=50,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.content[0].text.strip()
                    elif api == "gemini":
                        from google import genai as google_genai
                        client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                        return ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
                except Exception:
                    pass
        print(f"    Warning: {api} API call failed: {e}")
        return None


def query_llm_ancestor(name, cl_terms_subset, api="claude", paper_context=None):
    """Query an LLM for the closest ancestor CL term when exact mapping failed.

    Uses a relaxed prompt that accepts less specific terms — dropping maturity
    state, sub-region, or origin qualifiers — to find the best available
    ontology anchor.

    Args:
        name: The unmatched cell type label.
        cl_terms_subset: Dict of {CL_id: canonical_name} (candidate terms).
        api: 'claude' or 'gemini'.
        paper_context: Optional paper context string.

    Returns:
        CL term ID string, or None if no reasonable ancestor found.
    """
    candidates = "\n".join(f"  {cl_id}: {cl_name}"
                           for cl_id, cl_name in sorted(cl_terms_subset.items()))
    context_line = (
        f"This label comes from the following study: {paper_context}\n"
        if paper_context else ""
    )
    prompt = (
        f"The cell type label \"{name}\" could not be matched to a specific CL term. "
        f"Find the closest ancestor (parent) CL term from this list that is biologically "
        f"valid, even if less specific. You may ignore maturity qualifiers (e.g. 'immature', "
        f"'newborn'), sub-region qualifiers, or origin qualifiers to find the best match. "
        f"For example, 'immature intratelencephalic excitatory neuron' → "
        f"'intratelencephalic-projecting glutamatergic cortical neuron'.\n\n"
        f"{candidates}\n\n"
        f"{context_line}"
        f"Rules:\n"
        f"- Choose the most specific term that is still biologically correct.\n"
        f"- Do NOT add specificity absent from the label. If the label does not name a "
        f"specific brain region (e.g. 'primary motor cortex', 'striatum', 'medulla', "
        f"'L5') or a specific organ (e.g. 'kidney', 'liver', 'lung', 'heart'), "
        f"do NOT choose a term restricted to that region or organ.\n"
        f"- Do NOT choose 'cell' or 'neuron' alone — these are too generic.\n"
        f"- Do NOT choose a mature cell type for a label that indicates a progenitor, "
        f"precursor, or intermediate progenitor cell (IPC). A progenitor label must map "
        f"to a progenitor or stem cell CL term, not its downstream progeny.\n"
        f"- ALWAYS prefer species-agnostic terms over species-specific ones. "
        f"If a term ending in '(Homo sapiens)', '(Mus musculus)', or any other species "
        f"name is in the list, check whether a species-agnostic parent term also exists "
        f"in the list and use that instead. Only use a species-specific term if no "
        f"species-agnostic equivalent is present in the candidate list.\n"
        f"- If no reasonable ancestor exists, reply NONE.\n"
        f"Reply with ONLY the CL term ID (e.g. CL:0000540) or NONE, nothing else."
    )

    try:
        if api == "claude":
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text.strip()
        elif api == "gemini":
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            answer = ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
        else:
            return None

        cl_match = re.search(r'CL:\d+', answer)
        if cl_match and cl_match.group() in cl_terms_subset:
            return cl_match.group()
        if answer.upper() == "NONE":
            return "NONE"  # explicit veto — distinguishable from API failure (None)
        return None

    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ('credit', 'billing', 'unauthorized', 'invalid_api_key', 'permission')):
            raise FatalAPIError(f"{api} API fatal error: {e}")
        if _is_retryable(err_str):
            for attempt in range(2):
                wait = 5 * (attempt + 1)
                print(f"    Warning: {api} overloaded, retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                try:
                    if api == "claude":
                        import anthropic
                        client = anthropic.Anthropic()
                        response = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=50,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.content[0].text.strip()
                    elif api == "gemini":
                        from google import genai as google_genai
                        client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                        return ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
                except Exception:
                    pass
        print(f"    Warning: {api} API call failed: {e}")
        return None


def query_llm_label(name, api="claude", paper_context=None):
    """Query an LLM to generate a plain English description of a cell type label.

    Args:
        name: The cell type label (e.g. 'IN-NCx_dGE-Immature')
        api: 'claude' or 'gemini'
        paper_context: Optional title/abstract of the source paper to help decode
                       novel or dataset-specific labels.

    Returns:
        Plain English description string, or None on failure.
    """
    context_line = (
        f"The label comes from this study: {paper_context} "
        if paper_context else ""
    )
    prompt = (
        f"You are an expert in single-cell transcriptomics and cell biology. "
        f"The following label comes from a scRNA-seq dataset and may use abbreviated "
        f"or shorthand nomenclature common in neuroscience or developmental biology "
        f"(e.g. 'RG-oRG' = outer radial glial cell, 'EN-L2_3-IT' = excitatory "
        f"intratelencephalic neuron of cortical layer 2/3, 'IN-dLGE' = inhibitory "
        f"neuron from dorsal lateral ganglionic eminence). "
        f"{context_line}"
        f"If you are confident you recognise the cell type, return a concise plain "
        f"English name (no longer than 10 words, no trailing punctuation). "
        f"If you are not confident, return the label exactly as given — do NOT guess. "
        f"Label: \"{name}\""
    )

    try:
        if api == "claude":
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        elif api == "gemini":
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
        else:
            return None

    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ('credit', 'billing', 'unauthorized', 'invalid_api_key', 'permission')):
            raise FatalAPIError(f"{api} API fatal error: {e}")
        return None


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Parse OBO -> exact name dict and synonym dict
    print(f"Parsing OBO file {args.obo}...")
    cl_names = parse_obo_names(args.obo)  # {CL_id: canonical_name}
    print(f"  Parsed {len(cl_names)} ontology terms")

    # Build reverse lookup: {lowercase_name: CL_id}
    name_to_id = {name.lower(): cl_id for cl_id, name in cl_names.items()}

    # Parse synonyms
    synonym_to_id = parse_obo_synonyms(args.obo)
    print(f"  Parsed {len(synonym_to_id)} synonyms")

    # Build fuzzy-normalized lookups
    fuzzy_name_to_id = {fuzzy_normalize(name): cl_id for cl_id, name in cl_names.items()}
    fuzzy_synonym_to_id = {fuzzy_normalize(syn): cl_id for syn, cl_id in synonym_to_id.items()}

    # 2. Read input
    print(f"Loading input from {args.input}...")
    df = pd.read_csv(args.input, sep='\t')
    if args.name_col not in df.columns:
        print(f"Error: column '{args.name_col}' not found in input file.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"  Loaded {len(df)} rows")

    # 3. Get unique cell type names to map
    unique_names = df[args.name_col].dropna().unique().tolist()
    print(f"  {len(unique_names)} unique cell type names to map")

    # 4. Match each unique name
    mapping = {}  # {name: CL_id}
    match_method = {}  # {name: method_used}

    exact_count = 0
    synonym_count = 0
    fuzzy_count = 0
    unmatched_names = []

    for name in unique_names:
        name_lower = name.lower().strip()

        # Step 1: Exact match (case-insensitive)
        if name_lower in name_to_id:
            mapping[name] = name_to_id[name_lower]
            match_method[name] = 'exact'
            exact_count += 1
            continue

        # Step 2: Synonym match
        if name_lower in synonym_to_id:
            mapping[name] = synonym_to_id[name_lower]
            match_method[name] = 'synonym'
            synonym_count += 1
            continue

        # Step 2.5: Fuzzy normalization
        fuzzy = fuzzy_normalize(name)
        if fuzzy in fuzzy_name_to_id:
            mapping[name] = fuzzy_name_to_id[fuzzy]
            match_method[name] = 'fuzzy'
            fuzzy_count += 1
            continue
        if fuzzy in fuzzy_synonym_to_id:
            mapping[name] = fuzzy_synonym_to_id[fuzzy]
            match_method[name] = 'fuzzy_synonym'
            fuzzy_count += 1
            continue

        # Step 3: LLM fallback needed
        unmatched_names.append(name)

    print(f"\n  Exact matches: {exact_count}")
    print(f"  Synonym matches: {synonym_count}")
    print(f"  Fuzzy matches: {fuzzy_count}")
    print(f"  Unmatched (need LLM): {len(unmatched_names)}")

    # Step 3: LLM-assisted mapping for remaining unmatched names
    llm_consensus_count = 0
    user_resolved_count = 0
    unresolved_count = 0

    if unmatched_names:
        # Check for API keys
        has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
        has_gemini = bool(os.environ.get('GOOGLE_API_KEY'))

        if not has_anthropic and not has_gemini:
            print("\n  Warning: No API keys found (ANTHROPIC_API_KEY, GOOGLE_API_KEY).")
            print("  Unmatched names will be left without CL term IDs.")
            for name in unmatched_names:
                mapping[name] = ''
                match_method[name] = 'unresolved'
                unresolved_count += 1
        else:
            print(f"\n  Running LLM-assisted mapping for {len(unmatched_names)} names...")
            for name in unmatched_names:
                print(f"\n  Mapping: \"{name}\"")

                claude_result = None
                gemini_result = None

                if has_anthropic:
                    claude_result = query_llm_mapping(name, cl_names, api="claude")
                    if claude_result:
                        print(f"    Claude suggests: {claude_result} ({cl_names.get(claude_result, '?')})")

                if has_gemini:
                    gemini_result = query_llm_mapping(name, cl_names, api="gemini")
                    if gemini_result:
                        print(f"    Gemini suggests: {gemini_result} ({cl_names.get(gemini_result, '?')})")

                # If both agree, auto-accept
                if claude_result and gemini_result and claude_result == gemini_result:
                    mapping[name] = claude_result
                    match_method[name] = 'llm_consensus'
                    llm_consensus_count += 1
                    print(f"    -> Auto-accepted (both LLMs agree): {claude_result}")
                elif claude_result or gemini_result:
                    # LLMs disagree or only one responded -> interactive prompt
                    print(f"\n    LLMs disagree for \"{name}\":")
                    options = []
                    if claude_result:
                        options.append(('c', claude_result, f"Claude: {claude_result} ({cl_names.get(claude_result, '?')})"))
                    if gemini_result:
                        options.append(('o', gemini_result, f"Gemini: {gemini_result} ({cl_names.get(gemini_result, '?')})"))
                    options.append(('s', '', "Skip (leave unmapped)"))
                    options.append(('m', None, "Manual (type CL ID)"))

                    for key, _, desc in options:
                        print(f"      [{key}] {desc}")

                    try:
                        choice = input("    Your choice: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = 's'

                    if choice == 'm':
                        try:
                            custom_id = input("    Enter CL ID (e.g. CL:0000540): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            custom_id = ''
                        if custom_id and re.match(r'CL:\d+', custom_id):
                            mapping[name] = custom_id
                            match_method[name] = 'user_manual'
                            user_resolved_count += 1
                        else:
                            mapping[name] = ''
                            match_method[name] = 'unresolved'
                            unresolved_count += 1
                    elif choice == 's':
                        mapping[name] = ''
                        match_method[name] = 'unresolved'
                        unresolved_count += 1
                    elif choice == 'c' and claude_result:
                        mapping[name] = claude_result
                        match_method[name] = 'user_selected_claude'
                        user_resolved_count += 1
                    elif choice == 'o' and gemini_result:
                        mapping[name] = gemini_result
                        match_method[name] = 'user_selected_gemini'
                        user_resolved_count += 1
                    else:
                        # Default to first available suggestion
                        first = claude_result or gemini_result
                        if first:
                            mapping[name] = first
                            match_method[name] = 'user_default'
                            user_resolved_count += 1
                        else:
                            mapping[name] = ''
                            match_method[name] = 'unresolved'
                            unresolved_count += 1
                else:
                    # Neither LLM could find a match
                    print(f"    -> No LLM suggestions available")
                    mapping[name] = ''
                    match_method[name] = 'unresolved'
                    unresolved_count += 1

    # 5. Apply mapping to DataFrame
    df['cell_type_ontology_term_id'] = df[args.name_col].map(mapping).fillna('')

    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, sep='\t', index=False)
    print(f"\nSaved annotated output to {args.output}")

    # Summary
    print(f"\n{'=' * 60}")
    print("ANNOTATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique names:     {len(unique_names)}")
    print(f"  Exact match:          {exact_count}")
    print(f"  Synonym match:        {synonym_count}")
    print(f"  Fuzzy match:          {fuzzy_count}")
    print(f"  LLM consensus:        {llm_consensus_count}")
    print(f"  User resolved:        {user_resolved_count}")
    print(f"  Unresolved:           {unresolved_count}")
    mapped_total = exact_count + synonym_count + fuzzy_count + llm_consensus_count + user_resolved_count
    print(f"  Total mapped:         {mapped_total}/{len(unique_names)}")
    print(f"{'=' * 60}")

    if unresolved_count > 0:
        unresolved_names = [n for n, m in match_method.items() if m == 'unresolved']
        print(f"\nUnresolved names ({unresolved_count}):")
        for n in sorted(unresolved_names):
            print(f"  - {n}")


if __name__ == "__main__":
    main()
