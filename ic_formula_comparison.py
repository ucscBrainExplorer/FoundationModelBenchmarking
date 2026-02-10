"""
Compare different Intrinsic IC formulas on neuroscience cell types from CL ontology.

Formulas:
1. Seco (2004):      IC = 1 - log(desc(t)+1) / log(N)
   Pure descendant count. Simple but ignores depth.

2. Zhou (2008):       IC = k * Seco_IC + (1-k) * log(depth(t)+1) / log(max_depth+1)
   Combines descendant count with depth. k controls the blend.
   A deep node with few descendants gets high IC.

3. Sanchez (2011):    IC = -log( (leaves(t)/desc(t) + 1) / (max_leaves + 1) )
   Uses the ratio of leaf descendants to total descendants.
   Captures "specificity" — a node where most descendants are leaves is specific.

4. Depth-only:        IC = depth(t) / max_depth
   Pure structural depth. Simple baseline.

For all: IC(root) ≈ 0, IC(leaf) ≈ 1 (or maximal).
"""

import math
from collections import defaultdict, deque

def parse_obo(path):
    terms = {}
    current_term = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                current_term = {}
            elif line == '' or line.startswith('[') and line != '[Term]':
                if current_term and 'id' in current_term:
                    terms[current_term['id']] = current_term
                current_term = None if line.startswith('[') else current_term
            elif current_term is not None:
                if line.startswith('id: '):
                    current_term['id'] = line[4:].strip()
                elif line.startswith('name: '):
                    current_term['name'] = line[6:].strip()
                elif line.startswith('is_a: '):
                    if 'is_a' not in current_term:
                        current_term['is_a'] = []
                    parent_id = line[6:].split('!')[0].split('{')[0].strip()
                    current_term['is_a'].append(parent_id)
                elif line.startswith('is_obsolete: true'):
                    current_term['obsolete'] = True
        if current_term and 'id' in current_term:
            terms[current_term['id']] = current_term
    return {k: v for k, v in terms.items() if not v.get('obsolete', False)}

def build_graph(terms):
    parents = defaultdict(set)
    children = defaultdict(set)
    for term_id, info in terms.items():
        for parent_id in info.get('is_a', []):
            if parent_id in terms:
                parents[term_id].add(parent_id)
                children[parent_id].add(term_id)
    return parents, children

def get_all_descendants(node, children_map, include_self=True):
    """Get all descendants of node (transitive closure of children)."""
    desc = {node} if include_self else set()
    stack = list(children_map.get(node, set()))
    while stack:
        n = stack.pop()
        if n not in desc:
            desc.add(n)
            stack.extend(children_map.get(n, set()))
    return desc

def get_ancestors(node, parents, include_self=True):
    ancestors = {node} if include_self else set()
    stack = list(parents.get(node, set()))
    while stack:
        n = stack.pop()
        if n not in ancestors:
            ancestors.add(n)
            stack.extend(parents.get(n, set()))
    return ancestors

def get_depth(node, parents):
    """Shortest path to root (CL:0000000). In DAG, use shortest path as depth."""
    root = 'CL:0000000'
    if node == root:
        return 0
    visited = {node}
    queue = deque([(node, 0)])
    while queue:
        current, dist = queue.popleft()
        for parent in parents.get(current, set()):
            if parent == root:
                return dist + 1
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, dist + 1))
    return -1  # disconnected from root

def get_leaf_descendants(node, children_map):
    """Get leaf descendants (nodes with no children) of node."""
    desc = get_all_descendants(node, children_map, include_self=True)
    leaves = {d for d in desc if not children_map.get(d, set())}
    return leaves

# --- IC Formulas ---

def ic_seco(desc_count, total_terms):
    """Seco (2004): IC = 1 - log(desc+1)/log(N)"""
    if total_terms <= 1:
        return 0.0
    return 1.0 - math.log(desc_count + 1) / math.log(total_terms)

def ic_zhou(desc_count, total_terms, depth, max_depth, k=0.5):
    """Zhou (2008): IC = k * Seco + (1-k) * depth_component"""
    seco = ic_seco(desc_count, total_terms)
    if max_depth == 0:
        depth_comp = 0.0
    else:
        depth_comp = math.log(depth + 1) / math.log(max_depth + 1)
    return k * seco + (1.0 - k) * depth_comp

def ic_sanchez(leaf_count, desc_count, max_leaves):
    """Sanchez (2011): IC = -log( (leaves(t)/desc(t) + 1) / (max_leaves + 1) )"""
    if desc_count == 0:
        return 0.0
    ratio = leaf_count / desc_count
    # Add 1 to numerator and denominator to avoid log(0)
    return -math.log((ratio + 1) / (max_leaves + 1))

def ic_depth(depth, max_depth):
    """Simple depth-based IC"""
    if max_depth == 0:
        return 0.0
    return depth / max_depth

# --- MICA and Lin Similarity ---

def find_mica(node_a, node_b, parents, ic_values):
    """Find Most Informative Common Ancestor (common ancestor with highest IC)."""
    ancestors_a = get_ancestors(node_a, parents, include_self=True)
    ancestors_b = get_ancestors(node_b, parents, include_self=True)
    common = ancestors_a & ancestors_b

    if not common:
        return None, 0.0

    mica = max(common, key=lambda n: ic_values.get(n, 0.0))
    return mica, ic_values.get(mica, 0.0)

def lin_similarity(node_a, node_b, parents, ic_values):
    """Lin similarity: 2*IC(MICA) / (IC(A) + IC(B))"""
    mica, mica_ic = find_mica(node_a, node_b, parents, ic_values)
    ic_a = ic_values.get(node_a, 0.0)
    ic_b = ic_values.get(node_b, 0.0)

    if ic_a + ic_b == 0:
        return 0.0, mica, mica_ic

    sim = (2.0 * mica_ic) / (ic_a + ic_b)
    return sim, mica, mica_ic

# --- Main ---
OBO_PATH = "unit-tests/mock_data/reference_data/cl.obo"
terms = parse_obo(OBO_PATH)
parents, children_map = build_graph(terms)
N = len(terms)

print(f"Ontology: {N} non-obsolete terms")

# Precompute for all terms
print("Precomputing descendants, depths, leaves...")
desc_counts = {}
depths = {}
leaf_counts = {}

for tid in terms:
    desc = get_all_descendants(tid, children_map, include_self=True)
    desc_counts[tid] = len(desc)
    depths[tid] = get_depth(tid, parents)
    leaves = get_leaf_descendants(tid, children_map)
    leaf_counts[tid] = len(leaves)

max_depth = max(d for d in depths.values() if d >= 0)
max_leaves = max(leaf_counts.values())
total_leaves = sum(1 for tid in terms if not children_map.get(tid, set()))

print(f"  Max depth: {max_depth}")
print(f"  Total leaf nodes: {total_leaves}")
print(f"  Max leaves under any node: {max_leaves}")

# Compute IC for all terms under each formula
ic_formulas = {
    'Seco': {},
    'Zhou(k=0.5)': {},
    'Sanchez': {},
    'Depth-only': {},
}

for tid in terms:
    d = depths[tid] if depths[tid] >= 0 else 0
    ic_formulas['Seco'][tid] = ic_seco(desc_counts[tid], N)
    ic_formulas['Zhou(k=0.5)'][tid] = ic_zhou(desc_counts[tid], N, d, max_depth, k=0.5)
    ic_formulas['Sanchez'][tid] = ic_sanchez(leaf_counts[tid], desc_counts[tid], max_leaves)
    ic_formulas['Depth-only'][tid] = ic_depth(d, max_depth)

# --- Show IC values for key neuroscience cell types ---
neuro = [
    'CL:0000000',  # cell (root)
    'CL:0000540',  # neuron
    'CL:0000617',  # GABAergic neuron
    'CL:0000679',  # glutamatergic neuron
    'CL:0000127',  # astrocyte
    'CL:0000128',  # oligodendrocyte
    'CL:0000129',  # microglial cell
    'CL:0000099',  # interneuron
    'CL:0000121',  # Purkinje cell
    'CL:0000125',  # glial cell
    'CL:0000126',  # macroglial cell
    'CL:0002605',  # astrocyte of the cerebral cortex
    'CL:0000151',  # secretory cell
]

print("\n" + "=" * 110)
print(f"{'Cell Type':<40} {'Depth':>5} {'Desc':>5} {'Leaves':>6} | {'Seco':>6} {'Zhou':>6} {'Sanchez':>7} {'Depth':>6}")
print("=" * 110)

for tid in neuro:
    if tid not in terms:
        continue
    n = terms[tid]['name']
    d = depths[tid] if depths[tid] >= 0 else 0
    print(f"{n:<40} {d:>5} {desc_counts[tid]:>5} {leaf_counts[tid]:>6} | "
          f"{ic_formulas['Seco'][tid]:>6.3f} "
          f"{ic_formulas['Zhou(k=0.5)'][tid]:>6.3f} "
          f"{ic_formulas['Sanchez'][tid]:>7.3f} "
          f"{ic_formulas['Depth-only'][tid]:>6.3f}")

# --- Compare Lin similarity for interesting pairs ---
pairs = [
    ('CL:0000617', 'CL:0000679'),  # GABAergic vs glutamatergic
    ('CL:0000540', 'CL:0000617'),  # neuron vs GABAergic
    ('CL:0000617', 'CL:0000121'),  # GABAergic vs Purkinje
    ('CL:0000127', 'CL:0000128'),  # astrocyte vs oligodendrocyte
    ('CL:0000127', 'CL:0000129'),  # astrocyte vs microglial cell
    ('CL:0000540', 'CL:0000127'),  # neuron vs astrocyte
    ('CL:0000540', 'CL:0000129'),  # neuron vs microglial cell
    ('CL:0002605', 'CL:0000127'),  # astrocyte of cerebral cortex vs astrocyte
    ('CL:0000679', 'CL:0000121'),  # glutamatergic vs Purkinje
]

for formula_name, ic_vals in ic_formulas.items():
    print(f"\n{'='*110}")
    print(f"LIN SIMILARITY using {formula_name} IC")
    print(f"{'='*110}")
    print(f"{'Cell A':<25} {'Cell B':<25} {'Sim':>6} | {'MICA':<30} {'IC_A':>5} {'IC_B':>5} {'IC_MICA':>7}")
    print("-" * 110)

    for id_a, id_b in pairs:
        if id_a not in terms or id_b not in terms:
            continue
        name_a = terms[id_a]['name']
        name_b = terms[id_b]['name']
        sim, mica, mica_ic = lin_similarity(id_a, id_b, parents, ic_vals)
        mica_name = terms[mica]['name'] if mica else "N/A"
        ic_a = ic_vals.get(id_a, 0)
        ic_b = ic_vals.get(id_b, 0)
        print(f"{name_a:<25} {name_b:<25} {sim:>6.3f} | {mica_name:<30} {ic_a:>5.3f} {ic_b:>5.3f} {mica_ic:>7.3f}")

# --- Show the problematic case from earlier ---
print(f"\n{'='*110}")
print("PREVIOUSLY PROBLEMATIC CASES (shortest path != LCA distance)")
print(f"{'='*110}")

problem_pairs = [
    ('CL:0000783', 'CL:0002461'),  # multinucleated phagocyte vs CD103+ DC
]

for id_a, id_b in problem_pairs:
    if id_a not in terms or id_b not in terms:
        print(f"  Skipping {id_a} or {id_b} - not found")
        continue
    name_a = terms[id_a]['name']
    name_b = terms[id_b]['name']

    print(f"\n{name_a}  vs  {name_b}")
    print(f"  IC values:")
    for fname, ic_vals in ic_formulas.items():
        ic_a = ic_vals.get(id_a, 0)
        ic_b = ic_vals.get(id_b, 0)
        sim, mica, mica_ic = lin_similarity(id_a, id_b, parents, ic_vals)
        mica_name = terms[mica]['name'] if mica else "N/A"
        print(f"  {fname:<15}: IC_A={ic_a:.3f}, IC_B={ic_b:.3f}, MICA={mica_name} (IC={mica_ic:.3f}), Lin={sim:.3f}")

print("\nDone.")
