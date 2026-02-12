"""Parse Cell Ontology OBO files into {term_id: name} dictionaries."""


def parse_obo_names(obo_path: str) -> dict:
    """Parse an OBO file and return a dict of {term_id: name}.

    Reads [Term] blocks and extracts the ``id:`` and ``name:`` fields.
    Non-Term sections (e.g. [Typedef]) are skipped.

    Args:
        obo_path: Path to an OBO ontology file (e.g. cl.obo).

    Returns:
        Dict mapping term IDs to canonical names,
        e.g. ``{"CL:0000540": "neuron", ...}``.
    """
    cl_map = {}
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
                elif line.startswith('name: ') and current_id:
                    cl_map[current_id] = line[6:]

    return cl_map
