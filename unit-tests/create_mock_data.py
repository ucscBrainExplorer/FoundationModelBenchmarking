"""
Helper script to create mock data for unit tests.
This creates minimal but realistic test fixtures.
"""
import os
import numpy as np
import pandas as pd
import faiss


def create_mock_faiss_index(output_path: str, n_vectors: int = 100, dimension: int = 64):
    """Create a simple FAISS IVFFlat index for testing."""
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype('float32')

    # Create a simple flat index (or IVFFlat for more realism)
    # For simplicity, use Flat first
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save index
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    print(f"Created mock FAISS index at {output_path}")
    return vectors


def create_mock_ivfflat_index(output_path: str, n_vectors: int = 100, dimension: int = 64):
    """Create an IVFFlat index for testing."""
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype('float32')

    # Create IVFFlat index
    nlist = 10  # number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    # Train the index
    index.train(vectors)
    index.add(vectors)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    print(f"Created mock IVFFlat index at {output_path}")
    return vectors


def create_mock_reference_annotations(output_path: str, n_cells: int = 100):
    """Create mock reference annotations TSV file."""
    np.random.seed(42)

    # Create mock cell type ontology IDs and names
    cell_types = [
        ('CL:0000001', 'neuron'),
        ('CL:0000002', 'astrocyte'),
        ('CL:0000003', 'oligodendrocyte'),
        ('CL:0000004', 'microglia'),
        ('CL:0000005', 'endothelial cell')
    ]

    # Randomly assign cell types
    data = {
        'cell_type_ontology_term_id': [],
        'cell_type': []
    }

    for i in range(n_cells):
        idx = np.random.randint(0, len(cell_types))
        data['cell_type_ontology_term_id'].append(cell_types[idx][0])
        data['cell_type'].append(cell_types[idx][1])

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Created mock reference annotations at {output_path}")
    return df


def create_mock_test_dataset(test_dir: str, dataset_id: str, n_queries: int = 20, dimension: int = 64):
    """Create a mock test dataset with embeddings and metadata."""
    np.random.seed(42)

    # Create embeddings
    embeddings = np.random.randn(n_queries, dimension).astype('float32')
    embedding_path = os.path.join(test_dir, f"{dataset_id}_embeddings.npy")

    # Create metadata with ground truth labels
    cell_types = [
        ('CL:0000001', 'neuron'),
        ('CL:0000002', 'astrocyte'),
        ('CL:0000003', 'oligodendrocyte'),
        ('CL:0000004', 'microglia'),
        ('CL:0000005', 'endothelial cell')
    ]

    data = {
        'cell_type_ontology_term_id': [],
        'cell_type': []
    }

    for i in range(n_queries):
        idx = np.random.randint(0, len(cell_types))
        data['cell_type_ontology_term_id'].append(cell_types[idx][0])
        data['cell_type'].append(cell_types[idx][1])

    metadata_df = pd.DataFrame(data)
    metadata_path = os.path.join(test_dir, f"{dataset_id}_prediction_obs.tsv")

    os.makedirs(test_dir, exist_ok=True)
    np.save(embedding_path, embeddings)
    metadata_df.to_csv(metadata_path, sep='\t', index=False)

    print(f"Created mock test dataset '{dataset_id}' in {test_dir}")
    return embeddings, metadata_df


def create_mock_obo_file(output_path: str):
    """Create a minimal mock OBO file for Cell Ontology."""
    obo_content = """format-version: 1.2
data-version: cl/releases/2024-01-01
ontology: cl

[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000001
name: neuron
is_a: CL:0000000 ! cell

[Term]
id: CL:0000002
name: astrocyte
is_a: CL:0000000 ! cell

[Term]
id: CL:0000003
name: oligodendrocyte
is_a: CL:0000000 ! cell

[Term]
id: CL:0000004
name: microglia
is_a: CL:0000000 ! cell

[Term]
id: CL:0000005
name: endothelial cell
is_a: CL:0000000 ! cell

[Term]
id: CL:0000006
name: GABAergic neuron
is_a: CL:0000001 ! neuron

[Term]
id: CL:0000007
name: glutamatergic neuron
is_a: CL:0000001 ! neuron
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(obo_content)
    print(f"Created mock OBO file at {output_path}")


def create_all_mock_data(base_dir: str = "unit-tests/mock_data"):
    """Create all mock data files needed for testing."""
    print("Creating all mock data files...")

    # Create FAISS indices
    create_mock_faiss_index(
        os.path.join(base_dir, "indices/index_flat.faiss"),
        n_vectors=100,
        dimension=64
    )

    create_mock_ivfflat_index(
        os.path.join(base_dir, "indices/index_ivfflat.faiss"),
        n_vectors=100,
        dimension=64
    )

    # Create reference annotations (must align with 100 vectors in index)
    create_mock_reference_annotations(
        os.path.join(base_dir, "reference_data/prediction_obs.tsv"),
        n_cells=100
    )

    # Create test datasets
    test_dir = os.path.join(base_dir, "test_data")
    create_mock_test_dataset(test_dir, "organoid_test", n_queries=20, dimension=64)
    create_mock_test_dataset(test_dir, "brain_test", n_queries=15, dimension=64)
    create_mock_test_dataset(test_dir, "positive_control", n_queries=10, dimension=64)

    # Create OBO file
    create_mock_obo_file(
        os.path.join(base_dir, "reference_data/cl.obo")
    )

    print("\n✓ All mock data created successfully!")


if __name__ == "__main__":
    create_all_mock_data()
