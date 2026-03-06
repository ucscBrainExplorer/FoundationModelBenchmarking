"""
Session-scoped fixtures for cell_labelling tests.
Creates small synthetic query h5ad (50 cells) so CLI tests run fast.
"""

import os
import pytest
import numpy as np
import anndata

DEMODATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demodata')


@pytest.fixture(scope='session')
def small_adata_path(tmp_path_factory):
    """50-cell subset of query_uce_adata.h5ad for fast CLI tests."""
    src = os.path.join(DEMODATA, 'query_uce_adata.h5ad')
    adata = anndata.read_h5ad(src)
    small = adata[:50].copy()
    out = tmp_path_factory.mktemp('data') / 'small_query.h5ad'
    small.write_h5ad(str(out))
    return str(out)
