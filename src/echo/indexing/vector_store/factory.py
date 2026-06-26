from echo.indexing.vector_store.faiss_variants import FaissFlatStore, FaissHNSWStore

def get_faiss_store(config: dict):
    variant = config.get("variant")
    dim = config.get("dim")

    if not variant:
        raise ValueError("vector_store config missing required field: 'variant'")
    if not dim:
        raise ValueError("vector_store config missing required field: 'dim'")

    variant_params = config.get(variant, {})

    variants = {
        "flat": FaissFlatStore,
        "hnsw": FaissHNSWStore
    }

    cls = variants.get(variant)
    if cls is None:
        raise ValueError(f"Unknown FAISS variant: '{variant}' Options are: {list(variants.keys())}")

    return cls(dim, **variant_params)