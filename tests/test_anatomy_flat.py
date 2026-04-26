from src.classifier import build_anatomy_flat, ANATOMY_GROUPS

def test_anatomy_flat_has_no_duplicates():
    try:
        build_anatomy_flat(ANATOMY_GROUPS)
    except ValueError as e:
        assert False, f"Duplicate anatomy terms found: {e}"