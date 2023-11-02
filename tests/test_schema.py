import os.path
from nomad.client import parse, normalize_all

def test_schema():
    test_file = os.path.join(os.path.dirname(__file__), 'data', 'catalyst.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.name == 'sample'
