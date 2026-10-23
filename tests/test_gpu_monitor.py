import uuid
import pytest
from scripts.gpu_monitor import resolve_uuid

ID = '12345678-1234-5678-1234-567812345678'
@pytest.mark.parametrize('value', [ID, 'GPU-'+ID, uuid.UUID(ID), uuid.UUID(ID).bytes])
def test_resolves_torch_uuid_against_smi(value):
    assert resolve_uuid(value, ['GPU-'+ID, 'GPU-87654321-1234-5678-1234-567812345678']) == 'GPU-'+ID

def test_never_falls_back_to_first_gpu():
    with pytest.raises(ValueError):
        resolve_uuid(ID, ['GPU-87654321-1234-5678-1234-567812345678'])
