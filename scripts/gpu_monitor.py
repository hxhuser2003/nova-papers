"""Resolve CUDA-visible device 0 to an actual nvidia-smi UUID, never index 0."""
import subprocess
import uuid


def canonical_uuid(value):
    if isinstance(value, bytes):
        if len(value) == 16:
            return str(uuid.UUID(bytes=value))
        value = value.decode('ascii')
    return str(uuid.UUID(str(value).removeprefix('GPU-')))


def resolve_uuid(torch_uuid, smi_uuids):
    target = canonical_uuid(torch_uuid)
    matches = [s.strip() for s in smi_uuids if canonical_uuid(s.strip()) == target]
    if len(matches) != 1:
        raise ValueError('CUDA-visible GPU UUID does not uniquely match nvidia-smi; refusing another GPU')
    return matches[0]


def main():
    import torch
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    print(resolve_uuid(torch.cuda.get_device_properties(0).uuid, listing.splitlines()))


if __name__ == '__main__':
    main()
