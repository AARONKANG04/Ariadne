"""Download and load the ogbn-arxiv dataset into data/."""

from pathlib import Path
from contextlib import contextmanager
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_ogbn_arxiv(root: str | Path | None = None) -> "NodePropPredDataset":
    """Download ogbn-arxiv to data/ and return the dataset.

    Args:
        root: Directory to save the dataset. Defaults to ml_pipeline/data/.

    Returns:
        NodePropPredDataset instance for ogbn-arxiv.
    """
    from ogb.nodeproppred import NodePropPredDataset

    root = Path(root) if root is not None else DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(root))
    return dataset


def load_ogbn_arxiv(root: str | Path | None = None) -> "NodePropPredDataset":
    """Load ogbn-arxiv from data/, downloading if not present."""
    return download_ogbn_arxiv(root=root)

def unsafe_load_ogbn_arxiv() -> "NodePropPredDataset":
    @contextmanager
    def allow_unsafe_torch_load():
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            yield
        finally:
            torch.load = original_load

    with allow_unsafe_torch_load():
        return load_ogbn_arxiv()

if __name__ == "__main__":
    print(f"Downloading ogbn-arxiv to {DATA_DIR}...")
    dataset = download_ogbn_arxiv()
    print(f"Downloaded. Graph: {dataset[0]}")
    print(f"Number of classes: {dataset.num_classes}")