"""Download and load the ogbn-arxiv dataset into data/."""

from pathlib import Path

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


if __name__ == "__main__":
    print(f"Downloading ogbn-arxiv to {DATA_DIR}...")
    dataset = download_ogbn_arxiv()
    print(f"Downloaded. Graph: {dataset[0]}")
    print(f"Number of classes: {dataset.num_classes}")
