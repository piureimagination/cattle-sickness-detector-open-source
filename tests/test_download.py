from pathlib import Path

import pytest

from src.data.download import download_data, normalize_dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")


@pytest.fixture
def workdir(tmp_path_factory):
    """A scratch directory with a neutral name.

    normalize_dataset() classifies files by keyword substrings anywhere in
    their full path, which is realistic for how Kaggle datasets are laid
    out - but it also means we can't use pytest's default `tmp_path`
    fixture here: its directory name is derived from the test function
    name, and several of these tests have "normalize" in their name, whose
    "normal" substring would itself spuriously match HEALTHY_KEYWORDS. A
    neutral basename sidesteps that.
    """
    return tmp_path_factory.mktemp("case")


def test_normalize_dataset_labels_by_keyword(workdir):
    dataset_path = workdir / "downloaded"
    _touch(dataset_path / "Healthy Cows" / "a.jpg")
    _touch(dataset_path / "Lumpy Cows" / "b.jpg")
    raw_dir = workdir / "raw"

    normalize_dataset(dataset_path, raw_dir)

    assert [f.name for f in (raw_dir / "healthy").iterdir()] == ["a.jpg"]
    assert [f.name for f in (raw_dir / "lumpy").iterdir()] == ["b.jpg"]


def test_normalize_dataset_matches_alternate_keywords(workdir):
    dataset_path = workdir / "downloaded"
    _touch(dataset_path / "normal" / "a.jpg")
    _touch(dataset_path / "infected" / "b.jpg")
    _touch(dataset_path / "diseased" / "c.jpg")
    raw_dir = workdir / "raw"

    normalize_dataset(dataset_path, raw_dir)

    assert len(list((raw_dir / "healthy").iterdir())) == 1
    assert len(list((raw_dir / "lumpy").iterdir())) == 2


def test_normalize_dataset_raises_on_unrecognized_file(workdir):
    dataset_path = workdir / "downloaded"
    _touch(dataset_path / "healthy" / "a.jpg")
    _touch(dataset_path / "mystery_folder" / "b.jpg")
    raw_dir = workdir / "raw"

    with pytest.raises(ValueError, match="matched neither"):
        normalize_dataset(dataset_path, raw_dir)


def test_normalize_dataset_raises_on_empty_source(workdir):
    dataset_path = workdir / "downloaded"
    dataset_path.mkdir()
    raw_dir = workdir / "raw"

    with pytest.raises(ValueError, match="No files found"):
        normalize_dataset(dataset_path, raw_dir)


def test_normalize_dataset_raises_when_one_class_empty(workdir):
    dataset_path = workdir / "downloaded"
    _touch(dataset_path / "healthy" / "a.jpg")
    _touch(dataset_path / "healthy" / "b.jpg")
    raw_dir = workdir / "raw"

    with pytest.raises(ValueError, match="empty class"):
        normalize_dataset(dataset_path, raw_dir)


def test_normalize_dataset_dedupes_by_filename(workdir):
    dataset_path = workdir / "downloaded"
    _touch(dataset_path / "healthy" / "dup.jpg")
    _touch(dataset_path / "healthy" / "subdir" / "dup.jpg")
    _touch(dataset_path / "lumpy" / "other.jpg")
    raw_dir = workdir / "raw"

    normalize_dataset(dataset_path, raw_dir)

    assert len(list((raw_dir / "healthy").iterdir())) == 1


def test_download_data_resets_raw_dir_and_normalizes(workdir, monkeypatch):
    stale = workdir / "raw" / "healthy" / "stale.jpg"
    _touch(stale)
    assert stale.exists()

    source = workdir / "kaggle_download"
    _touch(source / "healthy" / "a.jpg")
    _touch(source / "lumpy" / "b.jpg")
    monkeypatch.setattr("src.data.download.kagglehub.dataset_download", lambda dataset_id: str(source))

    raw_dir = workdir / "raw"
    result = download_data("someuser/some-dataset", raw_dir)

    assert result == raw_dir
    assert not stale.exists()
    assert len(list((raw_dir / "healthy").iterdir())) == 1
    assert len(list((raw_dir / "lumpy").iterdir())) == 1
