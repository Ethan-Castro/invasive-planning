from pathlib import Path

import pytest

from communication_lab.schemas import CompareRequest, DomainPack, InputVariant, Modality


def write_file(path: Path, content: str = "hello") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_input_variant_validates_suffix_and_path(tmp_path: Path):
    path = write_file(tmp_path / "variant.txt")
    variant = InputVariant(
        id="v1",
        label="Variant 1",
        modality=Modality.text,
        file_path=path,
        domain_pack=DomainPack.teacher,
    )
    assert variant.file_path == path.resolve()


def test_compare_request_requires_unique_ids(tmp_path: Path):
    path = write_file(tmp_path / "variant.txt")
    variant = InputVariant(
        id="duplicate",
        label="Variant",
        modality=Modality.text,
        file_path=path,
        domain_pack=DomainPack.teacher,
    )
    with pytest.raises(ValueError):
        CompareRequest(
            variants=[variant, variant],
            export_dir=tmp_path / "exports",
        )


def test_compare_request_accepts_single_variant(tmp_path: Path):
    path = write_file(tmp_path / "variant.txt")
    variant = InputVariant(
        id="single",
        label="Single Variant",
        modality=Modality.text,
        file_path=path,
        domain_pack=DomainPack.teacher,
    )
    request = CompareRequest(
        variants=[variant],
        export_dir=tmp_path / "exports",
    )
    assert len(request.variants) == 1


def test_input_variant_rejects_wrong_suffix(tmp_path: Path):
    path = write_file(tmp_path / "variant.mp4")
    with pytest.raises(ValueError):
        InputVariant(
            id="v1",
            label="Variant",
            modality=Modality.audio,
            file_path=path,
            domain_pack=DomainPack.marketing,
        )
