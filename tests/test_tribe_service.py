from pathlib import Path

import pytest

from communication_lab.schemas import DomainPack, InputVariant, Modality
from communication_lab.tribe_service import TribeService


def build_text_variant(path: Path, content: str) -> InputVariant:
    path.write_text(content, encoding="utf-8")
    return InputVariant(
        id="v1",
        label="Variant 1",
        modality=Modality.text,
        file_path=path,
        domain_pack=DomainPack.teacher,
    )


def test_prepare_text_input_trims_long_text(tmp_path: Path):
    service = TribeService(cache_root=tmp_path)
    text = "word " * 500
    variant = build_text_variant(tmp_path / "long.txt", text)
    prepared_path, warnings = service._prepare_text_input(variant, max_duration_s=30)
    assert prepared_path != variant.file_path
    assert prepared_path.read_text(encoding="utf-8")
    assert warnings


def test_prepare_text_input_rejects_empty_text(tmp_path: Path):
    service = TribeService(cache_root=tmp_path)
    variant = build_text_variant(tmp_path / "empty.txt", "   ")
    with pytest.raises(ValueError):
        service._prepare_text_input(variant, max_duration_s=30)
