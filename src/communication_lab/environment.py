from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from pydantic import BaseModel

from communication_lab.config import LLAMA_GATE_MODEL_ID
from communication_lab.explanations import has_openai_api_key


class CheckStatus(BaseModel):
    ok: bool
    summary: str
    details: str = ""


class EnvironmentReport(BaseModel):
    python_version: str
    ffmpeg: CheckStatus
    huggingface_token: CheckStatus
    llama_gate: CheckStatus
    openai_api_key: CheckStatus
    warnings: list[str]

    @property
    def ready_for_inference(self) -> bool:
        return self.ffmpeg.ok and self.huggingface_token.ok and self.llama_gate.ok

    @property
    def ready_for_explanations(self) -> bool:
        return self.openai_api_key.ok


def _get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip() or None
    return None


def check_ffmpeg() -> CheckStatus:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return CheckStatus(
            ok=False,
            summary="ffmpeg not found",
            details="Install ffmpeg so audio/video duration checks and trimming work.",
        )
    completed = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ffmpeg_path
    return CheckStatus(ok=completed.returncode == 0, summary=first_line)


def check_huggingface_token() -> CheckStatus:
    token = _get_hf_token()
    if token:
        masked = f"{token[:4]}...{token[-4:]}" if len(token) >= 8 else "***"
        return CheckStatus(ok=True, summary="Hugging Face token found", details=masked)
    return CheckStatus(
        ok=False,
        summary="No Hugging Face token found",
        details="Set HF_TOKEN or run `huggingface-cli login` after installing the runtime extras.",
    )


def check_llama_gate(token: str | None = None) -> CheckStatus:
    token = token or _get_hf_token()
    if not token:
        return CheckStatus(
            ok=False,
            summary="Skipped LLaMA gate check",
            details="No token available.",
        )
    api = HfApi(token=token)
    try:
        api.model_info(LLAMA_GATE_MODEL_ID)
    except GatedRepoError as exc:
        return CheckStatus(
            ok=False,
            summary="Token lacks gated access to LLaMA 3.2-3B",
            details=str(exc),
        )
    except HfHubHTTPError as exc:
        return CheckStatus(
            ok=False,
            summary="Could not verify gated model access",
            details=str(exc),
        )
    return CheckStatus(
        ok=True,
        summary="Gated access to LLaMA 3.2-3B confirmed",
    )


def check_openai_api_key() -> CheckStatus:
    if has_openai_api_key():
        return CheckStatus(
            ok=True,
            summary="OpenAI API key found",
        )
    return CheckStatus(
        ok=False,
        summary="No OpenAI API key found",
        details="Set OPENAI_API_KEY to enable optional natural-language explanations.",
    )


def build_environment_report(skip_remote_checks: bool = False) -> EnvironmentReport:
    token = _get_hf_token()
    hf_token = check_huggingface_token()
    llama_gate = (
        CheckStatus(
            ok=False,
            summary="Remote gate check skipped",
            details="Skipping remote verification by request.",
        )
        if skip_remote_checks
        else check_llama_gate(token=token)
    )
    warnings: list[str] = []
    if not hf_token.ok:
        warnings.append(hf_token.summary)
    if not llama_gate.ok:
        warnings.append(llama_gate.summary)
    ffmpeg = check_ffmpeg()
    if not ffmpeg.ok:
        warnings.append(ffmpeg.summary)
    openai_api_key = check_openai_api_key()
    return EnvironmentReport(
        python_version=os.sys.version.split()[0],
        ffmpeg=ffmpeg,
        huggingface_token=hf_token,
        llama_gate=llama_gate,
        openai_api_key=openai_api_key,
        warnings=warnings,
    )
