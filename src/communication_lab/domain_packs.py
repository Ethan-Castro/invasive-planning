from __future__ import annotations

from dataclasses import dataclass

from communication_lab.schemas import DomainPack, ProxyFamily


@dataclass(frozen=True)
class DomainPackSpec:
    title: str
    description: str
    focus_proxies: tuple[ProxyFamily, ...]
    disclaimers: tuple[str, ...]
    interpretation_notes: tuple[str, ...]


DOMAIN_PACK_SPECS: dict[DomainPack, DomainPackSpec] = {
    DomainPack.teacher: DomainPackSpec(
        title="Teacher Communication Pack",
        description="Compare lesson plans and instructional explanations with an emphasis on language-network engagement and how broadly responses spread across cortical parcels.",
        focus_proxies=(
            ProxyFamily.language_processing_proxy,
            ProxyFamily.cross_region_spread_proxy,
        ),
        disclaimers=(
            "This does not predict classroom learning outcomes or student behavior.",
        ),
        interpretation_notes=(
            "Use language-processing proxies to compare wording density and structure.",
            "Use cross-region spread as a rough indicator of how widely cortical responses distribute across parcels.",
        ),
    ),
    DomainPack.doctor: DomainPackSpec(
        title="Doctor Communication Pack",
        description="Compare bad-news explanations and clinical scripts with an emphasis on language-related and emotional-social neural proxies.",
        focus_proxies=(
            ProxyFamily.language_processing_proxy,
            ProxyFamily.emotional_social_proxy,
        ),
        disclaimers=(
            "Clinical messaging still requires clinician judgment, ethics review, and human testing.",
            "The app does not estimate patient outcomes, comprehension, or distress directly.",
        ),
        interpretation_notes=(
            "Treat emotional-social signals as exploratory and not as patient-state measurements.",
        ),
    ),
    DomainPack.marketing: DomainPackSpec(
        title="Marketing Communication Pack",
        description="Compare ad scripts and assets with an emphasis on auditory/visual salience plus multimodal spread.",
        focus_proxies=(
            ProxyFamily.auditory_salience_proxy,
            ProxyFamily.visual_salience_proxy,
            ProxyFamily.cross_region_spread_proxy,
        ),
        disclaimers=(
            "This app does not predict CTR, conversion, or purchase intent.",
        ),
        interpretation_notes=(
            "Use pairwise deltas to compare creative variants, not to infer business outcomes.",
        ),
    ),
    DomainPack.investor: DomainPackSpec(
        title="Investor Communication Pack",
        description="Compare investor narratives and demos with an emphasis on language-processing structure and emotional-social proxies.",
        focus_proxies=(
            ProxyFamily.language_processing_proxy,
            ProxyFamily.emotional_social_proxy,
        ),
        disclaimers=(
            "This app does not estimate investor interest or funding probability.",
        ),
        interpretation_notes=(
            "Use the output to compare draft structure and temporal divergence, not to make funding claims.",
        ),
    ),
}


def get_domain_pack_spec(domain_pack: DomainPack) -> DomainPackSpec:
    return DOMAIN_PACK_SPECS[domain_pack]
