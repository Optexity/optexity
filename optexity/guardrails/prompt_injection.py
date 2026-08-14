import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionMatch:
    pattern_name: str
    excerpt: str


# High-confidence phrases only.  Broad words such as "instruction" are
# intentionally absent to avoid blocking ordinary page copy.
BUILTIN_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "ignore_instructions",
        r"\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?\b",
    ),
    (
        "override_system",
        r"\b(?:override|bypass)\s+(?:the\s+)?(?:system|developer|security)\s+(?:prompt|instructions?|policy)\b",
    ),
    ("role_hijack", r"\byou\s+are\s+now\s+(?:an?|the)\b"),
    (
        "secret_exfiltration",
        r"\b(?:reveal|send|upload|exfiltrate|print)\s+(?:all\s+)?(?:passwords?|secrets?|tokens?|credentials?|api[_ -]?keys?)\b",
    ),
    (
        "prompt_disclosure",
        r"\b(?:reveal|repeat|print|show)\s+(?:your\s+)?(?:system|developer)\s+(?:prompt|instructions?)\b",
    ),
    (
        "tool_coercion",
        r"\buse\s+(?:the\s+)?(?:browser|tool|terminal)\s+to\s+(?:ignore|bypass|steal|exfiltrate)\b",
    ),
)


def detect_prompt_injection(
    text: str, additional_patterns: list[str] | None = None
) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    patterns = list(BUILTIN_PATTERNS)
    patterns.extend(
        (f"custom_{index}", pattern)
        for index, pattern in enumerate(additional_patterns or [])
    )
    for name, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            matches.append(InjectionMatch(name, text[start:end]))
    return matches
