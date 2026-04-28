"""Shared text utilities used across agents."""

TRIVIAL_FOLLOWUPS: frozenset[str] = frozenset({
    "yes", "no", "sure", "ok", "okay", "please", "yes please",
    "no thanks", "proceed", "go ahead", "continue", "yeah", "yep",
})
