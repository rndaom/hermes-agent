"""Shared gateway restart constants and parsing helpers."""

from math import ceil

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)

# Leave headroom between the passive drain window and the service manager's
# hard stop deadline so the gateway can interrupt stuck agents, disconnect
# adapters, flush shutdown state, and exit cleanly before systemd sends SIGKILL.
GATEWAY_SERVICE_STOP_TIMEOUT_BUFFER_SECONDS = 15.0


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def compute_service_stop_timeout(drain_timeout: object) -> int:
    """Return a service stop timeout that leaves cleanup headroom after drain."""
    parsed = parse_restart_drain_timeout(drain_timeout)
    return max(60, int(ceil(parsed + GATEWAY_SERVICE_STOP_TIMEOUT_BUFFER_SECONDS)))
