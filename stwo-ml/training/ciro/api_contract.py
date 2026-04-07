"""
CIRO ↔ ObelyZK API Contract

This file defines the exact API surface that CIRO must implement
and ObelyZK consumes. Both teams use this as the source of truth.

CIRO endpoints live under: /api/singularity/{org}/blockchain/
ObelyZK consumes them via ciro_client.py (training) and
the TypeScript SDK (runtime enrichment).

Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class ThreatLabel(Enum):
    """Ground-truth label for a transaction."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    UNLABELED = "unlabeled"


class LabelSource(Enum):
    """Where the label came from — determines trust level."""
    ON_CHAIN_REVERT = "on_chain_revert"          # Tier 1: tx reverted on-chain
    ON_CHAIN_SLASH = "on_chain_slash"             # Tier 1: validator slashed
    FORTA_ALERT = "forta_alert"                   # Tier 2: Forta bot detection
    DEFIHACKLAB = "defihacklab"                   # Tier 2: post-mortem analysis
    REKT_NEWS = "rekt_news"                       # Tier 2: incident report
    CHAINALYSIS_SANCTION = "chainalysis_sanction"  # Tier 2: sanctions list
    OFAC = "ofac"                                 # Tier 2: US Treasury
    COMMUNITY_REPORT = "community_report"         # Tier 2: user flagged
    HEURISTIC = "heuristic"                       # Tier 3: behavioral pattern
    MANUAL = "manual"                             # human analyst label
    SYNTHETIC = "synthetic"                       # generated data


class RiskLevel(Enum):
    """Address risk assessment."""
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"          # sanctioned / known exploit
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════
# Request / Response Types
# ═══════════════════════════════════════════════════════════════════════

# ── GET /transactions/labeled ─────────────────────────────────────────

@dataclass
class LabeledTransactionQuery:
    """Query parameters for fetching labeled transactions."""
    label: Optional[ThreatLabel] = None   # filter by label
    source: Optional[LabelSource] = None  # filter by label source
    chain: str = "starknet"               # blockchain
    min_confidence: float = 0.0           # 0.0-1.0 label confidence
    since: Optional[str] = None           # ISO 8601 timestamp
    until: Optional[str] = None           # ISO 8601 timestamp
    limit: int = 1000                     # page size
    offset: int = 0                       # pagination offset


@dataclass
class LabeledTransaction:
    """A single labeled transaction from CIRO's data lake."""
    # Identity
    tx_hash: str                          # transaction hash
    chain: str                            # "starknet", "ethereum", etc.
    block_number: int
    timestamp: str                        # ISO 8601

    # Core transaction data
    sender: str                           # sender address (hex)
    target: str                           # target contract address (hex)
    value: str                            # decimal string (u256)
    selector: str                         # function selector (hex, 4 bytes)
    calldata_hex: str                     # full calldata (hex)
    calldata_len: int                     # calldata byte length

    # Label
    label: ThreatLabel
    label_source: LabelSource
    label_confidence: float               # 0.0-1.0
    label_reason: str                     # human-readable reason

    # Target intelligence (pre-computed by CIRO)
    target_verified: bool
    target_is_proxy: bool
    target_has_source: bool
    target_first_seen: Optional[str]      # ISO 8601
    target_interaction_count: int         # total interactions on-chain
    target_unique_callers: int            # unique addresses that called it
    target_risk: RiskLevel

    # Behavioral context (computed by CIRO from on-chain data)
    sender_tx_count_24h: int
    sender_unique_targets_24h: int
    sender_avg_value_24h: str             # decimal string
    sender_max_value_24h: str             # decimal string
    sender_age_blocks: int                # blocks since first tx

    # Forta alerts (if any)
    forta_alert_ids: list[str] = field(default_factory=list)
    forta_severity: Optional[str] = None  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"

    # Sanctions
    is_sanctioned_sender: bool = False
    is_sanctioned_target: bool = False
    sanction_list: Optional[str] = None   # "OFAC", "EU", etc.


@dataclass
class LabeledTransactionResponse:
    """Response from GET /transactions/labeled."""
    transactions: list[LabeledTransaction]
    total: int                            # total matching (for pagination)
    offset: int
    limit: int
    query_time_ms: int


# ── GET /addresses/{addr}/risk ────────────────────────────────────────

@dataclass
class AddressRiskQuery:
    """Query for address risk assessment."""
    address: str                          # hex address
    chain: str = "starknet"


@dataclass
class AddressRiskResponse:
    """Address intelligence from CIRO."""
    address: str
    chain: str
    risk: RiskLevel
    risk_score: int                       # 0-100

    # On-chain profile
    first_seen: Optional[str]             # ISO 8601
    last_seen: Optional[str]
    total_tx_count: int
    total_value_sent: str                 # decimal
    total_value_received: str             # decimal
    unique_counterparties: int
    is_contract: bool
    is_verified: bool
    is_proxy: bool
    has_source: bool

    # Threat signals
    is_sanctioned: bool
    sanction_lists: list[str]             # ["OFAC", "EU", ...]
    forta_alerts_count: int
    forta_severity_max: Optional[str]
    known_exploit_involvement: bool
    exploit_names: list[str]              # ["Euler Finance", ...]

    # Labels from CIRO's data lake
    labeled_tx_count: int                 # how many txs are labeled
    malicious_tx_count: int
    suspicious_tx_count: int

    # Cluster intelligence
    cluster_id: Optional[str]             # address cluster (same entity)
    cluster_size: int                     # addresses in cluster
    cluster_risk: RiskLevel


# ── POST /transactions/enrich ─────────────────────────────────────────

@dataclass
class EnrichRequest:
    """Real-time enrichment request from ObelyZK classifier."""
    target: str                           # target address
    sender: Optional[str] = None          # sender address
    value: str = "0"                      # decimal string
    selector: str = "0x0"                 # function selector
    chain: str = "starknet"


@dataclass
class EnrichResponse:
    """Enriched features from CIRO for real-time classification."""
    # Target intelligence
    target_risk: RiskLevel
    target_risk_score: int                # 0-100
    target_verified: bool
    target_is_proxy: bool
    target_has_source: bool
    target_interaction_count: int
    target_unique_callers: int
    target_first_seen_blocks_ago: int     # 0 = brand new

    # Sanctions check
    is_sanctioned: bool
    sanction_lists: list[str]

    # Forta intelligence
    forta_alerts_24h: int                 # alerts in last 24h for this target
    forta_severity_max: Optional[str]

    # Behavioral context (if sender provided)
    sender_tx_frequency: int              # txs per hour (24h window)
    sender_unique_targets_24h: int
    sender_avg_value_24h: int             # quantized
    sender_max_value_24h: int             # quantized
    sender_age_blocks: int

    # CIRO confidence
    data_freshness_s: int                 # seconds since last index update
    enrichment_time_ms: int               # how long the enrichment took


# ── GET /alerts/recent ────────────────────────────────────────────────

@dataclass
class AlertQuery:
    """Query for recent alerts."""
    chain: str = "starknet"
    severity: Optional[str] = None        # filter by severity
    since: Optional[str] = None           # ISO 8601
    limit: int = 100


@dataclass
class Alert:
    """A single alert from CIRO (aggregated from Forta + custom bots)."""
    alert_id: str
    source: str                           # "forta", "ciro_heuristic", "community"
    severity: str                         # "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"
    chain: str
    timestamp: str                        # ISO 8601
    addresses: list[str]                  # involved addresses
    tx_hash: Optional[str]
    name: str                             # alert name
    description: str                      # human-readable description
    metadata: dict                        # source-specific metadata


@dataclass
class AlertResponse:
    """Response from GET /alerts/recent."""
    alerts: list[Alert]
    total: int


# ── GET /stats ────────────────────────────────────────────────────────

@dataclass
class DataLakeStats:
    """Statistics about CIRO's blockchain data lake."""
    total_transactions: int
    labeled_transactions: int
    label_distribution: dict              # {"safe": N, "suspicious": N, "malicious": N}
    source_distribution: dict             # {"forta_alert": N, "heuristic": N, ...}
    chains: list[str]
    last_indexed_block: dict              # {"starknet": 12345, "ethereum": 67890}
    last_updated: str                     # ISO 8601
    forta_bots_active: int
    addresses_indexed: int
    sanctioned_addresses: int


# ═══════════════════════════════════════════════════════════════════════
# API Endpoints Summary
# ═══════════════════════════════════════════════════════════════════════

API_ENDPOINTS = {
    "base_url": "/api/singularity/{org}/blockchain",

    "labeled_transactions": {
        "method": "GET",
        "path": "/transactions/labeled",
        "query": LabeledTransactionQuery,
        "response": LabeledTransactionResponse,
        "description": "Fetch labeled transactions for classifier training",
    },
    "address_risk": {
        "method": "GET",
        "path": "/addresses/{address}/risk",
        "query": AddressRiskQuery,
        "response": AddressRiskResponse,
        "description": "Get risk assessment for a specific address",
    },
    "enrich": {
        "method": "POST",
        "path": "/transactions/enrich",
        "body": EnrichRequest,
        "response": EnrichResponse,
        "description": "Real-time transaction enrichment for classifier",
    },
    "alerts": {
        "method": "GET",
        "path": "/alerts/recent",
        "query": AlertQuery,
        "response": AlertResponse,
        "description": "Recent alerts from Forta + custom detection bots",
    },
    "stats": {
        "method": "GET",
        "path": "/stats",
        "response": DataLakeStats,
        "description": "Data lake statistics and coverage",
    },
}
