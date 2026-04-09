"""
CIRO Intelligence Client — Training Data Pipeline

Fetches labeled transaction data from CIRO's blockchain data lake
and converts it to the ObelyZK classifier's 48-feature format.

Usage:
  from ciro.client import CiroClient

  client = CiroClient(base_url="https://ciro.example.com", api_key="...")
  features, labels = client.fetch_training_dataset(limit=50000)
  np.savez("dataset_ciro.npz", features=features, labels=labels)

Or from CLI:
  python -m ciro.client --url https://ciro.example.com --api-key $CIRO_API_KEY
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import urllib.request
    import urllib.error
except ImportError:
    raise ImportError("urllib required — included in Python stdlib")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import TransactionFeatures, encode_features, M31_MASK


class CiroClient:
    """Client for CIRO Intelligence blockchain data lake."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        org: str = "obelyzk",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.org = org
        self.timeout = timeout
        self._api_base = f"{self.base_url}/api/v1/blockchain"

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated request to CIRO."""
        url = f"{self._api_base}{path}"

        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query:
                url += f"?{query}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Client": "obelyzk-training/1.0",
        }

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"CIRO API error {e.code}: {e.reason}\n{error_body}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(f"CIRO unreachable: {e.reason}")

    # ── Training Data ──────────────────────────────────────────────────

    def fetch_labeled_transactions(
        self,
        network: Optional[str] = None,
        risk_level: Optional[str] = None,
        label_source: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        """
        Fetch labeled transactions from CIRO's data lake.
        Maps to: GET /api/v1/blockchain/transactions/labeled
        """
        params = {
            "network": network,
            "riskLevel": risk_level,
            "labelSource": label_source,
            "limit": limit,
            "offset": offset,
        }
        return self._request("GET", "/transactions/labeled", params=params)

    def fetch_all_labeled(
        self,
        network: Optional[str] = "STARKNET",
        max_total: int = 100000,
        page_size: int = 1000,
    ) -> list[dict]:
        """Fetch all labeled transactions with pagination."""
        all_txs = []
        offset = 0

        while offset < max_total:
            resp = self.fetch_labeled_transactions(
                network=network,
                limit=page_size,
                offset=offset,
            )
            txs = resp.get("transactions", [])
            if not txs:
                break

            all_txs.extend(txs)
            offset += len(txs)

            total = resp.get("total", 0)
            print(f"  Fetched {len(all_txs)}/{total} transactions...")

            if len(txs) < page_size:
                break

            time.sleep(0.5)  # rate limit

        return all_txs

    # ── Real-Time Enrichment ───────────────────────────────────────────

    def enrich_transaction(
        self,
        target: str,
        sender: Optional[str] = None,
        value: str = "0",
        selector: str = "0x0",
    ) -> dict:
        """Get real-time enrichment for a transaction."""
        body = {
            "target": target,
            "sender": sender,
            "value": value,
            "selector": selector,
            "chain": "starknet",
        }
        return self._request("POST", "/transactions/enrich", body=body)

    def get_address_risk(self, address: str) -> dict:
        """Get risk assessment for an address."""
        return self._request("GET", f"/addresses/{address}/risk")

    def get_recent_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """Get recent alerts."""
        params = {"severity": severity, "limit": limit}
        return self._request("GET", "/alerts/recent", params=params)

    def get_stats(self) -> dict:
        """Get data lake statistics."""
        return self._request("GET", "/stats")

    # ── Feature Conversion ─────────────────────────────────────────────

    @staticmethod
    def ciro_tx_to_features(tx: dict) -> TransactionFeatures:
        """
        Convert a CIRO LabeledTransaction to ObelyZK's TransactionFeatures.

        CIRO response fields (from blockchain-intelligence.service.ts):
          txHash, network, blockNumber, fromAddress, toAddress, value,
          status, functionName, labels[], riskLevel, riskScore,
          labelSource, blockTimestamp
        """
        # Parse value
        value_str = tx.get("value", "0") or "0"
        try:
            value = int(value_str) if not value_str.startswith("0x") else int(value_str, 16)
        except (ValueError, TypeError):
            value = 0

        # Parse function selector from functionName (CIRO provides name, not raw selector)
        # Map common function names to selectors
        func_name = tx.get("functionName", "") or ""
        FUNC_TO_SELECTOR = {
            "transfer": 0xA9059CBB,
            "transferFrom": 0x23B872DD,
            "approve": 0x095EA7B3,
            "swap": 0x38ED1739,
            "swapExactTokensForTokens": 0x38ED1739,
            "swapExactETHForTokens": 0x7FF36AB5,
            "flashLoan": 0x5CFFE9DE,
            "multicall": 0xAC9650D8,
            "withdraw": 0x3CCFD60B,
            "deposit": 0xB6B55F25,
            "mint": 0x40C10F19,
        }
        selector = FUNC_TO_SELECTOR.get(func_name, 0)

        # Compute derived features
        log2_val = max(0, value.bit_length() - 1) if value > 0 else 0

        # Risk-based features (CIRO provides riskLevel and riskScore)
        risk_level = tx.get("riskLevel", "UNKNOWN")
        risk_score = tx.get("riskScore", 0) or 0

        # Labels from CIRO
        labels = tx.get("labels", []) or []

        # Use target address (toAddress in CIRO)
        target = tx.get("toAddress", "0x0") or "0x0"

        # Selector feature flags
        TRANSFER_SELS = {0xA9059CBB, 0x23B872DD}
        APPROVE_SELS = {0x095EA7B3}
        SWAP_SELS = {0x38ED1739, 0x7FF36AB5, 0x18CBAFE5}

        return TransactionFeatures(
            target=target,
            value=value,
            selector=selector,
            calldata_len=0,  # CIRO doesn't provide raw calldata
            agent_trust_score=int(risk_score * 1000),  # map 0-100 → 0-100000
            agent_strikes=0,
            agent_age_blocks=0,
            is_verified="verified" in labels or risk_level == "LOW",
            is_proxy="proxy" in labels,
            has_source="source" in labels,
            interaction_count=0,  # not available per-tx from CIRO
            log2_value=log2_val,
            value_balance_ratio=min(int(risk_score * 1000), 100000),
            is_max_approval=value >= 2**128 - 1,
            is_zero_value=value == 0,
            is_transfer=selector in TRANSFER_SELS,
            is_approve=selector in APPROVE_SELS,
            is_swap=selector in SWAP_SELS,
            is_unknown=selector not in (TRANSFER_SELS | APPROVE_SELS | SWAP_SELS) and selector != 0,
            tx_frequency=0,
            unique_targets_24h=0,
            avg_value_24h=0,
            max_value_24h=0,
        )

    @staticmethod
    def ciro_label_to_int(risk_level: str) -> int:
        """
        Convert CIRO riskLevel to classifier label integer.
        CIRO levels: CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN
        Classifier labels: 0=safe, 1=suspicious, 2=malicious
        """
        mapping = {
            "LOW": 0,           # safe
            "UNKNOWN": 1,       # suspicious (insufficient data)
            "MEDIUM": 1,        # suspicious
            "HIGH": 2,          # malicious
            "CRITICAL": 2,      # malicious
        }
        return mapping.get(risk_level, 1)

    # ── Full Training Pipeline ─────────────────────────────────────────

    def fetch_training_dataset(
        self,
        network: Optional[str] = "STARKNET",
        max_total: int = 100000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fetch labeled transactions from CIRO and convert to
        ObelyZK's (N, 64) feature matrix + (N,) label array.
        """
        print(f"Fetching labeled transactions from CIRO (network={network})...")
        txs = self.fetch_all_labeled(
            network=network,
            max_total=max_total,
        )

        if not txs:
            raise RuntimeError("No labeled transactions found in CIRO data lake")

        features_list = []
        labels_list = []

        for tx in txs:
            try:
                tx_features = self.ciro_tx_to_features(tx)
                encoded = encode_features(tx_features)
                label = self.ciro_label_to_int(tx.get("riskLevel", "UNKNOWN"))
                features_list.append(encoded)
                labels_list.append(label)
            except Exception as e:
                # Skip malformed transactions
                continue

        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        n_safe = np.sum(labels == 0)
        n_susp = np.sum(labels == 1)
        n_mal = np.sum(labels == 2)

        print(f"Dataset: {len(labels)} samples")
        print(f"  safe:       {n_safe} ({100*n_safe/len(labels):.1f}%)")
        print(f"  suspicious: {n_susp} ({100*n_susp/len(labels):.1f}%)")
        print(f"  malicious:  {n_mal} ({100*n_mal/len(labels):.1f}%)")

        return features, labels

    def enrich_to_features(
        self,
        target: str,
        sender: Optional[str] = None,
        value: str = "0",
        selector: str = "0x0",
    ) -> np.ndarray:
        """
        Real-time: enrich a transaction via CIRO and return 64-element
        feature vector ready for the classifier.
        """
        enrichment = self.enrich_transaction(target, sender, value, selector)

        tx = TransactionFeatures(
            target=target,
            value=int(value),
            selector=int(selector, 16) if selector.startswith("0x") else int(selector),
            calldata_len=0,
            agent_trust_score=0,
            agent_strikes=0,
            agent_age_blocks=enrichment.get("sender_age_blocks", 0),
            is_verified=enrichment.get("target_verified", False),
            is_proxy=enrichment.get("target_is_proxy", False),
            has_source=enrichment.get("target_has_source", False),
            interaction_count=enrichment.get("target_interaction_count", 0),
            log2_value=max(0, int(value).bit_length() - 1) if int(value) > 0 else 0,
            value_balance_ratio=0,
            is_max_approval=int(value) >= 2**128 - 1,
            is_zero_value=int(value) == 0,
            is_transfer=False,
            is_approve=False,
            is_swap=False,
            is_unknown=True,
            tx_frequency=enrichment.get("sender_tx_frequency", 0),
            unique_targets_24h=enrichment.get("sender_unique_targets_24h", 0),
            avg_value_24h=enrichment.get("sender_avg_value_24h", 0),
            max_value_24h=enrichment.get("sender_max_value_24h", 0),
        )

        return encode_features(tx)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch training data from CIRO")
    parser.add_argument("--url", required=True, help="CIRO base URL")
    parser.add_argument("--api-key", required=True, help="CIRO API key")
    parser.add_argument("--org", default="obelyzk", help="CIRO org slug")
    parser.add_argument("--output", default="dataset_ciro.npz")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-total", type=int, default=100000)
    parser.add_argument("--stats", action="store_true", help="Print data lake stats")
    args = parser.parse_args()

    client = CiroClient(
        base_url=args.url,
        api_key=args.api_key,
        org=args.org,
    )

    if args.stats:
        stats = client.get_stats()
        print(json.dumps(stats, indent=2))
        return

    features, labels = client.fetch_training_dataset(
        min_confidence=args.min_confidence,
        max_total=args.max_total,
    )

    np.savez(args.output, features=features, labels=labels)
    print(f"Saved to {args.output}")
    print(f"\nTrain with: python train.py --dataset {args.output}")


if __name__ == "__main__":
    main()
