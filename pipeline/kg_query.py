"""
kg_query.py
-----------
Knowledge-graph enrichment for FAISS retrieval results.

Usage:
    from kg_query import KGClient

    client = KGClient()                          # reads creds from .env
    enriched = client.enrich(["Great_Pyramid_of_Giza", "Luxor_Temple"])
    client.close()

Each enriched entry is a dict with keys:
    landmark_name, landmark_type, historical_era, city, geographic_region,
    architectural_style, lat, lon,
    nearby_landmarks  — list of {name, dist_km} for NEAR neighbours
"""

import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).parent / ".env")

_URI      = os.environ.get("NEO4J_URI")
_USER     = os.environ.get("NEO4J_USER", "neo4j")
_PASSWORD = os.environ.get("NEO4J_PASSWORD")


class KGClient:
    """Thin wrapper around the Neo4j driver for KG enrichment queries."""

    def __init__(self):
        if not _URI or not _PASSWORD:
            raise EnvironmentError(
                "NEO4J_URI and NEO4J_PASSWORD must be set in V0/.env"
            )
        self._driver = GraphDatabase.driver(_URI, auth=(_USER, _PASSWORD))

    def close(self):
        self._driver.close()

    # ── Primary: enrich a list of landmark names ──────────────────────────────
    def enrich(self, landmark_names: List[str]) -> List[Dict]:
        """
        Given a list of canonical landmark names (as returned by FAISS metadata),
        returns a list of enriched dicts, one per name found in the KG.

        Unknown names are included with a 'found': False flag.
        """
        results = []
        with self._driver.session() as session:
            for name in landmark_names:
                rec = self._query_one(session, name)
                results.append(rec)
        return results

    def _query_one(self, session, name: str) -> Dict:
        cypher = """
        MATCH (lm:Landmark {name: $name})
        OPTIONAL MATCH (lm)-[:NEAR]->(nb:Landmark)
        RETURN
            lm.name                AS landmark_name,
            lm.landmark_type       AS landmark_type,
            lm.historical_era      AS historical_era,
            lm.city                AS city,
            lm.geographic_region   AS geographic_region,
            lm.architectural_style AS architectural_style,
            lm.lat                 AS lat,
            lm.lon                 AS lon,
            collect({name: nb.name, dist_km: [(lm)-[r:NEAR]->(nb) | r.dist_km][0]})
                AS nearby_landmarks
        """
        result = session.run(cypher, name=name).single()
        if result is None:
            return {"landmark_name": name, "found": False}
        rec = dict(result)
        rec["found"] = True
        # Filter out null entries in nearby_landmarks (when OPTIONAL MATCH finds nothing)
        rec["nearby_landmarks"] = [
            nb for nb in rec.get("nearby_landmarks", [])
            if nb.get("name") is not None
        ]
        return rec

    # ── Convenience: S-constraint check ──────────────────────────────────────
    def check_schema_constraint(
        self,
        landmark_name: str,
        expected_type: str = None,
        expected_era: str = None,
    ) -> Dict:
        """
        Returns {'passes': bool, 'violations': list[str]}.
        Used by the Phase D verification module to detect S-violations.
        """
        facts = self._query_one(self._driver.session(), landmark_name)
        if not facts.get("found"):
            return {"passes": False, "violations": [f"Landmark '{landmark_name}' not in KG"]}

        violations = []
        if expected_type and facts["landmark_type"] != expected_type:
            violations.append(
                f"type mismatch: KG={facts['landmark_type']}, expected={expected_type}"
            )
        if expected_era and facts["historical_era"] != expected_era:
            violations.append(
                f"era mismatch: KG={facts['historical_era']}, expected={expected_era}"
            )
        return {"passes": len(violations) == 0, "violations": violations}

    # ── Convenience: same-type neighbours ────────────────────────────────────
    def same_type_in_city(self, city: str, landmark_type: str) -> List[str]:
        """Return names of all landmarks of a given type in a city."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (lm:Landmark {landmark_type: $ltype})-[:LOCATED_IN]->(c:City {name: $city})
                RETURN lm.name AS name
                """,
                ltype=landmark_type, city=city,
            )
            return [r["name"] for r in result]
