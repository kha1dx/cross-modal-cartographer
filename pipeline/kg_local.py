"""
kg_local.py
-----------
In-memory Knowledge Graph client using NetworkX.
Drop-in replacement for KGClient (kg_query.py) that requires no Neo4j connection.

Usage:
    from kg_local import LocalKGClient

    client = LocalKGClient(faiss_metadata_list)
    enriched = client.enrich(["Great_Pyramid_of_Giza", "Luxor_Temple"])
    client.close()
"""

import math
from typing import List, Dict, Optional

import networkx as nx

from build_registry import (
    CITY_LOOKUP, CITY_COORDS, CITY_TO_REGION,
    classify_type, classify_era, derive_style,
)

NEAR_KM_THRESHOLD = 5.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km (same formula as populate_neo4j.py)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class LocalKGClient:
    """
    In-memory KG built from FAISS metadata at startup.
    Same public API as KGClient so the backend can swap transparently.
    """

    def __init__(self, metadata_list: list):
        self._graph = nx.DiGraph()
        self._landmarks: Dict[str, dict] = {}  # name -> properties
        self._build_graph(metadata_list)

    def close(self):
        """No-op — nothing to close for in-memory graph."""
        pass

    # ── Graph construction ───────────────────────────────────────────────────
    def _build_graph(self, metadata_list: list):
        # Deduplicate: one entry per landmark_name (take first occurrence)
        seen = set()
        unique_landmarks = []
        for m in metadata_list:
            name = m.get("landmark_name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            unique_landmarks.append(m)

        # Re-classify each landmark using updated build_registry functions
        for m in unique_landmarks:
            name = m["landmark_name"]
            ltype = classify_type(name)
            era = classify_era(name)
            city = CITY_LOOKUP.get(name, m.get("city", "Cairo"))
            region = CITY_TO_REGION.get(city, "Greater Cairo")
            coords = CITY_COORDS.get(city, (30.0444, 31.2357))
            style = derive_style(ltype, era)

            props = {
                "landmark_name": name,
                "landmark_type": ltype,
                "historical_era": era,
                "city": city,
                "geographic_region": region,
                "architectural_style": style,
                "lat": coords[0],
                "lon": coords[1],
            }
            self._landmarks[name] = props

            # Add nodes
            self._graph.add_node(f"lm:{name}", kind="Landmark", **props)
            self._graph.add_node(f"city:{city}", kind="City", name=city)
            self._graph.add_node(f"region:{region}", kind="Region", name=region)
            self._graph.add_node(f"era:{era}", kind="HistoricalPeriod", name=era)
            self._graph.add_node(f"style:{style}", kind="ArchitecturalStyle", name=style)

            # Add edges
            self._graph.add_edge(f"lm:{name}", f"city:{city}", rel="LOCATED_IN")
            self._graph.add_edge(f"city:{city}", f"region:{region}", rel="IN_REGION")
            self._graph.add_edge(f"lm:{name}", f"era:{era}", rel="BUILT_DURING")
            self._graph.add_edge(f"lm:{name}", f"style:{style}", rel="HAS_STYLE")

        # Build NEAR edges (pairwise within threshold)
        lm_list = list(self._landmarks.values())
        for i, a in enumerate(lm_list):
            for b in lm_list[i + 1:]:
                dist = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
                if dist <= NEAR_KM_THRESHOLD:
                    d = round(dist, 3)
                    self._graph.add_edge(
                        f"lm:{a['landmark_name']}", f"lm:{b['landmark_name']}",
                        rel="NEAR", dist_km=d,
                    )
                    self._graph.add_edge(
                        f"lm:{b['landmark_name']}", f"lm:{a['landmark_name']}",
                        rel="NEAR", dist_km=d,
                    )

        print(f"[LocalKG] Built graph: {len(self._landmarks)} landmarks, "
              f"{sum(1 for _, _, d in self._graph.edges(data=True) if d.get('rel') == 'NEAR') // 2} NEAR pairs")

    # ── Public API (matches KGClient) ────────────────────────────────────────
    def enrich(self, landmark_names: List[str]) -> List[Dict]:
        results = []
        for name in landmark_names:
            if name not in self._landmarks:
                results.append({"landmark_name": name, "found": False})
                continue

            props = dict(self._landmarks[name])
            props["found"] = True

            # Collect NEAR neighbours
            lm_node = f"lm:{name}"
            nearby = []
            if self._graph.has_node(lm_node):
                for _, target, data in self._graph.edges(lm_node, data=True):
                    if data.get("rel") == "NEAR" and target.startswith("lm:"):
                        nb_name = target[3:]  # strip "lm:" prefix
                        nearby.append({
                            "name": nb_name,
                            "dist_km": data.get("dist_km"),
                        })
            props["nearby_landmarks"] = nearby
            results.append(props)

        return results

    def check_schema_constraint(
        self,
        landmark_name: str,
        expected_type: Optional[str] = None,
        expected_era: Optional[str] = None,
    ) -> Dict:
        if landmark_name not in self._landmarks:
            return {
                "passes": False,
                "violations": [f"Landmark '{landmark_name}' not in KG"],
            }

        facts = self._landmarks[landmark_name]
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

    def same_type_in_city(self, city: str, landmark_type: str) -> List[str]:
        return [
            name for name, props in self._landmarks.items()
            if props["city"] == city and props["landmark_type"] == landmark_type
        ]
