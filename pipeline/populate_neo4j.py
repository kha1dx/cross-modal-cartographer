#!/usr/bin/env python3
"""
populate_neo4j.py
-----------------
Reads dataset/landmarks_registry.csv and populates an AuraDB Neo4j instance
with the full knowledge graph:

  Nodes   : Landmark, City, Region, HistoricalPeriod, ArchitecturalStyle
  Edges   : LOCATED_IN, IN_REGION, BUILT_DURING, HAS_STYLE, NEAR

Run from V0/:
    python populate_neo4j.py

Requires a .env file in V0/ with:
    NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=<your-password>
"""

import csv
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).parent / ".env")

URI      = os.environ.get("NEO4J_URI")
USER     = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not URI or not PASSWORD:
    sys.exit(
        "ERROR: NEO4J_URI and NEO4J_PASSWORD must be set in V0/.env\n"
        "Create a free AuraDB instance at https://console.neo4j.io"
    )

REGISTRY_PATH = Path(__file__).parent / ".." / "dataset" / "landmarks_registry.csv"
NEAR_KM_THRESHOLD = 5.0   # create NEAR edge for landmarks within this distance


# ── Haversine distance (km) ───────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Load registry ─────────────────────────────────────────────────────────────
def load_registry():
    rows = []
    with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["coordinates_lat"] = float(row["coordinates_lat"])
            row["coordinates_lon"] = float(row["coordinates_lon"])
            rows.append(row)
    return rows


# ── Cypher helpers ────────────────────────────────────────────────────────────
def create_constraints(session):
    """Unique constraints so MERGE works correctly."""
    constraints = [
        "CREATE CONSTRAINT landmark_name IF NOT EXISTS FOR (n:Landmark) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (n:City) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT region_name IF NOT EXISTS FOR (n:Region) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT period_name IF NOT EXISTS FOR (n:HistoricalPeriod) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT style_name IF NOT EXISTS FOR (n:ArchitecturalStyle) REQUIRE n.name IS UNIQUE",
    ]
    for c in constraints:
        session.run(c)
    print("Constraints created.")


def populate_nodes(session, rows):
    """MERGE all node types."""

    # HistoricalPeriod nodes with ordinal for chronological ordering
    period_order = {
        "Pharaonic":        1,
        "Greco-Roman":      2,
        "Coptic/Byzantine": 3,
        "Islamic":          4,
        "Ottoman":          5,
        "Contemporary":     6,
    }
    periods = {r["historical_era"] for r in rows}
    for p in periods:
        session.run(
            "MERGE (n:HistoricalPeriod {name: $name}) SET n.order = $order",
            name=p, order=period_order.get(p, 99)
        )

    # ArchitecturalStyle nodes
    styles = {r["architectural_style"] for r in rows}
    for s in styles:
        session.run("MERGE (n:ArchitecturalStyle {name: $name})", name=s)

    # Region nodes
    regions = {r["geographic_region"] for r in rows}
    for reg in regions:
        session.run("MERGE (n:Region {name: $name})", name=reg)

    # City nodes  →  IN_REGION relationship
    seen_cities = {}
    for r in rows:
        city = r["city"]
        if city not in seen_cities:
            session.run(
                """
                MERGE (c:City {name: $city})
                WITH c
                MATCH (reg:Region {name: $region})
                MERGE (c)-[:IN_REGION]->(reg)
                """,
                city=city, region=r["geographic_region"]
            )
            seen_cities[city] = True

    # Landmark nodes  →  LOCATED_IN / BUILT_DURING / HAS_STYLE
    for r in rows:
        session.run(
            """
            MERGE (lm:Landmark {name: $name})
            SET lm.landmark_type      = $ltype,
                lm.historical_era     = $era,
                lm.geographic_region  = $region,
                lm.city               = $city,
                lm.architectural_style = $style,
                lm.lat                = $lat,
                lm.lon                = $lon
            WITH lm
            MATCH (c:City           {name: $city})
            MATCH (p:HistoricalPeriod {name: $era})
            MATCH (s:ArchitecturalStyle {name: $style})
            MERGE (lm)-[:LOCATED_IN]->(c)
            MERGE (lm)-[:BUILT_DURING]->(p)
            MERGE (lm)-[:HAS_STYLE]->(s)
            """,
            name=r["canonical_name"],
            ltype=r["landmark_type"],
            era=r["historical_era"],
            region=r["geographic_region"],
            city=r["city"],
            style=r["architectural_style"],
            lat=r["coordinates_lat"],
            lon=r["coordinates_lon"],
        )

    print(f"Nodes + direct relationships created for {len(rows)} landmarks.")


def populate_near_edges(session, rows):
    """Create NEAR relationships for landmarks within NEAR_KM_THRESHOLD km."""
    count = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            dist = haversine(
                a["coordinates_lat"], a["coordinates_lon"],
                b["coordinates_lat"], b["coordinates_lon"],
            )
            if dist <= NEAR_KM_THRESHOLD:
                session.run(
                    """
                    MATCH (a:Landmark {name: $a}), (b:Landmark {name: $b})
                    MERGE (a)-[r:NEAR]->(b) SET r.dist_km = $dist
                    MERGE (b)-[r2:NEAR]->(a) SET r2.dist_km = $dist
                    """,
                    a=a["canonical_name"], b=b["canonical_name"],
                    dist=round(dist, 3),
                )
                count += 1
    print(f"NEAR relationships created: {count} pairs within {NEAR_KM_THRESHOLD} km.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rows = load_registry()
    print(f"Registry loaded: {len(rows)} landmarks")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"Connected to AuraDB: {URI}")
    except Exception as exc:
        driver.close()
        sys.exit(f"Connection failed: {exc}")

    with driver.session() as session:
        create_constraints(session)
        populate_nodes(session, rows)
        populate_near_edges(session, rows)

    # Verification queries
    with driver.session() as session:
        n_landmarks = session.run("MATCH (n:Landmark) RETURN count(n) AS c").single()["c"]
        n_cities    = session.run("MATCH (n:City)     RETURN count(n) AS c").single()["c"]
        n_near      = session.run("MATCH ()-[r:NEAR]->() RETURN count(r) AS c").single()["c"]
        print(f"\nVerification:")
        print(f"  Landmark nodes : {n_landmarks}")
        print(f"  City nodes     : {n_cities}")
        print(f"  NEAR edges     : {n_near}")

    driver.close()
    print("\nDone — AuraDB populated.")


if __name__ == "__main__":
    main()
