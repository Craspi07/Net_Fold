"""Async fetchers for UniProt / STRING-DB and NetworkX graph assembly.

Network calls go through ``aiohttp`` and are cached locally (utils.cache)
so repeated runs against the same seed proteins do not re-hit the APIs.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import aiohttp
import networkx as nx

from utils.cache import cache

log = logging.getLogger("phasenet.network_builder")

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
STRING_MAP_URL = "https://string-db.org/api/json/get_string_ids"
STRING_NETWORK_URL = "https://string-db.org/api/json/network"
STRING_INTERACTION_PARTNERS_URL = "https://string-db.org/api/json/interaction_partners"

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)
ProgressCB = Optional[Callable[[int, str], None]]


@dataclass
class ProteinRecord:
    identifier: str
    uniprot_acc: Optional[str] = None
    gene_name: Optional[str] = None
    sequence: str = ""
    length: int = 0
    disorder_regions: List[Dict] = field(default_factory=list)


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict, namespace: str):
    cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    cached = cache.get(namespace, cache_key)
    if cached is not None:
        return cached
    try:
        async with session.get(url, params=params, timeout=DEFAULT_TIMEOUT) as resp:
            if resp.status != 200:
                log.warning("Request failed (%s): %s", resp.status, url)
                return None
            data = await resp.json(content_type=None)
            cache.set(namespace, cache_key, data)
            return data
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        log.warning("Network error for %s: %s", url, exc)
        return None


async def fetch_uniprot_entry(session: aiohttp.ClientSession, query_id: str) -> ProteinRecord:
    """Resolve a UniProt ID or gene symbol to sequence + basic metadata."""
    params = {
        "query": f'(gene:{query_id} OR accession:{query_id}) AND reviewed:true',
        "fields": "accession,gene_names,sequence,ft_disorder,length",
        "format": "json",
        "size": "1",
    }
    data = await _get_json(session, UNIPROT_SEARCH_URL, params, "uniprot")
    record = ProteinRecord(identifier=query_id)
    if not data or not data.get("results"):
        log.warning("No UniProt entry found for '%s'", query_id)
        return record

    entry = data["results"][0]
    record.uniprot_acc = entry.get("primaryAccession")
    genes = entry.get("genes") or []
    if genes:
        record.gene_name = genes[0].get("geneName", {}).get("value")
    seq_block = entry.get("sequence", {})
    record.sequence = seq_block.get("value", "")
    record.length = seq_block.get("length", len(record.sequence))

    for feature in entry.get("features", []):
        if feature.get("type") == "Disordered region" or feature.get("type") == "Region":
            loc = feature.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start and end:
                record.disorder_regions.append({"start": start, "end": end,
                                                  "description": feature.get("description", "")})
    return record


async def fetch_string_ids(session: aiohttp.ClientSession, identifiers: List[str], species: int) -> Dict[str, str]:
    """Map free-text identifiers to STRING internal IDs."""
    params = {
        "identifiers": "\r".join(identifiers),
        "species": str(species),
        "limit": "1",
        "echo_query": "1",
    }
    data = await _get_json(session, STRING_MAP_URL, params, "string_map")
    mapping: Dict[str, str] = {}
    if not data:
        return mapping
    for row in data:
        mapping[row.get("queryItem", row.get("preferredName", ""))] = row.get("stringId")
    return mapping


async def fetch_string_network(session: aiohttp.ClientSession, string_ids: List[str], species: int,
                                min_score: float) -> List[Dict]:
    """Fetch confidence-weighted STRING functional/physical interactions."""
    params = {
        "identifiers": "\r".join(string_ids),
        "species": str(species),
        "required_score": str(int(min_score * 1000)),
    }
    data = await _get_json(session, STRING_NETWORK_URL, params, "string_network")
    return data or []


async def fetch_string_partners(session: aiohttp.ClientSession, string_ids: List[str], species: int,
                                  min_score: float, limit: int = 10) -> List[Dict]:
    """Fetch additional interaction partners for graph expansion."""
    params = {
        "identifiers": "\r".join(string_ids),
        "species": str(species),
        "required_score": str(int(min_score * 1000)),
        "limit": str(limit),
    }
    data = await _get_json(session, STRING_INTERACTION_PARTNERS_URL, params, "string_partners")
    return data or []


class NetworkBuilder:
    """Orchestrates UniProt + STRING-DB fetching into a NetworkX graph."""

    def __init__(self, species: int = 9606, min_score: float = 0.4, expand_depth: int = 1):
        self.species = species
        self.min_score = min_score
        self.expand_depth = max(0, min(expand_depth, 2))

    async def build(self, seed_ids: List[str], progress_cb: ProgressCB = None) -> nx.Graph:
        def report(pct: int, msg: str) -> None:
            log.info(msg)
            if progress_cb:
                progress_cb(pct, msg)

        seed_ids = [s.strip() for s in seed_ids if s.strip()]
        if not seed_ids:
            raise ValueError("No protein identifiers supplied.")

        graph = nx.Graph()
        records: Dict[str, ProteinRecord] = {}

        connector = aiohttp.TCPConnector(limit=8)
        async with aiohttp.ClientSession(connector=connector) as session:
            report(5, f"Resolving {len(seed_ids)} UniProt identifiers...")
            uniprot_tasks = [fetch_uniprot_entry(session, sid) for sid in seed_ids]
            uniprot_results = await asyncio.gather(*uniprot_tasks, return_exceptions=True)

            for sid, res in zip(seed_ids, uniprot_results):
                if isinstance(res, Exception):
                    log.warning("UniProt fetch failed for %s: %s", sid, res)
                    res = ProteinRecord(identifier=sid)
                records[sid] = res
                label = res.gene_name or sid
                graph.add_node(
                    sid,
                    label=label,
                    uniprot_acc=res.uniprot_acc,
                    sequence=res.sequence,
                    length=res.length,
                    disorder_regions=res.disorder_regions,
                    seed=True,
                )

            report(25, "Mapping identifiers to STRING-DB...")
            id_map = await fetch_string_ids(session, seed_ids, self.species)
            string_to_query = {v: k for k, v in id_map.items()}
            current_string_ids = list(id_map.values())

            frontier = set(current_string_ids)
            all_string_ids = set(current_string_ids)

            for depth in range(self.expand_depth + 1):
                if not frontier:
                    break
                report(30 + depth * 15, f"Fetching STRING-DB interactions (hop {depth})...")
                edges = await fetch_string_network(session, list(frontier), self.species, self.min_score)

                new_frontier = set()
                for e in edges:
                    a, b = e.get("stringId_A"), e.get("stringId_B")
                    a_name = e.get("preferredName_A", a)
                    b_name = e.get("preferredName_B", b)
                    score = float(e.get("score", 0)) / 1000.0
                    if score < self.min_score:
                        continue
                    key_a = string_to_query.get(a, a_name)
                    key_b = string_to_query.get(b, b_name)
                    for key, sid_full, name in ((key_a, a, a_name), (key_b, b, b_name)):
                        if key not in graph:
                            graph.add_node(key, label=name, uniprot_acc=None, sequence="",
                                            length=0, disorder_regions=[], seed=False)
                            string_to_query[sid_full] = key
                            all_string_ids.add(sid_full)
                    if not graph.has_edge(key_a, key_b):
                        graph.add_edge(key_a, key_b, weight=score, confidence=score)
                    for sid_full in (a, b):
                        if sid_full not in all_string_ids:
                            new_frontier.add(sid_full)
                frontier = new_frontier - all_string_ids
                all_string_ids |= frontier

            report(70, "Fetching sequences for newly discovered nodes...")
            missing = [n for n, d in graph.nodes(data=True) if not d.get("sequence")]
            if missing:
                fetch_tasks = [fetch_uniprot_entry(session, n) for n in missing]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                for n, res in zip(missing, results):
                    if isinstance(res, Exception) or res is None:
                        continue
                    graph.nodes[n]["sequence"] = res.sequence
                    graph.nodes[n]["length"] = res.length
                    graph.nodes[n]["uniprot_acc"] = res.uniprot_acc or graph.nodes[n].get("uniprot_acc")
                    graph.nodes[n]["disorder_regions"] = res.disorder_regions
                    if not graph.nodes[n].get("label"):
                        graph.nodes[n]["label"] = res.gene_name or n

        report(95, f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        if graph.number_of_nodes() == 0:
            raise RuntimeError("Failed to build any network nodes - check identifiers and connectivity.")
        report(100, "Network construction complete.")
        return graph


def build_network_sync(seed_ids: List[str], species: int = 9606, min_score: float = 0.4,
                        expand_depth: int = 1, progress_cb: ProgressCB = None) -> nx.Graph:
    """Convenience synchronous wrapper (used outside the async worker)."""
    builder = NetworkBuilder(species=species, min_score=min_score, expand_depth=expand_depth)
    return asyncio.run(builder.build(seed_ids, progress_cb=progress_cb))
