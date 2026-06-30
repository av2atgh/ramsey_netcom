"""Build cumulative arXiv co-authorship networks from the arXiv metadata dump.

Reads the Kaggle arXiv metadata snapshot (JSON-lines), keeps papers in a chosen
archive (ARXIV_CATEGORY, default "cond-mat"; matches "<cat>" or "<cat>.*"), and
builds the co-authorship graph (an edge between every pair of co-authors of a
paper). One *cumulative* network is written per year Y -- it contains all authors
and co-authorships of that archive's papers submitted up to and including Y --
saved as examples/data/<prefix>_coauthor_upto_<Y>.gt.gz (prefix = category with
'-' and '.' removed/underscored), plus an index CSV with (year, n_nodes, n_edges).

Author identity = "Last, First" from authors_parsed (see author_key; switch to
last name + first initial for Newman-style disambiguation). Paper year = the
first version's submission year.

Set ARXIV_LIMIT=<k> to stop after k matching papers (quick test).

Run from anywhere (large one-time read of the ~5 GB dump):
    ARXIV_CATEGORY=math python examples/build_coauthor.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
JSON_PATH = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot_20260628.json")
CATEGORY = os.environ.get("ARXIV_CATEGORY", "cond-mat")  # archive to build
PREFIX = CATEGORY.replace("-", "").replace(".", "_")  # filename-safe label
INDEX_CSV = os.path.join(DATA_DIR, f"{PREFIX}_coauthor_index.csv")
_CATEGORY_BYTES = CATEGORY.encode()  # cheap substring pre-filter before JSON parse

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def is_category(categories):
    for c in categories.split():
        if c == CATEGORY or c.startswith(CATEGORY + "."):
            return True
    return False


def paper_year(record):
    versions = record.get("versions")
    created = versions[0]["created"] if versions else record.get("update_date", "")
    m = _YEAR_RE.search(created)
    return int(m.group(1)) if m else None


def author_key(parsed):
    """'Last, First' from an authors_parsed entry [last, first, suffix]."""
    last, first = parsed[0].strip(), parsed[1].strip()
    return " ".join((f"{last}, {first}".strip().strip(",")).split())


def read_papers():
    """First pass: collect (year, [author_id]) for cond-mat papers; intern names."""
    limit = int(os.environ.get("ARXIV_LIMIT", "0"))
    author_id, id_name = {}, []
    papers = []

    def kid(name):
        i = author_id.get(name)
        if i is None:
            i = author_id[name] = len(id_name)
            id_name.append(name)
        return i

    with open(JSON_PATH, "rb") as f:
        for raw in f:
            if _CATEGORY_BYTES not in raw:  # cheap pre-filter before JSON parse
                continue
            rec = json.loads(raw)
            if not is_category(rec.get("categories", "")):
                continue
            year = paper_year(rec)
            if year is None:
                continue
            ids = [kid(author_key(p)) for p in rec.get("authors_parsed", []) if p and p[0].strip()]
            if ids:
                papers.append((year, ids))
                if limit and len(papers) >= limit:
                    break
    return papers, id_name


def build_and_save(papers, id_name):
    """Second pass: grow one graph in year order, saving a snapshot per year."""
    by_year = defaultdict(list)
    for year, ids in papers:
        by_year[year].append(ids)

    g = gt.Graph(directed=False)
    label = g.new_vertex_property("string")
    g.vertex_properties["label"] = label
    gv = {}            # author_id -> graph vertex index
    edge_set = set()   # {(a, b)} with a < b, to keep the graph simple
    index = []

    for year in sorted(by_year):
        for ids in by_year[year]:
            vs = []
            for aid in ids:
                v = gv.get(aid)
                if v is None:
                    v = int(g.add_vertex())
                    gv[aid] = v
                    label[v] = id_name[aid]
                vs.append(v)
            for a, b in combinations(sorted(set(vs)), 2):
                if (a, b) not in edge_set:
                    edge_set.add((a, b))
                    g.add_edge(a, b)
        out = os.path.join(DATA_DIR, f"{PREFIX}_coauthor_upto_{year}.gt.gz")
        g.save(out)
        index.append((year, g.num_vertices(), g.num_edges()))
        print(f"{year}: n={g.num_vertices()} E={g.num_edges()}  -> {os.path.basename(out)}")

    with open(INDEX_CSV, "w") as f:
        f.write("year,n_nodes,n_edges\n")
        for year, n, e in index:
            f.write(f"{year},{n},{e}\n")
    print("saved", INDEX_CSV)


def main():
    print(f"reading {JSON_PATH} ...")
    papers, id_name = read_papers()
    print(f"{CATEGORY} papers: {len(papers)}  unique authors: {len(id_name)}")
    build_and_save(papers, id_name)


if __name__ == "__main__":
    main()
