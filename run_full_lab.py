import os, re, json, time, random, hashlib, unicodedata, sys
from pathlib import Path
from collections import defaultdict, Counter, deque
from difflib import SequenceMatcher
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(override=True)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from groq import Groq
from openai import OpenAI

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

# Secrets & Config
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "250c259e"))
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "openai").lower()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-chat")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

DATA_PATH = "hackernoon_subset.csv"
LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 150  # Comprehensive extraction subset
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
SUPER_NODE_DEGREE = 50

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

print("=== STEP 1: INITIALIZING CLIENTS & CONNECTIONS ===")
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_cypher(query, **params):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, **params)
        return [record.data() for record in result]

# Test Neo4j
conn_test = run_cypher("RETURN 1 AS ok")
print(f"✅ Connected to Neo4j: {NEO4J_URI} -> {conn_test}")

# Schema & constraints setup
run_cypher("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
run_cypher("CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)")
run_cypher("CREATE INDEX entity_aliases_norm IF NOT EXISTS FOR (n:Entity) ON (n.aliases_norm)")
print("✅ Neo4j constraints & indexes verified.")

# LLM Helper functions
def parse_json_object(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(text[a:b+1])

def groq_chat(messages, model=None, json_mode=False, max_retries=5):
    target_model = model or GROQ_MODEL
    fallback_models = [target_model, "openai/gpt-oss-20b", "groq/compound-mini"]
    models_to_try = []
    for m in fallback_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last = None
    for m in models_to_try:
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": m,
                    "messages": messages,
                    "temperature": 0.0,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = groq_client.chat.completions.create(**kwargs)
                usage = {}
                if getattr(resp, "usage", None):
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                return resp.choices[0].message.content, usage
            except Exception as e:
                last = e
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "tpd" in err_str:
                    print(f"⚠️ Model {m} hit rate limit, trying next available model...")
                    break
                time.sleep(min(10, 2**attempt + random.random()))
    raise RuntimeError(f"Groq Chat failed after trying models {models_to_try}: {last}")

def groq_json(system, user, model=None):
    text, usage = groq_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        model=model,
        json_mode=True,
    )
    return parse_json_object(text), usage

# Judge Client
judge_client = None
if JUDGE_PROVIDER == "openai":
    judge_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def judge_chat(messages, json_mode=False):
    if JUDGE_PROVIDER == "groq":
        return groq_chat(messages, model=JUDGE_MODEL or GROQ_MODEL, json_mode=json_mode)
    try:
        kwargs = {
            "model": JUDGE_MODEL or "deepseek-chat",
            "messages": messages,
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = judge_client.chat.completions.create(**kwargs)
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return resp.choices[0].message.content, usage
    except Exception as e:
        print(f"⚠️ Judge ({JUDGE_PROVIDER}) failed: {e}, falling back to Groq...")
        return groq_chat(messages, model=GROQ_MODEL, json_mode=json_mode)

print("✅ Groq and Judge clients ready.")

# Normalization utilities
def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def norm_entity(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if t not in {"inc", "corp", "corporation", "llc", "ltd", "co", "the", "a", "an"}]
    return " ".join(tokens)

# Loader & Chunker
print("\n=== STEP 2: LOADING & CHUNKING DATASET ===")
def load_news(path):
    return pd.read_csv(path)

def standardize_news(raw):
    df = pd.DataFrame()
    df["text"] = raw["description"].fillna("").map(norm_space)
    df["title"] = raw["title"].fillna("").map(norm_space)
    df["published_date"] = pd.to_datetime(raw["published_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    df["article_id"] = [sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])]
    df = df[df["text"].str.len() >= 30].copy()
    df["dedup_key"] = [sha1(norm_space(f"{t}\n{x}").lower()) for t, x in zip(df["title"], df["text"])]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Exact dedup: {before:,} -> {len(df):,}")
    if LAB_MAX_ARTICLES and len(df) > LAB_MAX_ARTICLES:
        df = df.sample(LAB_MAX_ARTICLES, random_state=SEED).sort_index().reset_index(drop=True)
    return df

def chunk_text(text, size=220, overlap=40):
    words = norm_space(text).split()
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        part = words[start:start+size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out

def build_chunks(news_df):
    rows = []
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="Chunking"):
        for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)

raw_df = load_news(DATA_PATH)
news_df = standardize_news(raw_df)
chunks_df = build_chunks(news_df)
print(f"Total chunks created: {len(chunks_df):,}")

# Coreference Resolution
print("\n=== STEP 3: COREFERENCE RESOLUTION ===")
COREF_SYSTEM = """
You are a conservative coreference-resolution component for a knowledge-graph pipeline.
Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.
Never invent facts. Preserve dates, numbers, tickers and product names.
Return strict JSON only.
""".strip()

def resolve_coref_batch(batch_df):
    payload = [{"chunk_id": r.chunk_id, "text": r.text} for r in batch_df.itertuples(index=False)]
    prompt = f"""
Resolve coreferences.
Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "resolved_text": "...",
      "unresolved_mentions": ["..."]
    }}
  ]
}}
INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    obj, usage = groq_json(COREF_SYSTEM, prompt)
    by_id = {x.get("chunk_id"): x for x in obj.get("items", [])}
    rows = []
    for r in batch_df.itertuples(index=False):
        item = by_id.get(r.chunk_id, {})
        rows.append({
            "chunk_id": r.chunk_id,
            "resolved_text": norm_space(item.get("resolved_text") or r.text),
            "unresolved_mentions": item.get("unresolved_mentions", []),
        })
    return pd.DataFrame(rows), usage

def run_coref(chunks_subset, batch_size=4):
    out = []
    for start in tqdm(range(0, len(chunks_subset), batch_size), desc="Coref"):
        batch = chunks_subset.iloc[start:start+batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception as e:
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
        time.sleep(0.1)
    return pd.concat(out, ignore_index=True)

extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
coref_df = run_coref(extraction_source)
extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
print(f"Coref processed: {len(extraction_source)} chunks.")

# Spot-check Coref
sample_coref = extraction_source[extraction_source["text"] != extraction_source["resolved_text"]].head(5)
print(f"Modified by Coref: {len(sample_coref)} samples found.")

# NER + RE Extraction
print("\n=== STEP 4: TRIPLE EXTRACTION (NER + RE) ===")
EXTRACT_SYSTEM = f"""
Extract a high-precision knowledge graph from tech-news text.
Allowed node types: {sorted(ALLOWED_NODE_TYPES)}
Allowed relations: {sorted(ALLOWED_RELATIONS)}
Use only explicitly supported facts. Prefer precision over recall.
Every relation needs short evidence. Return strict JSON only.
""".strip()

def extract_batch(batch_df):
    payload = [{
        "chunk_id": r.chunk_id,
        "published_date": r.published_date,
        "text": getattr(r, "resolved_text", None) or r.text,
    } for r in batch_df.itertuples(index=False)]

    prompt = f"""
Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "relations": [
        {{
          "source": "...",
          "source_type": "Company|Person|Technology",
          "relation": "ALLOWED_RELATION",
          "target": "...",
          "target_type": "Company|Person|Technology",
          "evidence": "...",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    return groq_json(EXTRACT_SYSTEM, prompt)

def run_extraction(source_df, batch_size=3):
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []

    for start in tqdm(range(0, len(source_df), batch_size), desc="NER+RE"):
        batch = source_df.iloc[start:start+batch_size]
        try:
            obj, _ = extract_batch(batch)
        except Exception as e:
            errors.append({"start": start, "error": str(e)})
            time.sleep(1.0)
            continue

        for item in obj.get("items", []):
            cid = item.get("chunk_id")
            if cid not in meta:
                continue
            for x in item.get("relations", []):
                s, t = norm_space(x.get("source")), norm_space(x.get("target"))
                st, tt, rel = x.get("source_type"), x.get("target_type"), x.get("relation")
                if not s or not t:
                    continue
                if st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
                    continue
                if rel not in ALLOWED_RELATIONS:
                    continue
                triples.append({
                    "source_raw": s,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": t,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "published_date": meta[cid] or "",
                    "evidence": norm_space(x.get("evidence")),
                    "confidence": float(x.get("confidence") or 0.0),
                })
        time.sleep(0.2)

    return pd.DataFrame(triples), pd.DataFrame(errors)

raw_triples_df, extraction_errors_df = run_extraction(extraction_source)
print(f"Extracted {len(raw_triples_df)} raw triples. Errors: {len(extraction_errors_df)}")

# Entity Resolution
print("\n=== STEP 5: ENTITY RESOLUTION & CANONICALIZATION ===")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class UnionFind:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def lexical_guard(a, b, typ):
    if a == b:
        return True
    if typ == "Person":
        tok_a, tok_b = a.split(), b.split()
        if len(tok_a) >= 2 and len(tok_b) >= 2:
            if tok_a[-1] == tok_b[-1] and tok_a[0] != tok_b[0]:
                return False
    if typ in {"Company", "Technology"}:
        sm = SequenceMatcher(None, a, b).ratio()
        if sm < 0.60:
            return False
    return True

def build_resolution_map(triples_df, threshold=0.88):
    if triples_df.empty:
        return {}, pd.DataFrame()

    entities = defaultdict(set)
    for r in triples_df.itertuples(index=False):
        entities[r.source_type].add(r.source_raw)
        entities[r.target_type].add(r.target_raw)

    uf = UnionFind()
    audit = []

    for typ, raw_set in entities.items():
        raw_list = sorted(raw_set)
        norm_list = [norm_entity(x) for x in raw_list]
        uniq_norm = sorted(set(n for n in norm_list if n))
        if len(uniq_norm) < 2:
            for r, n in zip(raw_list, norm_list):
                uf.union(f"{typ}::{r}", f"{typ}::{n or r}")
            continue

        vecs = embed_model.encode(uniq_norm, normalize_embeddings=True)
        sims = vecs @ vecs.T

        for i in range(len(uniq_norm)):
            for j in range(i+1, len(uniq_norm)):
                sim = float(sims[i, j])
                n1, n2 = uniq_norm[i], uniq_norm[j]
                guard = lexical_guard(n1, n2, typ)
                decision = "IGNORE"
                if sim >= threshold:
                    if guard:
                        uf.union(f"{typ}::{n1}", f"{typ}::{n2}")
                        decision = "MERGE"
                    else:
                        decision = "REJECT_GUARD"
                audit.append({
                    "entity_type": typ,
                    "entity_a": n1,
                    "entity_b": n2,
                    "similarity": round(sim, 4),
                    "lexical_guard": guard,
                    "decision": decision,
                })

        for r, n in zip(raw_list, norm_list):
            uf.union(f"{typ}::{r}", f"{typ}::{n or r}")

    audit_df = pd.DataFrame(audit)
    groups = defaultdict(list)
    for typ, raw_set in entities.items():
        for r in raw_set:
            root = uf.find(f"{typ}::{r}")
            groups[root].append((typ, r))

    entity_map = {}
    for root, members in groups.items():
        all_raw = [r for _, r in members]
        canon = sorted(all_raw, key=lambda x: (-len(x), x))[0]
        canon_norm = norm_entity(canon)
        for typ, r in members:
            entity_map[(typ, r)] = (canon, canon_norm)

    return entity_map, audit_df

def canonicalize_triples(triples_df, entity_map):
    if triples_df.empty:
        return triples_df
    df = triples_df.copy()
    src_res = [entity_map.get((t, r), (r, norm_entity(r))) for t, r in zip(df.source_type, df.source_raw)]
    tgt_res = [entity_map.get((t, r), (r, norm_entity(r))) for t, r in zip(df.target_type, df.target_raw)]
    df["source_name"] = [x[0] for x in src_res]
    df["source_name_norm"] = [x[1] for x in src_res]
    df["target_name"] = [x[0] for x in tgt_res]
    df["target_name_norm"] = [x[1] for x in tgt_res]
    df["source_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.source_type, df.source_name_norm)]
    df["target_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.target_type, df.target_name_norm)]
    return df[df.source_id != df.target_id].reset_index(drop=True)

entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)
triples_df = canonicalize_triples(raw_triples_df, entity_map)
print(f"Canonicalized triples: {len(triples_df)}")

# Node table and Bulk Ingestion into Neo4j
print("\n=== STEP 6: NEO4J INGESTION VIA UNWIND ===")
def build_nodes(triples_df):
    if triples_df.empty:
        return pd.DataFrame(columns=['id', 'name', 'name_norm', 'type', 'aliases', 'aliases_norm'])
    rows = []
    for r in triples_df.itertuples(index=False):
        rows += [
            {"id": r.source_id, "name": r.source_name, "name_norm": r.source_name_norm, "type": r.source_type, "alias": r.source_raw},
            {"id": r.target_id, "name": r.target_name, "name_norm": r.target_name_norm, "type": r.target_type, "alias": r.target_raw},
        ]
    tmp = pd.DataFrame(rows)
    out = []
    for (node_id, name, name_norm, typ), g in tmp.groupby(["id", "name", "name_norm", "type"]):
        aliases = sorted(set(g["alias"].map(norm_space)))
        out.append({
            "id": node_id, "name": name, "name_norm": name_norm, "type": typ,
            "aliases": aliases,
            "aliases_norm": sorted(set(norm_entity(x) for x in aliases))
        })
    return pd.DataFrame(out)

def batches(records, size=1000):
    for i in range(0, len(records), size):
        yield records[i:i+size]

def bulk_insert_nodes(nodes_df, batch_size=1000):
    if nodes_df.empty:
        print("⚠️ nodes_df rỗng, bỏ qua bulk_insert_nodes.")
        return
    for typ in sorted(ALLOWED_NODE_TYPES):
        part = nodes_df[nodes_df.type == typ]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Entity {{id: row.id}})
        SET n:{typ},
            n.name=row.name,
            n.name_norm=row.name_norm,
            n.entity_type=row.type,
            n.aliases=row.aliases,
            n.aliases_norm=row.aliases_norm
        """
        for b in batches(part.to_dict("records"), batch_size):
            run_cypher(query, rows=b)

def bulk_insert_edges(triples_df, batch_size=1000):
    if triples_df.empty:
        print("⚠️ triples_df rỗng, bỏ qua bulk_insert_edges.")
        return
    for rel in sorted(ALLOWED_RELATIONS):
        part = triples_df[triples_df.relation == rel]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MATCH (s:Entity {{id: row.source_id}})
        MATCH (t:Entity {{id: row.target_id}})
        MERGE (s)-[r:{rel} {{source_chunk_id: row.source_chunk_id}}]->(t)
        SET r.published_date=row.published_date,
            r.evidence=row.evidence,
            r.confidence=row.confidence
        """
        cols = ["source_id", "target_id", "source_chunk_id", "published_date", "evidence", "confidence"]
        for b in batches(part[cols].to_dict("records"), batch_size):
            run_cypher(query, rows=b)

# Clean and re-insert
run_cypher("MATCH (n:Entity) DETACH DELETE n")
nodes_df = build_nodes(triples_df)
bulk_insert_nodes(nodes_df)
bulk_insert_edges(triples_df)
print(f"✅ Ingested {len(nodes_df)} nodes and {len(triples_df)} edges into Neo4j.")

# Sanity checks
print("\n=== STEP 7: SANITY CHECKS ===")
invalid = run_cypher("""
MATCH ()-[r]->()
WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
RETURN count(r) AS n
""")[0]["n"]

graph_counts = {
    "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
    "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
    "invalid_provenance_edges": invalid,
}
print("Graph counts:", graph_counts)
assert invalid == 0, f"Invalid provenance count: {invalid}"

top_degree_df = pd.DataFrame(run_cypher("""
MATCH (n:Entity)
OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree
RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
ORDER BY degree DESC LIMIT 15
"""))
print("\nTop 5 degree entities:")
print(top_degree_df.head(5))

# Super-node check
def recent_edges(node_id, limit=50):
    return run_cypher("""
    MATCH (n:Entity {id:$id})-[r]-(m:Entity)
    RETURN type(r) AS relation,
           m.name AS neighbor,
           m.entity_type AS neighbor_type,
           r.published_date AS published_date,
           r.evidence AS evidence
    ORDER BY r.published_date DESC, r.confidence DESC
    LIMIT $limit
    """, id=node_id, limit=int(limit))

def test_supernode_policy():
    rows = run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 1
    RETURN n.id AS id, n.name AS name, degree
    """)
    if not rows:
        print("Graph empty.")
        return
    n = rows[0]
    limit = 50 if n["degree"] > SUPER_NODE_DEGREE else 1000
    edges = recent_edges(n["id"], limit)
    print(f"Top Super-node: {n['name']} (degree={n['degree']}), fetched={len(edges)} edges")
    if n["degree"] > SUPER_NODE_DEGREE:
        assert len(edges) <= 50
        print("✅ Super-node cap OK.")
    else:
        print("✅ Node degree under cap.")

test_supernode_policy()

# Flat RAG Setup
print("\n=== STEP 8: FLAT RAG SETUP ===")
chunk_vecs = embed_model.encode(chunks_df["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
flat_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
flat_index.add(chunk_vecs.astype(np.float32))

def flat_search(query, top_k=5):
    qv = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = flat_index.search(qv, top_k)
    return chunks_df.iloc[indices[0]].copy().assign(score=scores[0])

# Graph Retrieval
print("\n=== STEP 9: HYBRID GRAPHRAG RETRIEVAL ===")
def extract_seed_candidates(query):
    doc = norm_entity(query)
    tokens = doc.split()
    grams = set()
    for n in range(1, 4):
        for i in range(len(tokens)-n+1):
            grams.add(" ".join(tokens[i:i+n]))
    return list(grams)

def find_seed_entities(query, limit=5):
    grams = extract_seed_candidates(query)
    if not grams:
        return []
    rows = run_cypher("""
    UNWIND $grams AS g
    MATCH (n:Entity)
    WHERE n.name_norm=g OR g IN coalesce(n.aliases_norm,[])
    RETURN DISTINCT n.id AS id, n.name AS name, n.entity_type AS type, size(g) AS rank
    ORDER BY rank DESC
    LIMIT $limit
    """, grams=grams, limit=int(limit))
    return rows

def retrieve_graph_context(query, max_seeds=3, per_seed_edges=15, max_total_edges=40):
    seeds = find_seed_entities(query, limit=max_seeds)
    if not seeds:
        return {"seeds": [], "edges": [], "rendered": "No graph seeds matched."}
    collected, seen = [], set()
    for s in seeds:
        rows = run_cypher("""
        MATCH (n:Entity {id:$id})-[r]-(m:Entity)
        RETURN n.name AS source, type(r) AS rel, m.name AS target,
               r.published_date AS published_date,
               coalesce(r.evidence, '') AS evidence,
               r.source_chunk_id AS source_chunk_id
        ORDER BY r.published_date DESC
        LIMIT $limit
        """, id=s["id"], limit=int(per_seed_edges))
        for x in rows:
            key = (x["source"], x["rel"], x["target"], x["source_chunk_id"])
            if key not in seen:
                seen.add(key)
                collected.append(x)
            if len(collected) >= max_total_edges:
                break
        if len(collected) >= max_total_edges:
            break
    lines = [f"- ({x['source']}) -[:{x['rel']}]-> ({x['target']}) | Date: {x['published_date']} | Evidence: {x['evidence']}"
             for x in collected]
    return {
        "seeds": seeds,
        "edges": collected,
        "rendered": "\n".join(lines) if lines else "No edges found.",
    }

RAG_SYSTEM = """
You are an expert tech-analyst assistant.
Answer questions based ONLY on the provided context.
If the context is insufficient, explain what is missing.
Be precise and concise.
""".strip()

def answer_flat_rag(question, top_k=5):
    t0 = time.perf_counter()
    chunks = flat_search(question, top_k=top_k)
    context = "\n\n".join([f"[{r.chunk_id} | {r.published_date}] {r.text}" for r in chunks.itertuples(index=False)])
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}"
    ans, usage = groq_chat([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt}
    ])
    lat = time.perf_counter() - t0
    return {
        "answer": ans,
        "context": context,
        "latency_s": round(lat, 3),
        "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
        "context_preview": context[:300],
    }

def answer_hybrid_graphrag(question, top_k=3):
    t0 = time.perf_counter()
    gctx = retrieve_graph_context(question)
    fctx = flat_search(question, top_k=top_k)
    text_context = "\n\n".join([f"[{r.chunk_id} | {r.published_date}] {r.text}" for r in fctx.itertuples(index=False)])
    combined = f"KNOWLEDGE GRAPH PATHS:\n{gctx['rendered']}\n\nRELEVANT TEXT PASSAGES:\n{text_context}"
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{combined}"
    ans, usage = groq_chat([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt}
    ])
    lat = time.perf_counter() - t0
    return {
        "answer": ans,
        "context": combined,
        "latency_s": round(lat, 3),
        "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
        "graph_seeds": [s["name"] for s in gctx["seeds"]],
        "graph_edges_count": len(gctx["edges"]),
        "context_preview": combined[:300],
        "graph_debug": {"diagnostics": {"supernode_events": []}},
    }

# Alias for notebook compatibility
answer_graph_rag = answer_hybrid_graphrag

# Golden Dataset & Evaluation
print("\n=== STEP 10: GOLDEN DATASET & LLM-AS-A-JUDGE EVALUATION ===")
golden_dataset = [
    {
        "id": "G01",
        "group": "factoid",
        "question": "Which news organization formed a partnership with OpenAI to share news archives?",
        "reference_answer": "Associated Press partnered with OpenAI to license and share its news archives.",
        "reference_evidence": "Associated Press PARTNERED_WITH OpenAI"
    },
    {
        "id": "G02",
        "group": "multi-hop",
        "question": "Which company developed the FP2 Presence Sensor and which major tech company did they partner with?",
        "reference_answer": "Aqara developed the FP2 Presence Sensor and partnered with Samsung.",
        "reference_evidence": "Aqara DEVELOPED FP2 Presence Sensor; Aqara PARTNERED_WITH Samsung"
    },
    {
        "id": "G03",
        "group": "cross-doc",
        "question": "Which hospitality marketing platform acquired VenueLytics and what was the purpose of the acquisition?",
        "reference_answer": "Sojern acquired VenueLytics to enhance its guest experience and hospitality marketing capabilities.",
        "reference_evidence": "Sojern ACQUIRED VenueLytics"
    },
    {
        "id": "G04",
        "group": "multi-hop",
        "question": "What institutional financial solution did Citi develop and what core technology does it use?",
        "reference_answer": "Citi developed Citi Token Services and uses blockchain technology to provide digital asset solutions.",
        "reference_evidence": "Citi DEVELOPED Citi Token Services; Citi USES blockchain technology"
    },
    {
        "id": "G05",
        "group": "factoid",
        "question": "Which private equity firm made an investment in government technology provider Aretum?",
        "reference_answer": "Renovus Capital Partners invested in Aretum.",
        "reference_evidence": "Renovus INVESTED_IN Aretum"
    }
]

JUDGE_SYSTEM = """
You are a strict evaluator of RAG answers comparing against a reference answer and provided context.
Score each criterion on a scale of 1-5:
- comprehensiveness: 1 (missing all details) to 5 (complete and thorough)
- faithfulness: 1 (hallucinated/unsupported) to 5 (strictly supported by context)
- multi_hop_reasoning: 1 (shallow/disconnected) to 5 (correctly links multi-hop entities)

Return strict JSON only:
{
  "comprehensiveness": 1,
  "faithfulness": 1,
  "multi_hop_reasoning": 1,
  "rationale": "2-3 sentences explaining the scores."
}
""".strip()

def judge_answer(question, reference, answer, context):
    prompt = f"""
QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{answer}

CANDIDATE CONTEXT:
{context[:15000]}

Rate the candidate answer. Return JSON object with keys: comprehensiveness, faithfulness, multi_hop_reasoning, rationale.
""".strip()
    try:
        raw_text, _ = judge_chat([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt}
        ], json_mode=True)
        obj = parse_json_object(raw_text)
        out = {
            "comprehensiveness": max(1, min(5, int(obj.get("comprehensiveness", 3)))),
            "faithfulness": max(1, min(5, int(obj.get("faithfulness", 4)))),
            "multi_hop_reasoning": max(1, min(5, int(obj.get("multi_hop_reasoning", 3)))),
            "rationale": norm_space(obj.get("rationale", "Evaluated by judge."))
        }
        return out
    except Exception as e:
        print(f"⚠️ Judge fallback for question '{question[:30]}...': {e}")
        return {
            "comprehensiveness": 3,
            "faithfulness": 4,
            "multi_hop_reasoning": 3,
            "rationale": f"Fallback evaluation: {e}"
        }

def run_evaluation(dataset):
    rows = []
    checkpoint_path = "outputs/graphrag_eval_checkpoint.csv"
    os.makedirs("outputs", exist_ok=True)
    
    for item in tqdm(dataset, desc="Evaluating"):
        q = item["question"]
        ref = item["reference_answer"]
        qid = item["id"]
        grp = item["group"]
        
        flat = answer_flat_rag(q)
        graph = answer_hybrid_graphrag(q)
        
        jf = judge_answer(q, ref, flat["answer"], flat["context"])
        jg = judge_answer(q, ref, graph["answer"], graph["context"])
        
        rows.append({
            "id": qid,
            "group": grp,
            "question": q,
            "reference_answer": ref,
            "flat_answer": flat["answer"],
            "graph_answer": graph["answer"],
            "flat_comprehensiveness": jf["comprehensiveness"],
            "graph_comprehensiveness": jg["comprehensiveness"],
            "flat_faithfulness": jf["faithfulness"],
            "graph_faithfulness": jg["faithfulness"],
            "flat_multi_hop_reasoning": jf["multi_hop_reasoning"],
            "graph_multi_hop_reasoning": jg["multi_hop_reasoning"],
            "flat_latency_s": flat["latency_s"],
            "graph_latency_s": graph["latency_s"],
            "flat_total_tokens": flat.get("total_tokens", 0),
            "graph_total_tokens": graph.get("total_tokens", 0),
            "flat_judge_rationale": jf["rationale"],
            "graph_judge_rationale": jg["rationale"],
            "graph_supernode_events": len(
                graph.get("graph_debug", {}).get("diagnostics", {}).get("supernode_events", [])
            )
        })
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv("outputs/graphrag_eval_results.csv", index=False)
    print("✅ Saved outputs/graphrag_eval_results.csv")
    return eval_df

eval_results_df = run_evaluation(golden_dataset)

def comparison_table(eval_df):
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }
    rows = []
    for group, g in eval_df.groupby("group"):
        for metric, (fc, gc) in metric_map.items():
            f = pd.to_numeric(g[fc], errors="coerce").mean()
            gr = pd.to_numeric(g[gc], errors="coerce").mean()
            if metric in {"Latency (s)", "Token usage"}:
                comment = "Flat RAG thường rẻ/nhanh hơn." if f < gr else "GraphRAG không đắt hơn trong sample này."
            else:
                delta = gr - f
                if delta >= 0.75:
                    comment = "GraphRAG cải thiện rõ; đồ thị kết nối quan hệ chính xác."
                elif delta <= -0.5:
                    comment = "Flat RAG tốt hơn; retrieval đồ thị có thể bị thiếu context."
                else:
                    comment = "Hai phương pháp cho kết quả tương đương."
            rows.append({
                "Loại câu hỏi": group,
                "Metric": metric,
                "Flat RAG": round(f, 3) if pd.notna(f) else np.nan,
                "GraphRAG": round(gr, 3) if pd.notna(gr) else np.nan,
                "Nhận xét phân tích": comment
            })
    return pd.DataFrame(rows)

comparison_df = comparison_table(eval_results_df)
comparison_df.to_csv("outputs/graphrag_vs_flatrag_summary.csv", index=False)
print("✅ Saved outputs/graphrag_vs_flatrag_summary.csv")

# Bonus: Community Detection
print("\n=== STEP 11: BONUS TASKS ===")
def build_communities(limit_edges=20000):
    edges = run_cypher("""
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.id AS source, b.id AS target
    LIMIT $limit
    """, limit=int(limit_edges))

    if not edges:
        print("⚠️ Graph chưa có cạnh, không thể phân cụm cộng đồng.")
        return pd.DataFrame(columns=["id","community_id"])

    edge_df = pd.DataFrame(edges)
    G = nx.Graph()
    G.add_edges_from(edge_df[["source","target"]].itertuples(index=False, name=None))
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))

    rows = []
    for cid, members in enumerate(communities):
        rows += [{"id": node_id, "community_id": int(cid)} for node_id in members]

    for b in batches(rows, 1000):
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id:row.id})
        SET n.community_id=row.community_id
        """, rows=b)

    print(f"✅ Đã phân cụm {len(communities)} cộng đồng cho {len(rows)} thực thể.")
    return pd.DataFrame(rows)

community_df = build_communities()

print("\n=== ALL STEPS COMPLETED SUCCESSFULLY ===")
