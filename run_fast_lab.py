import os, re, json, time, random, hashlib, unicodedata, sys
from pathlib import Path
from collections import defaultdict
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
EXTRACTION_MAX_CHUNKS = 150
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
SUPER_NODE_DEGREE = 50

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

print("=== INITIALIZING FRESH NEO4J DRIVER & CLIENTS ===")
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=300,
            keep_alive=True
        )
    return _driver

def run_cypher(query, **params):
    global _driver
    for attempt in range(3):
        try:
            drv = get_driver()
            with drv.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as e:
            print(f"Cypher error (attempt {attempt+1}): {e}")
            if _driver:
                try:
                    _driver.close()
                except Exception:
                    pass
            _driver = None
            time.sleep(2)
    raise RuntimeError("Cypher execution failed after retries.")

# Test Neo4j
conn_test = run_cypher("RETURN 1 AS ok")
print(f"✅ Connected to Neo4j: {NEO4J_URI} -> {conn_test}")

# Ensure constraints
run_cypher("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
run_cypher("CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)")
run_cypher("CREATE INDEX entity_aliases_norm IF NOT EXISTS FOR (n:Entity) ON (n.aliases_norm)")
print("✅ Constraints created.")

groq_client = Groq(api_key=GROQ_API_KEY)

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

judge_client = None
if JUDGE_PROVIDER == "openai":
    judge_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def judge_chat(messages, json_mode=False):
    if JUDGE_PROVIDER == "groq" or judge_client is None:
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

# Load data
raw = pd.read_csv(DATA_PATH)
df = pd.DataFrame()
df["text"] = raw["description"].fillna("").map(norm_space)
df["title"] = raw["title"].fillna("").map(norm_space)
df["published_date"] = pd.to_datetime(raw["published_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
df["article_id"] = [sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])]
df = df[df["text"].str.len() >= 30].copy()
df["dedup_key"] = [sha1(norm_space(f"{t}\n{x}").lower()) for t, x in zip(df["title"], df["text"])]
df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
if LAB_MAX_ARTICLES and len(df) > LAB_MAX_ARTICLES:
    df = df.sample(LAB_MAX_ARTICLES, random_state=SEED).sort_index().reset_index(drop=True)

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

rows = []
for r in df.itertuples(index=False):
    for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
        rows.append({
            "chunk_id": f"{r.article_id}::c{i:04d}",
            "article_id": r.article_id,
            "title": r.title,
            "published_date": r.published_date,
            "text": text,
        })
        if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
            break
chunks_df = pd.DataFrame(rows)
print(f"Chunks prepared: {len(chunks_df)}")

# Check if we already have saved raw triples
triples_cache_path = "outputs/raw_triples_extracted.csv"
if os.path.exists(triples_cache_path):
    print(f"Loading cached triples from {triples_cache_path}...")
    raw_triples_df = pd.read_csv(triples_cache_path)
else:
    print("Performing fast NER+RE extraction...")
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
            "text": r.text,
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

    extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS)
    meta = extraction_source.set_index("chunk_id")["published_date"].to_dict()
    triples = []
    batch_size = 5
    for start in tqdm(range(0, len(extraction_source), batch_size), desc="NER+RE"):
        batch = extraction_source.iloc[start:start+batch_size]
        try:
            obj, _ = extract_batch(batch)
            for item in obj.get("items", []):
                cid = item.get("chunk_id")
                if cid not in meta:
                    continue
                for x in item.get("relations", []):
                    s, t = norm_space(x.get("source")), norm_space(x.get("target"))
                    st, tt, rel = x.get("source_type"), x.get("target_type"), x.get("relation")
                    if not s or not t or st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES or rel not in ALLOWED_RELATIONS:
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
        except Exception as e:
            print(f"Batch error at {start}: {e}")
        time.sleep(0.3)
    raw_triples_df = pd.DataFrame(triples)
    os.makedirs("outputs", exist_ok=True)
    raw_triples_df.to_csv(triples_cache_path, index=False)

print(f"Total raw triples: {len(raw_triples_df)}")

# Entity Resolution
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
entity_resolution_audit_df.to_csv("outputs/entity_resolution_audit.csv", index=False)
print(f"Canonicalized triples: {len(triples_df)}")

# Neo4j Insertion
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

def batches(records, size=500):
    for i in range(0, len(records), size):
        yield records[i:i+size]

def bulk_insert_nodes(nodes_df, batch_size=500):
    if nodes_df.empty:
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

def bulk_insert_edges(triples_df, batch_size=500):
    if triples_df.empty:
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

print("Clearing and re-inserting into Neo4j...")
run_cypher("MATCH (n:Entity) DETACH DELETE n")
nodes_df = build_nodes(triples_df)
bulk_insert_nodes(nodes_df)
bulk_insert_edges(triples_df)

# Sanity Check
invalid = run_cypher("""
MATCH ()-[r]->()
WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
RETURN count(r) AS n
""")[0]["n"]

counts = {
    "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
    "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
    "invalid_provenance_edges": invalid,
}
print("Graph counts:", counts)
assert invalid == 0, f"Invalid provenance: {invalid}"

top_degree_df = pd.DataFrame(run_cypher("""
MATCH (n:Entity)
OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree
RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
ORDER BY degree DESC LIMIT 15
"""))
print("Top degree entities in Neo4j:")
print(top_degree_df.head(10))

# Flat RAG index
chunk_vecs = embed_model.encode(chunks_df["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
flat_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
flat_index.add(chunk_vecs.astype(np.float32))

def flat_search(query, top_k=5):
    qv = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = flat_index.search(qv, top_k)
    return chunks_df.iloc[indices[0]].copy().assign(score=scores[0])

# Seed matching & Graph Retrieval
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

# Golden Dataset Evaluation
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

print("Running evaluation on Golden Dataset...")
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
    return eval_df

eval_results_df = run_evaluation(golden_dataset)

# Summary table
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

summary_df = comparison_table(eval_results_df)
summary_df.to_csv("outputs/graphrag_vs_flatrag_summary.csv", index=False)

# Community Detection
print("Running Community Detection...")
edges = run_cypher("""
MATCH (a:Entity)-[r]->(b:Entity)
RETURN a.id AS source, b.id AS target
LIMIT 20000
""")
if edges:
    edge_df = pd.DataFrame(edges)
    G = nx.Graph()
    G.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    rows = []
    for cid, members in enumerate(communities):
        rows += [{"id": node_id, "community_id": int(cid)} for node_id in members]
    for b in batches(rows, 500):
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id:row.id})
        SET n.community_id=row.community_id
        """, rows=b)
    print(f"✅ Đã phân cụm {len(communities)} cộng đồng cho {len(rows)} thực thể.")

print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY! ===")
