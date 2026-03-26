"""

to run this file:
1. pip install requests spacy sentence-transformers
2. python -m spacy download en_core_web_sm

"""




import requests
import spacy
from sentence_transformers import SentenceTransformer, util
from datetime import datetime


nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

HEADERS = {
    "User-Agent": "kgg_techincal_test (contact: joel.ei.wilson@gmail.com)",
    "Accept": "application/json",
}


SCHEMA_MAP = {  # these can be updated for scale as the network grows - they allow for generalisable intent mapping through embeddings, and include transformation logic
    "PERSON": {
        "age": {"pid": "wdt:P569", "desc": "how old age", "transform": "calc_age"},
    },
    "GPE": {
        "population": {"pid": "wdt:P1082", "desc": "population headcount number of people", "transform": "to_int"},
    },
}  # for a more lightweight alternative to this, a PID could be hardcoded based on the spacy label




def ask(question: str, endpoint: str = "https://query.wikidata.org/sparql"):


    # 1: NER - IDENTIFY SUBJECT AND LABEL
    doc = nlp(question)
    if not doc.ents:
        return None

    subject = doc.ents[0]
    label = "GPE" if subject.label_ == "LOC" else subject.label_ # conflating LOC into GPE for the sake of convenience + safety

    if label not in SCHEMA_MAP:
        return None



    # 2: EMBEDDINGS - MAP QUESTION TO INTENT SCHEMA WITH EMBEDDING SEMANTIC SEARCH
    intents = SCHEMA_MAP[label]
    intent_keys = list(intents.keys())
    intent_descs = [intents[k]["desc"] for k in intent_keys]

    q_vec = embedder.encode(question, convert_to_tensor=True)
    p_vecs = embedder.encode(intent_descs, convert_to_tensor=True)
    hits = util.semantic_search(q_vec, p_vecs, top_k=1)

    if hits[0][0]["score"] < 0.3:
        return None

    selected_intent_key = intent_keys[hits[0][0]["corpus_id"]]
    best_intent = intents[selected_intent_key]
    pid = best_intent["pid"]



    # 3: WIKIDATA QUERY - GET CANDIDATE QIDs
    try:
        qid_res = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "language": "en",
                "format": "json",
                "search": subject.text,
                "limit": 5, # excess allows for resolution of ambiguities (such as New York state vs New York city) in next step
            },
            headers=HEADERS,
            timeout=10,
        )
        qid_res.raise_for_status()
        qid_json = qid_res.json()
        candidate_qids = [item["id"] for item in qid_json.get("search", [])]
        if not candidate_qids:
            return None
    except Exception:
        return None



    # 4: SPARQL - CONSTRUCT & EXECUTE QUERY
    val = None
    try:
        if pid == "wdt:P1082": # goes through lists of candidates and removes any that aren't a city
            values_str = " ".join(f"wd:{c}" for c in candidate_qids) # (could be a reusable function for clarity - for formatting lists of QIDs into VALUES statements)
            query = f"""
            SELECT ?val WHERE {{
              VALUES ?qid {{ {values_str} }}
              ?qid wdt:P31/wdt:P279* wd:Q515 .
              ?qid wdt:P1082 ?val .
            }} LIMIT 1
            """
            res = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            bindings = data.get("results", {}).get("bindings", [])
            val = bindings[0]["val"]["value"] if bindings else None

        else: # tries each candidate until one has the target property - this could be made into an external function, but leaving it here for the sake of everything being within the 'ask' function
            for qid in candidate_qids:
                query = f"SELECT ?val WHERE {{ wd:{qid} {pid} ?val . }} LIMIT 1"
                res = requests.get(
                    endpoint,
                    params={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=10,
                )
                res.raise_for_status()
                data = res.json()
                bindings = data.get("results", {}).get("bindings", [])
                if bindings:
                    val = bindings[0]["val"]["value"]
                    break
    except Exception:
        return None

    if val is None:
        return None



    # 5: TRANSFORM RESULT
    if best_intent["transform"] == "calc_age":
        try:
            date_part = val.lstrip("+").split("T")[0]
            birth_dt = datetime.strptime(date_part, "%Y-%m-%d")
            now = datetime.now()
            age = now.year - birth_dt.year - ((now.month, now.day) < (birth_dt.month, birth_dt.day))
            return str(age)
        except Exception:
            return val

    if best_intent["transform"] == "to_int":
        try:
            return str(int(float(val)))
        except Exception:
            return str(val)

    return str(val)





if __name__ == "__main__":
    assert "63" == ask("how old is Tom Cruise")
    assert "67" == ask("what age is Madonna?")
    assert "8799728" == ask("what is the population of London")
    assert "8804190" == ask("what is the population of New York?")
    print("All assertions passed")