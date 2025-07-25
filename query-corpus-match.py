#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statute and provision citation extraction using spaCy NER and regex fallback.
Performs alias mapping, exact and fuzzy matching, and Act/Code caching.
"""

import re
import json
import spacy
import pandas as pd
from rapidfuzz import process, fuzz

# === Config ===
LAWS_CSV     = "Indian_Laws_Dataset/train.csv"
RAW_JSON     = "raw.json"
OUT_CSV      = "matched_citations.csv"
SCORE_THRESH = 85

# === Model & Alias Mapping ===
NLP = spacy.load("en_legal_ner_trf")
if "sentencizer" not in NLP.pipe_names:
    NLP.add_pipe("sentencizer")

ACT_ALIAS = {
    "ipc": "Indian Penal Code",
    "i.p.c.": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure Act",
    "cr.p.c.": "Code of Criminal Procedure Act",
}
SHORT_MAP = {
    "penal code": "Indian Penal Code",
    "the code":   "Indian Penal Code",
    "criminal procedure": "Code of Criminal Procedure Act",
    "the criminal procedure": "Code of Criminal Procedure Act",
    "societies act": "Maharashtra Co-operative Societies Act",
    "prohibition act": "Dowry Prohibition Act",
    "letters patent": "Letters Patent",
    "the letters patent": "Letters Patent",
    **ACT_ALIAS,
}

# === Regex Patterns ===
_YEAR = re.compile(r"[,–-]?\s*\(?\b\d{4}\)?")
_PUNC = re.compile(r"[\.,\-\(\)\'\"]")
_STOP = re.compile(r"\b(the|of|and|&)\b", flags=re.IGNORECASE)
_CIT_RE = re.compile(
    r"\b(?:Section|Sections|Clause|Clauses|Article|Articles)\s*"
    r"(?P<nums>[\dA-Za-z\-\s,\(\)]+?)\s+"
    r"(?:of\s+the\s+)?(?P<act>[A-Z][A-Za-z0-9&\s\-,]*?(?:Act|Code|Bill|Patent(?:s)?|Rules|Procedure))\b",
    flags=re.IGNORECASE | re.DOTALL
)

# === Normalization & Mapping ===
def normalize(name: str) -> str:
    """Normalize act names for comparison."""
    s = name.lower()
    s = _YEAR.sub("", s)
    s = _PUNC.sub(" ", s)
    s = _STOP.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def resolve_alias(act_raw: str) -> str:
    """Resolve act aliases and short forms."""
    key = re.sub(r"[.\s]", "", act_raw.lower())
    if key in ACT_ALIAS:
        return ACT_ALIAS[key]
    key2 = normalize(act_raw)
    return SHORT_MAP.get(key2, act_raw)

def is_valid_act_name(act_name: str) -> bool:
    """Return True if act_name is a valid act title."""
    if not act_name or len(act_name.strip()) == 0:
        return False
    act_clean = act_name.strip().lower()
    if act_clean.startswith(('of ', 'and ')):
        return False
    invalid_patterns = [
        r'^of\s+the\s*$',
        r'^and\s*$',
        r'^of\s*$',
        r'^the\s*$'
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, act_clean):
            return False
    return True

def extract_provisions_from_text(prov_text: str) -> list:
    """Extract all provision numbers from a text span."""
    nums = re.sub(r"\band\b", ",", prov_text, flags=re.IGNORECASE)
    provisions = []
    for num in re.split(r"[,\s]+", nums):
        if re.search(r"\d", num):
            sec = re.sub(r"[^\dA-Za-z\-\(\)]", "", num)
            if sec:
                provisions.append(sec)
    return provisions

def extract_citations(text: str) -> list:
    """
    Extract (section, act) pairs from text using NER and regex.
    Handles cross-sentence references and deduplication.
    """
    pairs = []
    doc = NLP(text)
    last_act = None
    last_code = None
    for sent in doc.sents:
        sent_txt = sent.text
        sent_low = sent_txt.lower()
        statutes = [ent.text for ent in sent.ents if ent.label_ == "STATUTE"]
        clean_statutes = [re.sub(r"[,–-]?\s*\(?\d{4}\)?", "", s).strip() for s in statutes]
        for act in clean_statutes:
            if re.search(r"\bAct\b", act, re.IGNORECASE):
                last_act = act
            if re.search(r"\bCode\b", act, re.IGNORECASE):
                last_code = act
        provisions = [ent.text for ent in sent.ents if ent.label_ == "PROVISION"]
        sentence_matches = []
        for m in _CIT_RE.finditer(sent_txt):
            nums_text = m.group("nums")
            act_text  = m.group("act").strip()
            if is_valid_act_name(act_text):
                prov_nums = extract_provisions_from_text(nums_text)
                for sec in prov_nums:
                    sentence_matches.append((sec, act_text))
        if sentence_matches:
            pairs.extend(sentence_matches)
        elif statutes and provisions:
            for act in clean_statutes:
                if is_valid_act_name(act):
                    for prov in provisions:
                        prov_nums = extract_provisions_from_text(prov)
                        for sec in prov_nums:
                            pairs.append((sec, act))
        elif provisions:
            target = None
            if " of the code" in sent_low or re.search(r"\bcode\b", sent_low):
                target = last_code
            else:
                target = last_act
            if target and is_valid_act_name(target):
                for prov in provisions:
                    prov_nums = extract_provisions_from_text(prov)
                    for sec in prov_nums:
                        pairs.append((sec, target))
    if not pairs:
        regex_pairs = []
        for m in _CIT_RE.finditer(text):
            nums_text = m.group("nums")
            act_text  = m.group("act").strip()
            if is_valid_act_name(act_text):
                prov_nums = extract_provisions_from_text(nums_text)
                for sec in prov_nums:
                    regex_pairs.append((sec, act_text))
        pairs.extend(regex_pairs)
    seen, unique = set(), []
    for sec, act in pairs:
        if (sec, act) not in seen:
            seen.add((sec, act))
            unique.append((sec, act))
    return unique

# === Main Pipeline ===
def main():
    """Main entry: load data, extract citations, match, and output results."""
    df_laws = pd.read_csv(LAWS_CSV, dtype={"section": str})
    df_laws["act_norm"] = df_laws["act_title"].map(normalize)
    raws = json.load(open(RAW_JSON, encoding="utf-8"))
    records = []
    print("\n=== DEBUG: Results after improved cache logic ===")
    for rec in raws:
        txt   = rec["raw"]
        cites = extract_citations(txt)
        print(f"ID={rec['_id']} -> {cites}")
        for sec, act in cites:
            std_act = resolve_alias(act)
            records.append({
                "_id": rec["_id"],
                "raw_text": txt,
                "section": sec,
                "act_raw": std_act,
                "act_norm": normalize(std_act)
            })
    if not records:
        print("No citations recognized.")
        return
    df_cites = pd.DataFrame(records)
    df = df_cites.merge(
        df_laws[["act_norm","section","act_title","law"]],
        on=["act_norm","section"], how="left"
    )
    def fill_fuzzy(r):
        if pd.isna(r["act_title"]):
            best, score, idx = process.extractOne(
                r["act_norm"], df_laws["act_norm"], scorer=fuzz.token_sort_ratio
            )
            if score >= SCORE_THRESH:
                hit = df_laws.iloc[idx]
                return pd.Series([hit["act_title"], hit["law"]])
        return pd.Series([r["act_title"], r["law"]])
    df[["act_title","law"]] = df.apply(fill_fuzzy, axis=1)
    df = df[df["act_title"].notna()].copy()
    df[["_id","raw_text","section","act_title","law"]]\
      .to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nDone: {len(df)} citations, {df['act_title'].notna().sum()} matched.")

if __name__ == "__main__":
    main()