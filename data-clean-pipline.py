#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal Document Anonymizer (precision-clean version)
"""

import argparse, json, logging, os, re, sys
from collections import OrderedDict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("anonymizer")

try:
    import spacy
    NLP = spacy.load("en_legal_ner_trf")
except Exception:
    log.warning("spaCy 模型未加载；实体映射将跳过。")
    NLP = None

PERSON_LABELS    = {"LAWYER", "JUDGE", "PETITIONER", "RESPONDENT", "WITNESS", "OTHER_PERSON"}
PLACE_LABELS     = {"GPE"}
STATUTE_LABELS   = {"STATUTE"}
PROVISION_LABELS = {"PROVISION"}

PW_RE        = re.compile(r"(?<!\()\bP\.W\.\s*(\d+)\b(?!\s*\))", re.IGNORECASE)
CASE_NO_RE   = re.compile(r"(T\.R\.|Case|C\.?C\.?|Complaint Case)\s+No\.?\s*\d+(?:\s*of\s*\d{2,4})?", re.IGNORECASE)
CRL_APPEAL_RE= re.compile(r"Criminal Appeal\s+No\.[^.]+?to the [^.]+", re.IGNORECASE)
BEING_CRL_RE = re.compile(r"\s*being\s+Criminal Appeal\s+No\.\s*\d+(?:\s*of\s*\d{2,4})?", re.IGNORECASE)
PREFERRED_APPEAL_RE = re.compile(r"\bpreferred\s+appeal\s+No\.?\s*\d+(?:\s*of\s*\d+)?\s+before\s+the\s+[^,.;]*?Tribunal(?:\s*,\s*place\s*L\d+)?", re.IGNORECASE)
PLACE_L_TAIL_RE     = re.compile(r"\s*,\s*place\s*L\d+", re.IGNORECASE)
SECTION_FRAG_RE = re.compile(r"\bSections?\s+[^.;,]*?\b(?:Act|Code|Rules|Ordinance)\b(?:,\s*\d{4})?", re.IGNORECASE)
OF_ACT_CODE_RE  = re.compile(r"\bof\s+the\s+([A-Z][A-Za-z0-9 \-\'&\.]*)\s+(Act|Code|Rules|Ordinance)\b(?:,\s*\d{4})?", re.IGNORECASE)
ACT_CODE_NAME_RE= re.compile(r"\b([A-Z][A-Za-z0-9 \-\'&\.]*)\s+(Act|Code|Rules|Ordinance)\b(?:,\s*\d{4})?", re.IGNORECASE)
UNDER_CORE_RE   = re.compile(r"\bunder\s+Sections?[^.;,]*?\b(?:Act|Code|Rules|Ordinance)\b(?:,\s*\d{4})?", re.IGNORECASE)
ACTION_TAIL_RE  = re.compile(r"^\s*(?:to|for|in\s+order\s+to|by\s+way\s+of|so\s+as\s+to)\b[^.;]*", re.IGNORECASE)
FOR_SHORT_RE    = re.compile(r"\(?\s*for\s+short\s+`?['\"]?the\s+[^)\'\"]+['\"]?\)?", re.IGNORECASE)
MO_RANGE_RE     = re.compile(r"M\.Os?\.\s*[IVX]+\s+to\s+[IVX]+", re.IGNORECASE)
MO_SINGLE_RE    = re.compile(r"M\.O\.\s*[IVX]+", re.IGNORECASE)
EXHIBIT_RE      = re.compile(r"\(Exhibit\s+[IVX]+\)", re.IGNORECASE)
LONG_DISTRICT_RE       = re.compile(r",\s*[A-Za-z]+\s+District(?:,\s*[A-Za-z ]+)?", re.IGNORECASE)
POLICE_STATION_TAIL_RE = re.compile(r"\bPolice Station,[^,\.]+", re.IGNORECASE)
STATE_OF_RE            = re.compile(r"\bin the State of [A-Za-z ]+", re.IGNORECASE)
PHONE_FROM_INDIA_RE = re.compile(r"by\s+way\s+of\s+phone\s+calls\s+from\s+India", re.IGNORECASE)
SP_INSTRUCTION_RE   = re.compile(r"on\s+the\s+instructions\s+of\s+the\s+Superintendent\s+of\s+Police,\s*[A-Za-z ]+", re.IGNORECASE)
TITLE_COMMA_PLACE_RE = re.compile(r"\b((?:Chief|Additional|Special|Assistant|Presidency|Metropolitan)\s+[A-Z][A-Za-z ]*(?:Judge|Magistrate|Court|Officer))\s*,\s*[A-Z][a-z]+", re.IGNORECASE)
TRAILING_NUM_AFTER_TITLE_RE = re.compile(r"\b((?:Addl\.|Additional|Chief|Assistant|Special)?\s*[A-Z][A-Za-z ]*(?:Judge|Magistrate|Registrar|Tribunal|Court))\s*\d\b", re.IGNORECASE)
ROLE_P_TAG_RE = re.compile(r"\b(The\s+)?(Petitioner\/husband|Respondent(?:\s+No\.?\s*\d+)?|Accused(?:\s+Nos?\.?\s*[\d\sto]+)?|Appellant|Complainant|Defendant|Witness)\s*,?\s*P\d+\s*,?", re.IGNORECASE)
LONELY_ACT_TAIL_RE = re.compile(r"(order\s+dated\s+[0-9A-Za-z,\s]+|vide\s+his\s+order\s+dated\s+[0-9A-Za-z,\s]+)\s*,\s*(Act A\d+)\.?", re.IGNORECASE)
HONORIFIC_P_RE = re.compile(r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Shri|Smt\.?|Sri|Prof\.?|Justice|Hon'?ble|Honorable)\s+P(\d+)\b", re.IGNORECASE)
HIGH_COURT_CLEAN_RE = re.compile(r"\bplace L\d+\s+High Court\b", re.IGNORECASE)
TRIBUNAL_WITH_DET_RE  = re.compile(r"\bthe\s+(?:[A-Z][\w&\.\-]+\s+){1,6}Tribunal\b(?:\s*,\s*place\s*L\d+)?", re.IGNORECASE)
TRIBUNAL_AFTER_PREP_RE= re.compile(r"\b(?P<prep>before|to|in|at)\s+(?:the\s+)?(?P<full>(?:[A-Z][\w&\.\-]+\s+){1,6}Tribunal\b(?:\s*,\s*place\s*L\d+)?)", re.IGNORECASE)
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:])")
DOUBLE_PUNCT_RE       = re.compile(r"([,.;:])\s*([,.;:])")
MULTI_SPACE_RE        = re.compile(r"\s+")
CO_ACCUSED_RE         = re.compile(r"co-\s+accused", re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r'(?<=[.;])\s+(?=[A-Z])')
PUNISH_TRIGGERS = {"guilty", "convicted", "framed", "sentenced", "punished", "charged"}
ROLE_WHITELIST  = {"petitioner","respondent","accused","appellant","complainant","defendant","prosecution","witness"}
PLACE_WHITELIST = {"india"}
ORG_HINTS       = ("society","bank","company","co-operative","cooperative","committee","club","association","corporation","ltd","pvt","university","college","institute","trust","board","department","tribunal","magistrate","court","ministry","commission")
TITLE_HINTS     = {"magistrate","presidency","metropolitan","judge","registrar","tribunal","court","minister","officer"}
ORG_ACRONYM_MAP = {"Manoj Co-operative Housing Society": "MCH Society"}
@dataclass
class AnonymizeResult:
    _id: str
    text: str
    original_text: str
class LegalAnonymizer:
    def __init__(self, nlp=None):
        self.nlp = nlp
        self.person_map = OrderedDict()
        self.place_map  = OrderedDict()
        self.act_map    = OrderedDict()
        self.code_map   = OrderedDict()
        self.p_cnt = self.l_cnt = self.a_cnt = self.c_cnt = 1
    def anonymize(self, text: str, doc_id: str) -> AnonymizeResult:
        self._reset()
        out = self._pipeline(text)
        return AnonymizeResult(doc_id, out, text)
    def _reset(self):
        self.person_map.clear(); self.place_map.clear()
        self.act_map.clear();   self.code_map.clear()
        self.p_cnt = self.l_cnt = self.a_cnt = self.c_cnt = 1
    def _pipeline(self, t: str) -> str:
        for full, short in sorted(ORG_ACRONYM_MAP.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(full)}\b", short, t)
        t = FOR_SHORT_RE.sub("", t)
        t = PW_RE.sub("prime witness", t)
        t = self._map_persons(t)
        t = HONORIFIC_P_RE.sub(r"P\1", t)
        t = CASE_NO_RE.sub("", t)
        t = BEING_CRL_RE.sub("", t)
        t = CRL_APPEAL_RE.sub("appealed to the High Court", t)
        t = self._map_places(t)
        t = HIGH_COURT_CLEAN_RE.sub("High Court", t)
        t = PREFERRED_APPEAL_RE.sub("appealed before Tribunal", t)
        t = PLACE_L_TAIL_RE.sub("", t)
        t = self._shrink_tribunal_names(t)
        t = self._map_statutes_and_provisions(t)
        t = self._process_under_clauses(t)
        t = self._map_acts_codes(t)
        t = self._replace_sections(t)
        t = MO_RANGE_RE.sub("", t)
        t = MO_SINGLE_RE.sub("", t)
        t = EXHIBIT_RE.sub("(Exhibit)", t)
        t = re.sub(r"(guilty)(?:\s*(?:under|and|read with)\s*Section[^-\.;,)]+)+", r"\1", t, flags=re.IGNORECASE)
        t = TITLE_COMMA_PLACE_RE.sub(r"\1", t)
        t = TRAILING_NUM_AFTER_TITLE_RE.sub(r"\1", t)
        t = ROLE_P_TAG_RE.sub(lambda m: (m.group(1) or "") + m.group(2), t)
        t = LONELY_ACT_TAIL_RE.sub(lambda m: f"{m.group(1)}, to make inquiries under Section S of {m.group(2)}", t)
        t = t.replace("Station House Officer,", "Station House Officer")
        t = t.replace("Additional Munsif Magistrate,", "Additional Munsif Magistrate")
        t = CO_ACCUSED_RE.sub("co-accused", t)
        t = re.sub(r'\(?\s*(?:hereinafter\s+referred\s+to\s+as\s+[`"\'“”‘’]?[A-Za-z0-9 \-_,;&]+[`"\'“”‘’]?|therein\s+under\s+Section\s+[A-Za-z0-9\(\)\s]+)\s*\)?', '', t, flags=re.IGNORECASE)
        t = re.sub(r',\s*the\s+[A-Za-z ]+?\s+of\s+M/S\.?[^,]+?(?:Ltd\.?|Limited|Pvt\.?|LLP)\s*,', ',', t, flags=re.IGNORECASE)
        t = re.sub(r'\b([A-Za-z ]*Court)\s+of\s+[A-Z][a-z]+(?:\s*(?:and|,)\s*[A-Z][a-z]+)*\b', r'\1', t)
        t = re.sub(r'\badulterated\s+"?Ashoka special supari"?', 'adulterated supari', t, flags=re.IGNORECASE)
        t = re.sub(r'sample\s+does\s+not\s+conform\s+to\s+the\s+Prevention\s+of\s+Food\s+Adulteration\s+Rules?,?\s*\d{4}', 'sample does not conform to standard rules', t, flags=re.IGNORECASE)
        t = re.sub(r'/Act\s*A\d+(?:\s*A\d+)?', '/kg', t, flags=re.IGNORECASE)
        t = re.sub(r',\s*\(Code\s*C\d+\)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*under\s+Section\s+S\b', '', t, flags=re.IGNORECASE)
        t = re.sub(r'(2000\s*mgs/?kg\.)\s*saccharin\s+and\s+that\s+the\s+sample\s+does\s+not\s+conform\s+to\s+standard\s+rules', r'\1 saccharin and that the sample does not conform to standard rules', t, flags=re.IGNORECASE)
        t = self._strip_ltd_orgs(t)
        t = SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
        t = DOUBLE_PUNCT_RE.sub(r"\2", t)
        t = MULTI_SPACE_RE.sub(" ", t).strip()
        return t
    @staticmethod
    def _ent_label(ent):
        return getattr(ent, "label_", getattr(ent, "label", ""))
    def _map_persons(self, t: str) -> str:
        if not self.nlp: return t
        doc = self.nlp(t)
        role_pos = {i for i, tok in enumerate(doc) if tok.text.lower() in ROLE_WHITELIST}
        org_spans = {(e.start_char, e.end_char) for e in doc.ents if self._ent_label(e) == "ORG"}
        def is_org_like(nm: str) -> bool:
            return any(k in nm.lower() for k in ORG_HINTS)
        for ent in doc.ents:
            if self._ent_label(ent) not in PERSON_LABELS: continue
            st, ed = ent.start_char, ent.end_char
            nm = ent.text.strip()
            if nm.lower() == "prime witness": continue
            if re.fullmatch(r"P\d+", nm, re.IGNORECASE): continue
            if any(s <= st < e or s < ed <= e for s, e in org_spans): continue
            if is_org_like(nm): continue
            if any(h in nm.lower() for h in TITLE_HINTS): continue
            if (ent.start - 1) in role_pos: continue
            if any(r in nm.lower() for r in ROLE_WHITELIST): continue
            if nm not in self.person_map:
                self.person_map[nm] = f"P{self.p_cnt}"; self.p_cnt += 1
        for nm, code in sorted(self.person_map.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(nm)}\b", code, t)
        return t
    def _map_places(self, t: str) -> str:
        if self.nlp:
            doc = self.nlp(t)
            for ent in doc.ents:
                if self._ent_label(ent) not in PLACE_LABELS: continue
                g = ent.text.strip()
                if re.fullmatch(r"place L\d+", g, re.IGNORECASE): continue
                if g.lower() in PLACE_WHITELIST: continue
                if g not in self.place_map:
                    self.place_map[g] = f"place L{self.l_cnt}"; self.l_cnt += 1
        for g, cd in sorted(self.place_map.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(g)}\b", cd, t)
        t = LONG_DISTRICT_RE.sub("", t)
        t = POLICE_STATION_TAIL_RE.sub("", t)
        t = STATE_OF_RE.sub("", t)
        t = PHONE_FROM_INDIA_RE.sub("by way of phone calls", t)
        t = SP_INSTRUCTION_RE.sub("on the instructions of the Superintendent of Police", t)
        return t
    def _ensure_act_code_tag(self, name: str, kind: str) -> str:
        key = name.strip()
        if kind == "act":
            if key not in self.act_map:
                self.act_map[key] = f"Act A{self.a_cnt}"; self.a_cnt += 1
            return self.act_map[key]
        else:
            if key not in self.code_map:
                self.code_map[key] = f"Code C{self.c_cnt}"; self.c_cnt += 1
            return self.code_map[key]
    def _map_statutes_and_provisions(self, t: str) -> str:
        if not self.nlp: return t
        doc = self.nlp(t)
        statutes, provisions = set(), set()
        for ent in doc.ents:
            lab = self._ent_label(ent)
            if lab in STATUTE_LABELS:   statutes.add(ent.text.strip())
            elif lab in PROVISION_LABELS: provisions.add(ent.text.strip())
        for nm in sorted(statutes, key=lambda x: -len(x)):
            tag = self._ensure_act_code_tag(nm, "act")
            t = re.sub(rf"\b{re.escape(nm)}\b", tag, t)
        for pv in sorted(provisions, key=lambda x: -len(x)):
            t = re.sub(rf"\b{re.escape(pv)}\b", "Section S", t)
        return t
    def _map_acts_codes(self, t: str) -> str:
        for m in ACT_CODE_NAME_RE.finditer(t):
            nm, kd = m.group(1).strip(), m.group(2).lower()
            self._ensure_act_code_tag(f"{nm} {kd.capitalize()}", kd)
        for nm, cd in sorted(self.act_map.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(nm)}\b(?:,\s*\d{{4}})?", cd, t)
        for nm, cd in sorted(self.code_map.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(nm)}\b(?:,\s*\d{{4}})?", cd, t)
        return t
    def _replace_sections(self, t: str) -> str:
        out, idx = [], 0
        for m in SECTION_FRAG_RE.finditer(t):
            out.append(t[idx:m.start()])
            blk = m.group(0)
            mo = OF_ACT_CODE_RE.search(blk)
            if mo:
                raw = f"{mo.group(1).strip()} {mo.group(2).capitalize()}"
                tag = self.act_map.get(raw) or self.code_map.get(raw)
                out.append(f"Section S of {tag}" if tag else "Section S")
            else:
                out.append("Section S")
            idx = m.end()
        out.append(t[idx:])
        return "".join(out)
    def _process_under_clauses(self, t: str) -> str:
        segs = SENT_SPLIT_RE.split(t)
        def handle(sent: str) -> str:
            m = UNDER_CORE_RE.search(sent)
            if not m:
                return sent
            head, core, tail = sent[:m.start()], sent[m.start():m.end()], sent[m.end():]
            if any(tr in sent.lower() for tr in PUNISH_TRIGGERS):
                return head + tail
            core = self._replace_sections(core)
            ts = tail.lstrip()
            m2 = ACTION_TAIL_RE.match(ts)
            if m2:
                act, rest = ts[m2.start():m2.end()], ts[m2.end():]
                return head + core + " " + act + rest
            return head + core + tail
        return " ".join(handle(s) for s in segs)
    def _shrink_tribunal_names(self, txt: str) -> str:
        def repl_prep(m): return f"{m.group('prep')} Tribunal"
        txt = TRIBUNAL_AFTER_PREP_RE.sub(repl_prep, txt)
        def repl_det(m):
            s = m.group(0)
            return re.sub(r"\bthe\s+.*?(Tribunal)\b", r"the \1", s, flags=re.IGNORECASE)
        return TRIBUNAL_WITH_DET_RE.sub(repl_det, txt)
    def _strip_ltd_orgs(self, text: str) -> str:
        if not self.nlp: return text
        doc = self.nlp(text)
        spans = [(e.start_char, e.end_char) for e in doc.ents if e.label_ == "ORG" and re.search(r'\b(Ltd\.?|Limited|Pvt\.?|LLP)\b', e.text, re.IGNORECASE)]
        for st, ed in sorted(spans, reverse=True):
            text = text[:st] + text[ed:]
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s+,', ',', text)
        return text.strip()
def batch_from_raw(inp: str, outp: str):
    if not os.path.exists(inp):
        log.error("找不到输入文件：%s", inp); sys.exit(1)
    with open(inp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    anon = LegalAnonymizer(NLP)
    res = []
    for i, item in enumerate(data):
        raw = item.get("raw") or item.get("text", "")
        _id  = item.get("_id", f"{i:05d}")
        result = anon.anonymize(raw, _id)
        res.append({"_id": result._id, "text": result.text, "original_text": result.original_text})
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log.info("Done -> %s", outp)
def main():
    ap = argparse.ArgumentParser(description="Legal anonymizer")
    ap.add_argument("-i", "--input", default="raw.json")
    ap.add_argument("-o", "--output", default="anonymized_results.json")
    args = ap.parse_args()
    batch_from_raw(args.input, args.output)
if __name__ == "__main__":
    main()
