"""
proposal_features.py
====================
Turns a submitted SETUP project proposal (one Supabase row with jsonb sections)
into the predictor variables used by the completion-risk model.

Formulas follow the Feature Engineering spec:

  Variable                      Source                          Formula
  ----------------------------  ------------------------------  ----------------------------------------
  Debt-service coverage         Proposal - projected cash flow  Cash available for amortization / (Principal + Interest)
  net_profit_margin             Proposal - projected income stmt (Net income / Gross sales) x 100
  Penetration Rate              Proposal - market analysis      (SOM / TAM) x 100
  Years in operation            Proposal - company profile      Proposal year - Year established
  Current capacity utilization  Proposal - capacity & volume    (Current production volume / Max capacity) x 100
  Total Assets                  PIS (fallback: proposal bal. sheet, Year 0)
  Total Employment              Proposal - company profile      Total employees

Characteristic variables (Province, Sector, Ownership, Enterprise size) come
from the PIS; if no PIS record is available they fall back to best-effort
values read from the proposal's company profile.

DESIGN NOTE
-----------
The exact shape of the jsonb columns is not fixed here, so the extractors are
tolerant: they search the JSON tree by normalised labels/keys and understand
list-of-row-dicts tables, header+rows grids, {item: {year: value}} maps and
label/value pairs. Every value is returned with the path it was read from and
a status (ok / fallback / assumed / missing) so the dashboard can show an audit
trail. If a label in your JSON differs, add it to the ALIASES dict below.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FEATURE_YEAR = 0         # Year 0 = the year before funding; used for every year-based variable

MODEL_NUMERIC = [
    "Debt-service coverage", "Penetration Rate", "net_profit_margin",
    "Current capacity utilization rate", "Total Employment", "Total Assets",
    "Years in operation",
]

# Default PIS column names (override via secrets -> pis_columns if yours differ)
DEFAULT_PIS_COLUMNS = {
    "Province": "province",
    "Sector": "sector",
    "type_of_ownership": "ownership_type",
    "size_of_enterprise": "enterprise_size",
    "Total Assets": "total_assets",
}

ALIASES = {
    "name":        ["nameoffirm", "firmname", "companyname", "businessname", "nameofbusiness", "beneficiary", "proponent"],
    "title":       ["projecttitle", "title"],
    "address":     ["address", "businessaddress", "plantlocation", "location"],
    "est_year":    ["yearestablished", "yearestablishment", "establishedyear", "yearofestablishment", "established"],
    "prop_year":   ["proposalyear", "yearofproposal", "fiscalyear"],
    "employees":   ["totalemployees", "totalemployment", "numberofemployees", "totalnumberofemployees"],
    "sector":      ["businessactivity", "businessactivities", "sector", "industry"],
    "ownership":   ["typeoforganization", "typeofownership", "ownershiptype", "organizationtype"],
    "size":        ["enterprisesize", "sizeofenterprise", "sizeofbusiness"],
    "project_cost": ["totalprojectcost", "projectcost"],
    "amount_req":  ["amountrequested", "dostifund", "setupfund"],
    "cash_avail":  ["cashavailableforamortization", "cashavailableforamortisation"],
    "principal":   ["principal", "principalpayment", "principalamortization"],
    "interest":    ["interest", "interestpayment"],
    "net_income":  ["netincome", "netprofit", "netincomeaftertax"],
    "gross_sales": ["grosssales", "sales", "totalsales", "grossrevenue", "revenue"],
    "total_assets": ["totalassets"],
    "tam":         ["totaladdressablemarket", "tam"],
    "som":         ["serviceableobtainablemarket", "som"],
    "cur_volume":  ["currentproductionvolume", "productionvolume", "currentvolume"],
    "max_cap":     ["maximumproductioncapacity", "maxproductioncapacity", "maximumcapacity", "maxcapacity"],
}

PROVINCE_KEYWORDS = {
    "iloilo": "Iloilo", "aklan": "Aklan", "kalibo": "Aklan", "boracay": "Aklan",
    "antique": "Antique", "capiz": "Capiz", "roxas city": "Capiz",
    "guimaras": "Guimaras", "negros": "Negros", "bacolod": "Negros",
}

LABEL_KEYS = {
    "", "particulars", "item", "items", "label", "name", "description", "account", "lineitem",
    "category", "field", "variable", "product", "products", "market", "equipment",
    "kindofwaste", "typeofemployment", "months", "designation",
}
VALUE_KEYS = ("value", "values", "details", "answer", "response", "data", "entry")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def nl(s: Any) -> str:
    """Normalise a row label: drops accounting prefixes like 'Less:' / 'Add:'."""
    t = re.sub(r"^\s*(less|add|plus)\s*[:\-]\s*", "", str(s).lower())
    return norm(t)


def to_num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if v == v else None
    s = str(v).strip()
    if s.lower() in {"", "-", "\u2014", "\u2013", "n/a", "na", "none", "null"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return -abs(x) if neg else x


def as_json(v: Any) -> Any:
    if isinstance(v, str) and v.strip()[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def walk(obj: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def iter_tables(obj: Any, path: str = "") -> Iterator[tuple[str, list[dict]]]:
    """Yield (path, rows) for every table-like structure; row keys are normalised."""
    for p, node in walk(obj, path):
        rows = None
        if isinstance(node, list) and node:
            if all(isinstance(x, dict) for x in node):
                rows = [{norm(k): v for k, v in x.items()} for x in node]
            elif (len(node) >= 2 and all(isinstance(x, list) for x in node)
                  and all(isinstance(h, str) for h in node[0])):
                hdr = [norm(h) for h in node[0]]
                rows = [dict(zip(hdr, r)) for r in node[1:]]
        elif isinstance(node, dict):
            cols = node.get("columns") or node.get("headers")
            data = node.get("rows") or node.get("data")
            if isinstance(cols, list) and isinstance(data, list) and data:
                hdr = [norm(c.get("label") or c.get("name") or c.get("key") or "") if isinstance(c, dict) else norm(c)
                       for c in cols]
                if all(isinstance(r, list) for r in data):
                    rows = [dict(zip(hdr, r)) for r in data]
                elif all(isinstance(r, dict) for r in data):
                    rows = [{norm(k): v for k, v in r.items()} for r in data]
        if rows:
            yield p, rows


def row_label(r: dict) -> str:
    for k in r:
        if k in LABEL_KEYS and isinstance(r[k], str) and r[k].strip():
            return nl(r[k])
    for v in r.values():
        if isinstance(v, str) and v.strip():
            return nl(v)
    return ""


def _is_year_key(k: str, year: int) -> bool:
    return norm(k) in {f"year{year}", f"y{year}", f"yr{year}", str(year)}


def pick_year(val: Any, year: int) -> Optional[float]:
    if isinstance(val, dict):
        for k, v in val.items():
            if _is_year_key(k, year):
                return to_num(v)
    elif isinstance(val, list):
        # Year 0..5 -> idx = year ; Year 1..5 -> idx = year-1
        idx = year if len(val) >= 6 else year - 1
        if 0 <= idx < len(val):
            return to_num(val[idx])
    return None


def find_scalar(obj: Any, aliases: list[str]) -> tuple[Any, str]:
    """First scalar value stored under a matching key or label/value pair."""
    al = {nl(a) for a in aliases}
    for p, node in walk(obj):
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if nl(k) in al:
                if not isinstance(v, (dict, list)) and v not in (None, ""):
                    return v, f"{p}.{k}".lstrip(".")
                if isinstance(v, dict):
                    nv = {norm(x): y for x, y in v.items()}
                    for vk in VALUE_KEYS:
                        if vk in nv and nv[vk] not in (None, ""):
                            return nv[vk], f"{p}.{k}.{vk}".lstrip(".")
        nn = {norm(k): k for k in node}
        for lk in LABEL_KEYS - {""}:
            if lk in nn and isinstance(node[nn[lk]], str) and nl(node[nn[lk]]) in al:
                for vk in VALUE_KEYS:
                    if vk in nn and node[nn[vk]] not in (None, ""):
                        return node[nn[vk]], f"{p}[{node[nn[lk]]}]".lstrip(".")
                others = [v for k, v in node.items() if norm(k) not in LABEL_KEYS and not isinstance(v, (dict, list))]
                if len(others) == 1 and others[0] not in (None, ""):
                    return others[0], f"{p}[{node[nn[lk]]}]".lstrip(".")
    return None, ""


def find_value(obj: Any, aliases: list[str]) -> tuple[Any, str]:
    """Like find_scalar, but also returns dict/list values (checkbox groups)."""
    v, src = find_scalar(obj, aliases)
    if v is not None:
        return v, src
    al = {nl(a) for a in aliases}
    for p, node in walk(obj):
        if isinstance(node, dict):
            for k, val in node.items():
                if nl(k) in al and isinstance(val, (dict, list)) and val:
                    return val, f"{p}.{k}".lstrip(".")
    return None, ""


def find_line_item(obj: Any, aliases: list[str], year: int) -> tuple[Optional[float], str, bool]:
    """Value of a financial-statement line for a given projection year.
    Returns (value, source_path, row_found). row_found is True if the line
    exists even when its cell is blank/dash (e.g. Interest = '-')."""
    al = {nl(a) for a in aliases}
    row_found = False
    for p, rows in iter_tables(obj):
        for r in rows:
            lab = row_label(r)
            if lab in al:
                row_found = True
                for k, v in r.items():
                    if _is_year_key(k, year):
                        x = to_num(v)
                        if x is not None:
                            return x, f"{p}[{lab}].year{year}", True
                for vk in VALUE_KEYS:
                    if vk in r:
                        x = pick_year(r[vk], year)
                        if x is not None:
                            return x, f"{p}[{lab}].{vk}", True
            # transposed layout: one row per year, one column per line item
            if norm(r.get("year", "")) in {str(year), f"year{year}"}:
                for k, v in r.items():
                    if nl(k) in al:
                        row_found = True
                        x = to_num(v)
                        if x is not None:
                            return x, f"{p}[year {year}].{k}", True
    for p, node in walk(obj):
        if isinstance(node, dict):
            for k, v in node.items():
                if nl(k) in al and isinstance(v, (dict, list)):
                    row_found = True
                    x = pick_year(v, year)
                    if x is not None:
                        return x, f"{p}.{k}".lstrip("."), True
    return None, "", row_found


def selected_option(v: Any, allowed: Optional[set] = None) -> Optional[str]:
    """Reads a checkbox-style field ({'Corporation': True}, ['Medium'], 'Medium').
    If `allowed` (normalised names) is given, only matching options are returned,
    so one mixed group (e.g. ownership + size) can feed several variables."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().replace("_", " ")
        if not s or (allowed and norm(s) not in allowed):
            return None
        return s.title() if s.islower() or s.isupper() else s
    if isinstance(v, list):
        for x in v:
            r = selected_option(x, allowed)
            if r:
                return r
        return None
    if isinstance(v, dict):
        for k in ("selected", "value", "choice"):
            if k in v:
                return selected_option(v[k], allowed)
        for k, x in v.items():
            if x is True or str(x).strip().lower() in {"x", "true", "yes", "checked", "1"}:
                r = selected_option(str(k), allowed)
                if r:
                    return r
    return None


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class FeatureResult:
    variable: str
    value: Any
    status: str      # ok | fallback | assumed | missing
    formula: str
    inputs: str
    source: str
    check: str = ""   # sanity-check warning ("" = passed)


def sanity_check(variable: str, v: Any) -> str:
    """Flags values that are arithmetically possible but almost certainly a wrong pick."""
    if v is None or not isinstance(v, (int, float)):
        return ""
    if variable == "Penetration Rate" and not 0 <= v <= 100:
        return "Outside 0-100%: SOM should not exceed TAM - check the market-analysis row used."
    if variable == "Current capacity utilization rate" and not 0 <= v <= 100:
        return "Outside 0-100%: volume vs. capacity looks swapped or mis-matched."
    if variable == "net_profit_margin" and not -100 <= v <= 100:
        return "Outside -100..100%: net income vs. gross sales looks mis-matched (units/year)."
    if variable == "Debt-service coverage" and (v <= 0 or v > 50):
        return "Non-positive or >50x: confirm cash available and principal/interest are for the same year and period."
    if variable == "Years in operation" and not 0 <= v <= 150:
        return "Implausible age: check the year established / proposal year."
    return ""


def _miss(variable, formula, why):
    return FeatureResult(variable, None, "missing", formula, why, "")


# --------------------------------------------------------------------------
# Individual extractors
# --------------------------------------------------------------------------
def f_dscr(fin: dict) -> FeatureResult:
    """Year-0 cash available for amortization / annual debt service.
    Year 0 is pre-funding, so the proposal shows no principal or interest in that
    year. In that case the denominator is the proposed first-year debt service
    (principal + interest of the first repayment year) and the result is marked
    'assumed' so the audit trail says so."""
    formula = "Cash available for amortization \u00f7 (Principal + Interest)"
    cash, c_src, _ = find_line_item(fin, ALIASES["cash_avail"], FEATURE_YEAR)
    if cash is None:
        return _miss("Debt-service coverage", formula, f"Year {FEATURE_YEAR} 'Cash Available for Amortization' not found.")

    def service_in(y):
        prin, _, _ = find_line_item(fin, ALIASES["principal"], y)
        intr, _, _ = find_line_item(fin, ALIASES["interest"], y)
        return (prin or 0.0), (intr or 0.0)

    prin, intr = service_in(FEATURE_YEAR)
    status, note, ds_year = "ok", "", FEATURE_YEAR
    if prin + intr <= 0:
        for y in range(FEATURE_YEAR + 1, FEATURE_YEAR + 13):
            prin, intr = service_in(y)
            if prin + intr > 0:
                ds_year, status = y, "assumed"
                note = f" (Year {FEATURE_YEAR} has no debt service yet, so the first proposed repayment year, Year {y}, is used)"
                break
        else:
            return _miss("Debt-service coverage", formula, "No principal/interest payment found in any year.")
    return FeatureResult(
        "Debt-service coverage", cash / (prin + intr), status, formula,
        f"Year {FEATURE_YEAR} cash available = {cash:,.0f}; principal = {prin:,.0f}; interest = {intr:,.0f}{note}",
        c_src)


def f_npm(fin: dict) -> FeatureResult:
    formula = "(Net income \u00f7 Gross sales) \u00d7 100"
    ni, ni_src, _ = find_line_item(fin, ALIASES["net_income"], FEATURE_YEAR)
    gs, gs_src, _ = find_line_item(fin, ALIASES["gross_sales"], FEATURE_YEAR)
    if ni is None or not gs:
        return _miss("net_profit_margin", formula, f"Year {FEATURE_YEAR} net income or gross sales not found.")
    return FeatureResult("net_profit_margin", ni / gs * 100, "ok", formula,
                         f"Year {FEATURE_YEAR}: net income = {ni:,.0f}; gross sales = {gs:,.0f}", ni_src)


def f_penetration(mkt: dict) -> FeatureResult:
    formula = "(SOM \u00f7 TAM) \u00d7 100"
    tam_al = {norm(a) for a in ALIASES["tam"]}
    som_al = {norm(a) for a in ALIASES["som"]}
    for p, rows in iter_tables(mkt):
        cols = set().union(*[set(r) for r in rows])
        tc, sc = next(iter(cols & tam_al), None), next(iter(cols & som_al), None)
        if not (tc and sc):
            continue
        total = next((r for r in rows if row_label(r) in {"total", "grandtotal"}), None)
        tam = som = None
        how = ""
        if total is not None:
            tam, som = to_num(total.get(tc)), to_num(total.get(sc))
            how = "Total row"
        if tam is None or som is None:
            body = [r for r in rows if row_label(r) not in {"total", "grandtotal"}]
            tam = sum(to_num(r.get(tc)) or 0 for r in body)
            som = sum(to_num(r.get(sc)) or 0 for r in body)
            how = "sum of market rows"
        if tam:
            return FeatureResult("Penetration Rate", som / tam * 100, "ok", formula,
                                 f"SOM = {som:,.0f}; TAM = {tam:,.0f} ({how})", p)
    return _miss("Penetration Rate", formula, "Market-analysis table with TAM and SOM columns not found.")


def f_capacity(ctx: dict) -> FeatureResult:
    formula = "(Current production volume \u00f7 Maximum production capacity) \u00d7 100"
    vol, v_src = find_scalar(ctx, ALIASES["cur_volume"])
    cap, c_src = find_scalar(ctx, ALIASES["max_cap"])
    if to_num(vol) is not None and to_num(cap):
        return FeatureResult("Current capacity utilization rate", to_num(vol) / to_num(cap) * 100, "ok", formula,
                             f"Volume = {to_num(vol):,.0f}; capacity = {to_num(cap):,.0f}", c_src)

    cap_tabs, vol_tabs = [], []     # (path, {product: value}, explicit_role)
    for p, rows in iter_tables(ctx):
        cols = set().union(*[set(r) for r in rows])
        cap_c = next((c for c in cols if "capacity" in c), None)
        vol_c = next((c for c in cols if "volume" in c), None)
        if not (cap_c or vol_c):
            continue
        if cap_c and vol_c:      # one table holding both
            cap_tabs.append((p, {row_label(r): to_num(r.get(cap_c)) for r in rows}, "cap"))
            vol_tabs.append((p, {row_label(r): to_num(r.get(vol_c)) for r in rows}, "vol"))
            break
        col = vol_c or cap_c
        data = {row_label(r): to_num(r.get(col)) for r in rows if row_label(r) not in {"total", "grandtotal"}}
        data = {k: v for k, v in data.items() if k and v is not None}
        if len(data) < 2:
            continue
        role = "vol" if (vol_c or "volume" in norm(p) or "current" in norm(p)) else "cap"
        (vol_tabs if role == "vol" else cap_tabs).append((p, data, "explicit" if (vol_c or role == "vol") else "none"))

    assumed = False
    if not vol_tabs and len(cap_tabs) >= 2:
        # Both tables carry the 'capacity' header (the SJL proposal mislabels the volume table).
        # Assume the first is capacity and the second is current volume, if products match.
        if set(cap_tabs[0][1]) & set(cap_tabs[1][1]):
            vol_tabs, cap_tabs, assumed = [cap_tabs[1]], [cap_tabs[0]], True
    if not cap_tabs or not vol_tabs:
        return _miss("Current capacity utilization rate", formula,
                     "Could not find both a capacity table and a current production volume table.")
    cp, cd, _ = cap_tabs[0]
    vp, vd, _ = vol_tabs[0]
    common = set(cd) & set(vd)
    if common:
        c_sum, v_sum = sum(cd[k] for k in common), sum(vd[k] for k in common)
    else:
        c_sum, v_sum = sum(cd.values()), sum(vd.values())
    if not c_sum:
        return _miss("Current capacity utilization rate", formula, "Maximum capacity is zero/blank.")
    note = " (second table assumed to be current volume)" if assumed else ""
    return FeatureResult("Current capacity utilization rate", v_sum / c_sum * 100,
                         "assumed" if assumed else "ok", formula,
                         f"Volume = {v_sum:,.0f}; capacity = {c_sum:,.0f} across {len(common) or len(cd)} product lines{note}",
                         vp)


_PROGRAM_YEAR = re.compile(r"(?:i[\s\-]?fund|setup)\D{0,15}((?:19|20)\d{2})", re.I)


def resolve_proposal_year(ctx: dict, created_at: Any) -> tuple[int, str, str]:
    """Proposal year, in order of trust: explicit field -> 'SETUP iFund 2024' style text -> record created_at."""
    raw, src = find_scalar(ctx, ALIASES["prop_year"])
    m = re.search(r"(19|20)\d{2}", str(raw or ""))
    if m:
        return int(m.group(0)), f"explicit proposal year at {src}", "ok"
    for _, node in walk(ctx):
        if isinstance(node, str):
            mm = _PROGRAM_YEAR.search(node)
            if mm:
                return int(mm.group(1)), f"year read from text '{node.strip()[:40]}'", "ok"
    try:
        return int(pd.to_datetime(created_at).year), "record created_at (no proposal year found in the JSON)", "assumed"
    except Exception:
        return datetime.now().year, "current year (no proposal year or created_at found)", "assumed"


def f_years(ctx: dict, created_at: Any) -> FeatureResult:
    formula = "Proposal year \u2212 Year established"
    est_raw, est_src = find_scalar(ctx, ALIASES["est_year"])
    m = re.search(r"(19|20)\d{2}", str(est_raw or ""))
    if not m:
        return _miss("Years in operation", formula, "Year established not found.")
    est = int(m.group(0))
    py, how, status = resolve_proposal_year(ctx, created_at)
    return FeatureResult("Years in operation", float(py - est), status, formula,
                         f"{py} \u2212 {est} (proposal year: {how})", est_src)


def f_employment(bg: dict) -> FeatureResult:
    formula = "Total employees (company profile)"
    v, src = find_scalar(bg, ALIASES["employees"])
    if to_num(v) is not None:
        return FeatureResult("Total Employment", to_num(v), "ok", formula, f"Total = {to_num(v):,.0f}", src)
    for p, rows in iter_tables(bg):
        cols = set().union(*[set(r) for r in rows])
        if {"male", "female", "total"} <= cols:
            tot = next((r for r in rows if row_label(r) == "total"), None)
            if tot and to_num(tot.get("total")) is not None:
                return FeatureResult("Total Employment", to_num(tot["total"]), "ok", formula,
                                     f"Total = {to_num(tot['total']):,.0f} (Number of Employee table)", p)
    return _miss("Total Employment", formula, "Number-of-employees table/total not found.")


def f_assets(fin: dict, pis: Optional[dict]) -> FeatureResult:
    formula = "From PIS"
    if pis and to_num(pis.get("Total Assets")) is not None:
        return FeatureResult("Total Assets", to_num(pis["Total Assets"]), "ok", formula, "Read from PIS record", "pis")
    v, src, _ = find_line_item(fin, ALIASES["total_assets"], FEATURE_YEAR)
    if v is not None:
        return FeatureResult("Total Assets", v, "fallback", formula + " (fallback: proposal balance sheet)",
                             f"No PIS record; using Year {FEATURE_YEAR} projected balance-sheet total assets = {v:,.0f}", src)
    return _miss("Total Assets", formula, "No PIS record and no balance-sheet total assets found.")


def _province_from_address(addr: str) -> Optional[str]:
    a = (addr or "").lower()
    for kw, prov in PROVINCE_KEYWORDS.items():
        if kw in a:
            return prov
    return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def engineer_features(row: dict, pis: Optional[dict] = None) -> tuple[dict, list[FeatureResult]]:
    """row: one proposals-table record. pis: optional dict keyed by model column names."""
    sec = {k: as_json(row.get(k)) for k in
           ("project_background", "marketing_aspect", "technological_aspect", "waste_management",
            "financial_aspect", "risk_management", "proposal_json")}
    sec = {k: v for k, v in sec.items() if v}

    def ctx(*order):
        rest = [k for k in sec if k not in order]
        return {k: sec[k] for k in [*order, *rest] if k in sec}

    fin = ctx("financial_aspect", "proposal_json")
    mkt = ctx("marketing_aspect", "proposal_json")
    bg = ctx("project_background", "proposal_json")
    cap = ctx("project_background", "technological_aspect", "proposal_json")

    results = [
        f_dscr(fin), f_penetration(mkt), f_npm(fin), f_capacity(cap),
        f_employment(bg), f_assets(fin, pis), f_years(bg, row.get("created_at")),
    ]

    for r in results:
        r.check = sanity_check(r.variable, r.value)

    # characteristic variables (filters / optional model inputs)
    chars = {}
    OWN = {"singleproprietorship", "partnership", "cooperative", "corporation"}
    SIZE = {"micro", "small", "medium", "large"}
    for col, alias_key, allowed in (("Sector", "sector", None), ("type_of_ownership", "ownership", OWN),
                                    ("size_of_enterprise", "size", SIZE)):
        if pis and pis.get(col):
            chars[col] = (str(pis[col]), "ok", "pis")
        else:
            raw, src = find_value(bg, ALIASES[alias_key])
            val = selected_option(raw, allowed)
            if val is None and col == "size_of_enterprise":   # size ticked inside the organisation-type group
                raw, src = find_value(bg, ALIASES["ownership"])
                val = selected_option(raw, allowed)
            chars[col] = (val or "Unknown", "fallback" if val else "missing", src or "")
    if pis and pis.get("Province"):
        chars["Province"] = (str(pis["Province"]), "ok", "pis")
    else:
        addr, src = find_scalar(bg, ALIASES["address"])
        prov = _province_from_address(str(addr or ""))
        chars["Province"] = (prov or "Unknown", "fallback" if prov else "missing", src or "")
    for col, (val, st_, src) in chars.items():
        results.append(FeatureResult(col, val, st_, "From PIS (fallback: company profile)", f"{val}", src))

    name, _ = find_scalar(bg, ALIASES["name"])
    title, _ = find_scalar(ctx("proposal_json", "project_background"), ALIASES["title"])
    cost, _ = find_scalar(ctx("proposal_json", "financial_aspect"), ALIASES["project_cost"])
    req, _ = find_scalar(ctx("proposal_json", "financial_aspect"), ALIASES["amount_req"])
    year, _, _ = resolve_proposal_year(bg, row.get("created_at"))

    rec = {
        "Project_ID": str(row.get("id")),
        "user_id": row.get("user_id"),
        "Beneficiary_Name": str(name).strip() if name else f"Proposal {str(row.get('id'))[:8]}",
        "Project_Title": title,
        "Year": year,
        "Proposal_Status": row.get("status"),
        "Project_Cost": to_num(cost),
        "Amount_Requested": to_num(req),
        "Province": chars["Province"][0], "Sector": chars["Sector"][0],
        "type_of_ownership": chars["type_of_ownership"][0],
        "size_of_enterprise": chars["size_of_enterprise"][0],
    }
    for r in results:
        if r.variable in MODEL_NUMERIC:
            rec[r.variable] = r.value
    rec["Missing_Inputs"] = ", ".join(r.variable for r in results if r.variable in MODEL_NUMERIC and r.value is None)
    rec["Check_Warnings"] = "; ".join(f"{r.variable}: {r.check}" for r in results if r.check)
    rec["Fallback_Inputs"] = ", ".join(r.variable for r in results if r.status in ("fallback", "assumed"))
    return rec, results


def proposals_to_frame(rows: list[dict], pis_rows: Optional[list[dict]] = None,
                       pis_columns: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features_df, audit_df). audit_df has one row per project x variable."""
    pcols = {**DEFAULT_PIS_COLUMNS, **(pis_columns or {})}
    pis_by_user = {}
    for r in pis_rows or []:
        uid = r.get("user_id")
        if uid:
            pis_by_user[uid] = {model_col: r.get(src) for model_col, src in pcols.items()}

    recs, audit = [], []
    for row in rows:
        rec, results = engineer_features(row, pis_by_user.get(row.get("user_id")))
        recs.append(rec)
        for r in results:
            audit.append({"Project_ID": rec["Project_ID"], "Variable": r.variable, "Value": r.value,
                          "Status": r.status, "Check": r.check or "ok", "Formula": r.formula,
                          "Inputs": r.inputs, "Source": r.source})
    df = pd.DataFrame(recs)
    for c in MODEL_NUMERIC:
        if c not in df:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, pd.DataFrame(audit)
