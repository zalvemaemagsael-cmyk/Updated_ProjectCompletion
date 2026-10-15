import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proposal_features import *
from fixtures import *

# Year-0 expectations, taken from the SJL proposal (income statement p.35, cash flow p.37, market p.11)
exp = {"Debt-service coverage": 7798007/1220400,
       "net_profit_margin": 5009417/35502000*100,
       "Penetration Rate": 1388/2791*100,
       "Current capacity utilization rate": 100.0,
       "Total Employment": 34, "Total Assets": 22260778, "Years in operation": 5}
fails = 0
def show(ok, msg):
    global fails; fails += (not ok); print("PASS" if ok else "FAIL", msg)

for name, row in [("shape A", shape_a), ("shape B", shape_b)]:
    rec, res = engineer_features(row)
    print("==", name, "| missing:", rec["Missing_Inputs"] or "none", "| fallback/assumed:", rec["Fallback_Inputs"])
    for r in res:
        if r.variable in exp:
            show(r.value is not None and abs(r.value-exp[r.variable]) < 1e-6,
                 f"{r.variable:36s} {r.value!s:20.20} [{r.status}] {r.inputs}")

print("\n== Year-0 behaviour")
rec, res = engineer_features(shape_a)
d = next(x for x in res if x.variable == "Debt-service coverage")
show(d.status == "assumed" and "Year 1" in d.inputs, "DSCR flags that Year 0 has no debt service and uses first repayment year as denominator")
n = next(x for x in res if x.variable == "net_profit_margin")
show("Year 0" in n.inputs, "NPM reads Year 0 columns")

# if Year 0 DID have debt service, it must be used directly (status ok)
w = copy.deepcopy(shape_a)
for row in w["financial_aspect"]["projected_cash_flow"]:
    if row["particulars"].endswith("Principal"): row["Year 0"] = 500000
d2 = next(x for x in engineer_features(w)[1] if x.variable == "Debt-service coverage")
show(d2.status == "ok" and abs(d2.value - 7798007/500000) < 1e-9, f"DSCR uses Year-0 debt service when present -> {d2.value:.3f} [{d2.status}]")

print("\n== Years in operation / proposal year")
r = copy.deepcopy(shape_a); r["created_at"] = "2026-03-01T00:00:00Z"
r["proposal_json"]["footer"] = "SETUP iFund 2024/SJL Corporation Page 1"
y = next(x for x in engineer_features(r)[1] if x.variable == "Years in operation")
show(y.value == 5 and y.status == "ok", f"created_at=2026 but footer says SETUP iFund 2024 -> {y.value} [{y.status}]")
r2 = copy.deepcopy(shape_a); r2["created_at"] = "2026-03-01T00:00:00Z"
y2 = next(x for x in engineer_features(r2)[1] if x.variable == "Years in operation")
show(y2.value == 7 and y2.status == "assumed", f"no year text -> falls back to created_at, flagged -> {y2.value} [{y2.status}]")

print("\n== Sanity checks")
bad = copy.deepcopy(shape_a)
bad["marketing_aspect"]["market_analysis"][-1]["Serviceable Obtainable Market"] = 5000
pen = next(x for x in engineer_features(bad)[1] if x.variable == "Penetration Rate")
show(bool(pen.check), f"SOM>TAM flagged -> {pen.value:.1f}")
show(not engineer_features(shape_a)[0]["Check_Warnings"], "SJL passes all sanity checks")

rec, _ = engineer_features(shape_a, {"Total Assets": "30,000,000", "Province": "Iloilo", "Sector": "Apparel",
                                     "type_of_ownership": "Corporation", "size_of_enterprise": "medium"})
show(rec["Total Assets"] == 30000000 and rec["size_of_enterprise"] == "medium", "PIS values override proposal fallbacks")
print("\nFAILURES:", fails)
