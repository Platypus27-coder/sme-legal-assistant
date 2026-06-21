"""
F2 Score Simulation — Tính toán hướng đi tối ưu.

Dựa trên metric thực tế: P=0.2667, R=0.489, F2=0.3976
Và quy tắc BTC: F2 = 5PR / (4P + R) — Recall nặng gấp 4 lần Precision
"""

def f2_score(precision, recall):
    if precision == 0 or recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)


print("=" * 80)
print("PHAN TICH F2 — TIM HUONG DI TOI UU")
print("=" * 80)

# Baseline hiện tại
P0, R0 = 0.2667, 0.489
F2_0 = f2_score(P0, R0)
print(f"\nHien tai: P={P0:.4f}, R={R0:.4f} => F2={F2_0:.4f}")
print(f"(Leaderboard F2-Macro = 0.3976, xap xi do macro averaging)")

print()
print("-" * 80)
print("HUONG A: Tang max_articles (5-8), ha threshold => Recall tang, Precision giam")
print("-" * 80)

scenarios_a = [
    ("max_articles=5, threshold thap",  0.20, 0.58),
    ("max_articles=5, threshold rat thap", 0.18, 0.62),
    ("max_articles=8, threshold thap",  0.15, 0.65),
    ("max_articles=8, threshold rat thap", 0.12, 0.70),
]
for name, p, r in scenarios_a:
    f2 = f2_score(p, r)
    delta = f2 - F2_0
    arrow = "+" if delta > 0 else ""
    print(f"  {name:42s} | P={p:.2f} R={r:.2f} => F2={f2:.4f} ({arrow}{delta:.4f})")

print()
print("-" * 80)
print("HUONG B: Giu max_articles=3, them BGE-M3 => ca Precision va Recall tang")
print("-" * 80)

scenarios_b = [
    ("BGE-M3 ket qua tot",    0.35, 0.55),
    ("BGE-M3 ket qua rat tot", 0.40, 0.60),
    ("BGE-M3 ket qua trung binh", 0.30, 0.52),
]
for name, p, r in scenarios_b:
    f2 = f2_score(p, r)
    delta = f2 - F2_0
    arrow = "+" if delta > 0 else ""
    print(f"  {name:42s} | P={p:.2f} R={r:.2f} => F2={f2:.4f} ({arrow}{delta:.4f})")

print()
print("-" * 80)
print("HUONG C: Tang max_articles=5 + BGE-M3 (ket hop)")
print("-" * 80)

scenarios_c = [
    ("max5 + BGE-M3 trung binh", 0.25, 0.65),
    ("max5 + BGE-M3 tot",       0.30, 0.70),
    ("max8 + BGE-M3 tot",       0.22, 0.75),
]
for name, p, r in scenarios_c:
    f2 = f2_score(p, r)
    delta = f2 - F2_0
    arrow = "+" if delta > 0 else ""
    print(f"  {name:42s} | P={p:.2f} R={r:.2f} => F2={f2:.4f} ({arrow}{delta:.4f})")

print()
print("=" * 80)
print("DO NHAY F2 VUNG HIEN TAI (P=0.2667, R=0.489)")
print("=" * 80)

# Sensitivity: tang 0.05 Recall vs tang 0.05 Precision
f2_more_recall = f2_score(P0, R0 + 0.05)
f2_more_precision = f2_score(P0 + 0.05, R0)
print(f"  Tang Recall  +0.05: F2 = {f2_more_recall:.4f}  (delta = +{f2_more_recall - F2_0:.4f})")
print(f"  Tang Precision+0.05: F2 = {f2_more_precision:.4f}  (delta = +{f2_more_precision - F2_0:.4f})")
print(f"  => Tang Recall loi gap {(f2_more_recall - F2_0)/(f2_more_precision - F2_0):.2f} lan so voi tang Precision")

print()
print("=" * 80)
print("KET LUAN")
print("=" * 80)
print("""
  1. NHANH NHAT: Tang max_articles tu 3 len 5, ha threshold
     - Chi can retune (co san script), khong can re-run LLM
     - Du kien F2 tang len ~0.44-0.48
     - Rui ro: thap (chi doi config, khong doi code)

  2. MANH NHAT: Ket hop tang max_articles + BGE-M3
     - Can re-index tren Colab (mat vai gio GPU)
     - Du kien F2 tang len ~0.50-0.55
     - Rui ro: trung binh (BGE-M3 co the them nhieu)

  3. QUAN TRONG: Chien luoc hien tai (max_articles=3, threshold cao)
     DI NGUOC lai khuyen nghi cua BTC!
     BTC noi ro: "Threshold nen THAP, MAX_ARTICLES nen CAO"
     => Viec ep max=3 da hi sinh Recall qua nhieu.
""")
