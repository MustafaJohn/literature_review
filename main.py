from orchestration.graph import build_graph

graph = build_graph()

print("=== Literature Review Agent ===")
print("Enter a research topic OR a paper title/DOI")
print("Citation style: APA or IEEE\n")

while True:
    q = input("Query> ").strip()
    if q.lower() in {"quit", "exit"}:
        break

    style = input("Citation style [APA/IEEE, default APA]> ").strip().upper()
    if style not in {"APA", "IEEE"}:
        style = "APA"

    result = graph.invoke({
        "query":             q,
        "input_type":        "",
        "citation_style":    style,
        "fetched_docs":      [],
        "vector_results":    [],
        "graph_results":     [],
        "clusters":          [],
        "final_context":     "",
        "citation_list":     "",
        "next_step":         "",
        "analysis_decision": "",
        "sources":           [],
        "logs":              [],
    })

    print("\n── CLUSTERS ──")
    for c in result.get("clusters", []):
        print(f"  • {c['theme']}: {c['description']}")

    print("\n── LITERATURE REVIEW ──\n")
    print(result.get("final_context", ""))

    print("\n── REFERENCES ──\n")
    print(result.get("citation_list", ""))