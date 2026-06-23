# Lodestone Ablation Study

_Generated: 2026-06-12 17:47 UTC_


## Results Grid (sorted by nDCG@5 desc)

| Config | Chunker | Retriever | Rerank | nDCG@5 (CI) | Recall@5 (CI) | MRR (CI) | p50 lat |
|------------------------------|----------|------------|--------|--------------------------|--------------------------|------------------------|----------|
| `dense_sentwin+rerank` | sentwin | dense | yes | 0.815 [0.772, 0.856] | 0.953 [0.913, 0.987] | 0.768 [0.716, 0.817] | 40.7 ms |
| `hybrid_rrf_sentwin+rerank` | sentwin | hybrid_rrf | yes | 0.812 [0.767, 0.855] | 0.947 [0.907, 0.980] | 0.766 [0.715, 0.816] | 40.7 ms |
| `dense_fixed+rerank` | fixed | dense | yes | 0.810 [0.767, 0.851] | 0.953 [0.920, 0.987] | 0.762 [0.710, 0.809] | 65.0 ms |
| `hybrid_rrf_fixed+rerank` | fixed | hybrid_rrf | yes | 0.810 [0.767, 0.851] | 0.953 [0.920, 0.987] | 0.762 [0.710, 0.809] | 64.2 ms |
| `bm25_fixed+rerank` | fixed | bm25 | yes | 0.798 [0.750, 0.844] | 0.927 [0.880, 0.967] | 0.754 [0.701, 0.805] | 58.6 ms |
| `bm25_sentwin+rerank` | sentwin | bm25 | yes | 0.781 [0.728, 0.830] | 0.893 [0.840, 0.940] | 0.742 [0.687, 0.797] | 36.1 ms |
| `hybrid_rrf_fixed` | fixed | hybrid_rrf | no | 0.710 [0.654, 0.761] | 0.893 [0.840, 0.940] | 0.655 [0.597, 0.710] | 4.4 ms |
| `dense_sentwin` | sentwin | dense | no | 0.702 [0.646, 0.751] | 0.873 [0.820, 0.920] | 0.651 [0.592, 0.704] | 4.3 ms |
| `hybrid_rrf_sentwin` | sentwin | hybrid_rrf | no | 0.698 [0.644, 0.749] | 0.887 [0.833, 0.933] | 0.639 [0.582, 0.697] | 4.5 ms |
| `dense_fixed` | fixed | dense | no | 0.666 [0.609, 0.716] | 0.867 [0.807, 0.920] | 0.613 [0.555, 0.667] | 4.3 ms |
| `bm25_fixed` | fixed | bm25 | no | 0.658 [0.599, 0.719] | 0.827 [0.767, 0.887] | 0.611 [0.550, 0.676] | 0.0 ms |
| `bm25_sentwin` | sentwin | bm25 | no | 0.621 [0.553, 0.688] | 0.747 [0.673, 0.813] | 0.592 [0.528, 0.660] | 0.0 ms |
