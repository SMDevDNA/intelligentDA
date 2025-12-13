# Wind turbine power: turbine similarity graph (NetworkX)

## Ідея
Вузли = турбіни. Ребра = висока кореляція часових рядів виробітку/потужності.
Traversal (BFS/DFS) дозволяє швидко обходити 'кластер' турбін зі схожою поведінкою.

## Параметри
- metric: energy_kwh
- freq: D
- corr_threshold: 0.85

## Базові статистики графа турбін
- **n_nodes**: 9
- **n_edges**: 0
- **avg_degree**: 0.0
- **deg_p50**: 0.0
- **deg_p90**: 0.0
- **deg_max**: 0
- **n_components**: 9
- **avg_clustering_sample**: 0.0

## Traversal
- **source**: Coastal-T1
- **bfs_edges_count**: 0
- **dfs_edges_count**: 0
- **bfs_tree_nodes**: 1
- **dfs_tree_nodes**: 1
- **bfs_layer_sizes**: [1]
- **desc_at_dist2**: 0
- **edge_bfs_first**: []
- **edge_dfs_first**: []

## Візуалізація
- wind_turbine_graph.png
