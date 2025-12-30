"""
AlgoAPI Algorithm Library
Production-grade algorithm implementations as REST API endpoints
"""

__version__ = "1.0.0"

# Algorithm categories
CATEGORIES = {
    "graph": "Graph algorithms and pathfinding",
    "optimization": "Optimization and constraint solving",
    "sorting": "Sorting algorithms",
    "searching": "Search algorithms",
    "ml": "Machine learning algorithms",
    "strings": "String manipulation and pattern matching",
    "crypto": "Cryptographic algorithms"
}

def get_available_algorithms():
    """Return list of all available algorithms"""
    return {
        "graph": [
            "dijkstra",
            "a_star",
            "bellman_ford",
            "floyd_warshall",
            "prim",
            "kruskal"
        ],
        "optimization": [
            "knapsack_01",
            "knapsack_fractional",
            "knapsack_unbounded",
            "tsp_genetic",
            "bin_packing"
        ],
        "sorting": [
            "quick_sort",
            "merge_sort",
            "heap_sort"
        ],
        "searching": [
            "binary_search",
            "jump_search"
        ],
        "ml": [
            "k_means",
            "linear_regression"
        ],
        "strings": [
            "kmp",
            "rabin_karp"
        ]
    }
