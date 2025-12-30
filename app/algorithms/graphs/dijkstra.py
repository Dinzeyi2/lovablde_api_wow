"""
Dijkstra's Shortest Path Algorithm
Time Complexity: O((V + E) log V) with min-heap
Space Complexity: O(V)

Use cases:
- GPS navigation and route planning
- Network packet routing
- Social network connections
- Logistics optimization
"""

import heapq
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, validator


class DijkstraInput(BaseModel):
    """Input schema for Dijkstra's algorithm"""
    
    graph: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Adjacency list representation. Format: {node: {neighbor: weight}}",
        example={
            "A": {"B": 5, "C": 3},
            "B": {"D": 2},
            "C": {"D": 4},
            "D": {}
        }
    )
    start: str = Field(..., description="Starting node")
    end: str = Field(..., description="Destination node")
    
    @validator('graph')
    def validate_graph(cls, v):
        if not v:
            raise ValueError("Graph cannot be empty")
        
        # Check for negative weights
        for node, neighbors in v.items():
            for neighbor, weight in neighbors.items():
                if weight < 0:
                    raise ValueError(f"Negative weight detected: {node} -> {neighbor} = {weight}. Use Bellman-Ford for negative weights.")
        
        return v
    
    @validator('start', 'end')
    def validate_nodes(cls, v, values):
        if 'graph' in values and v not in values['graph']:
            raise ValueError(f"Node '{v}' not found in graph")
        return v


class DijkstraOutput(BaseModel):
    """Output schema for Dijkstra's algorithm"""
    
    shortest_path: List[str] = Field(..., description="Sequence of nodes from start to end")
    total_distance: float = Field(..., description="Total distance/cost of the path")
    path_weights: List[float] = Field(..., description="Individual edge weights along the path")
    nodes_explored: int = Field(..., description="Number of nodes explored during search")
    execution_time_ms: float = Field(..., description="Algorithm execution time in milliseconds")
    success: bool = Field(True, description="Whether a path was found")


def dijkstra_algorithm(
    graph: Dict[str, Dict[str, float]],
    start: str,
    end: str
) -> Tuple[List[str], float, int]:
    """
    Find shortest path using Dijkstra's algorithm
    
    Args:
        graph: Adjacency list with weights
        start: Starting node
        end: Destination node
        
    Returns:
        Tuple of (path, distance, nodes_explored)
    """
    # Initialize distances and tracking
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    # Min-heap: (distance, node)
    heap = [(0, start)]
    visited = set()
    nodes_explored = 0
    
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        nodes_explored += 1
        
        # Found destination
        if current_node == end:
            break
        
        # Explore neighbors
        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor in visited:
                continue
            
            new_distance = current_distance + weight
            
            # Found shorter path
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(heap, (new_distance, neighbor))
    
    # Reconstruct path
    if distances[end] == float('inf'):
        return [], float('inf'), nodes_explored
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return path, distances[end], nodes_explored


def execute_dijkstra(input_data: DijkstraInput) -> DijkstraOutput:
    """
    Execute Dijkstra's algorithm with validation and performance tracking
    
    Args:
        input_data: Validated input schema
        
    Returns:
        Output schema with results and metrics
    """
    import time
    
    start_time = time.perf_counter()
    
    # Run algorithm
    path, distance, nodes_explored = dijkstra_algorithm(
        input_data.graph,
        input_data.start,
        input_data.end
    )
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    # Calculate individual edge weights
    path_weights = []
    if len(path) > 1:
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            weight = input_data.graph[current][next_node]
            path_weights.append(weight)
    
    success = distance != float('inf')
    
    return DijkstraOutput(
        shortest_path=path,
        total_distance=distance if success else 0,
        path_weights=path_weights,
        nodes_explored=nodes_explored,
        execution_time_ms=round(execution_time, 3),
        success=success
    )


# Example usage for testing
if __name__ == "__main__":
    # Test case 1: Simple path
    test_input = DijkstraInput(
        graph={
            "A": {"B": 5, "C": 3},
            "B": {"D": 2},
            "C": {"B": 1, "D": 4},
            "D": {}
        },
        start="A",
        end="D"
    )
    
    result = execute_dijkstra(test_input)
    print(f"Path: {result.shortest_path}")
    print(f"Distance: {result.total_distance}")
    print(f"Execution time: {result.execution_time_ms}ms")
