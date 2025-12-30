"""
A* Pathfinding Algorithm
Time Complexity: O(E log V) where E is edges, V is vertices
Space Complexity: O(V)

Use cases:
- Game development (NPC pathfinding)
- Robotics navigation
- Logistics route planning
- Map applications
"""

import heapq
from typing import List, Tuple, Literal
from pydantic import BaseModel, Field, validator
import math


class AStarInput(BaseModel):
    """Input schema for A* pathfinding"""
    
    grid: List[List[int]] = Field(
        ...,
        description="2D grid where 0=walkable, 1=obstacle",
        example=[[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    )
    start: Tuple[int, int] = Field(..., description="Starting position (row, col)", example=(0, 0))
    goal: Tuple[int, int] = Field(..., description="Goal position (row, col)", example=(2, 2))
    heuristic: Literal["manhattan", "euclidean", "diagonal", "chebyshev"] = Field(
        default="manhattan",
        description="Heuristic function to use"
    )
    allow_diagonal: bool = Field(
        default=False,
        description="Allow diagonal movement (8-way vs 4-way)"
    )
    
    @validator('grid')
    def validate_grid(cls, v):
        if not v or not v[0]:
            raise ValueError("Grid cannot be empty")
        
        width = len(v[0])
        for row in v:
            if len(row) != width:
                raise ValueError("Grid must be rectangular")
        
        return v
    
    @validator('start', 'goal')
    def validate_position(cls, v, values):
        if 'grid' in values:
            grid = values['grid']
            row, col = v
            
            if row < 0 or row >= len(grid):
                raise ValueError(f"Row {row} out of bounds")
            if col < 0 or col >= len(grid[0]):
                raise ValueError(f"Column {col} out of bounds")
            if grid[row][col] == 1:
                raise ValueError(f"Position {v} is an obstacle")
        
        return v


class AStarOutput(BaseModel):
    """Output schema for A* pathfinding"""
    
    path: List[Tuple[int, int]] = Field(..., description="Sequence of coordinates from start to goal")
    cost: float = Field(..., description="Total path cost")
    nodes_explored: int = Field(..., description="Number of nodes explored")
    path_found: bool = Field(..., description="Whether a valid path exists")
    execution_time_ms: float = Field(..., description="Algorithm execution time")


def heuristic_manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Manhattan distance (L1 norm) - good for grid with 4-way movement"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def heuristic_euclidean(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Euclidean distance (L2 norm) - straight line distance"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def heuristic_diagonal(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Diagonal distance (Chebyshev) - good for 8-way movement"""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return max(dx, dy)


def heuristic_chebyshev(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Chebyshev distance - maximum of absolute differences"""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return max(dx, dy)


HEURISTICS = {
    "manhattan": heuristic_manhattan,
    "euclidean": heuristic_euclidean,
    "diagonal": heuristic_diagonal,
    "chebyshev": heuristic_chebyshev
}


def get_neighbors(
    pos: Tuple[int, int],
    grid: List[List[int]],
    allow_diagonal: bool
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get walkable neighbors with movement costs
    
    Returns:
        List of (neighbor_position, cost) tuples
    """
    row, col = pos
    rows, cols = len(grid), len(grid[0])
    neighbors = []
    
    # 4-way movement
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 8-way movement (includes diagonals)
    directions_8 = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
    ]
    
    directions = directions_8 if allow_diagonal else directions_4
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        # Check bounds
        if 0 <= new_row < rows and 0 <= new_col < cols:
            # Check if walkable
            if grid[new_row][new_col] == 0:
                # Diagonal moves cost sqrt(2) ≈ 1.414
                cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                neighbors.append(((new_row, new_col), cost))
    
    return neighbors


def a_star_algorithm(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic_name: str,
    allow_diagonal: bool
) -> Tuple[List[Tuple[int, int]], float, int]:
    """
    A* pathfinding algorithm
    
    Returns:
        Tuple of (path, cost, nodes_explored)
    """
    heuristic = HEURISTICS[heuristic_name]
    
    # Priority queue: (f_score, counter, position)
    # Counter ensures stable ordering for equal f_scores
    counter = 0
    open_set = [(0, counter, start)]
    
    # Track visited nodes
    closed_set = set()
    
    # Cost from start to node
    g_score = {start: 0}
    
    # Previous node in optimal path
    came_from = {}
    
    nodes_explored = 0
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        nodes_explored += 1
        
        # Goal reached
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], nodes_explored
        
        # Explore neighbors
        for neighbor, move_cost in get_neighbors(current, grid, allow_diagonal):
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
    
    # No path found
    return [], float('inf'), nodes_explored


def execute_a_star(input_data: AStarInput) -> AStarOutput:
    """
    Execute A* pathfinding with validation and performance tracking
    """
    import time
    
    start_time = time.perf_counter()
    
    path, cost, nodes_explored = a_star_algorithm(
        input_data.grid,
        input_data.start,
        input_data.goal,
        input_data.heuristic,
        input_data.allow_diagonal
    )
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    return AStarOutput(
        path=path,
        cost=round(cost, 3) if cost != float('inf') else 0,
        nodes_explored=nodes_explored,
        path_found=cost != float('inf'),
        execution_time_ms=round(execution_time, 3)
    )


# Example usage
if __name__ == "__main__":
    # Test case: Navigate around obstacle
    test_input = AStarInput(
        grid=[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        start=(0, 0),
        goal=(4, 4),
        heuristic="manhattan",
        allow_diagonal=False
    )
    
    result = execute_a_star(test_input)
    print(f"Path found: {result.path_found}")
    print(f"Path: {result.path}")
    print(f"Cost: {result.cost}")
    print(f"Nodes explored: {result.nodes_explored}")
