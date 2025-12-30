"""
Algorithm API Routes
RESTful endpoints for all algorithm categories
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional
import hashlib
import json

# Algorithm implementations
from app.algorithms.graphs.dijkstra import execute_dijkstra, DijkstraInput
from app.algorithms.graphs.a_star import execute_a_star, AStarInput
from app.algorithms.optimization.knapsack import execute_knapsack, KnapsackInput

# Database
from app.models_algorithms import AlgorithmUsage, AlgorithmQuota

# You'll need to import these from your existing codebase
# from app.database import get_db
# from app.auth import get_current_user, get_optional_user


router = APIRouter(prefix="/api/v1/algorithms", tags=["algorithms"])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_input_hash(input_data: dict) -> str:
    """Create hash of input for duplicate detection"""
    input_str = json.dumps(input_data, sort_keys=True)
    return hashlib.md5(input_str.encode()).hexdigest()


def check_quota(db: Session, user_id: Optional[str], category: str) -> bool:
    """Check if user has remaining quota"""
    if not user_id:
        return True  # Allow anonymous usage (can change this)
    
    quota = db.query(AlgorithmQuota).filter(
        AlgorithmQuota.user_id == user_id
    ).first()
    
    if not quota:
        # Create default quota for new user
        quota = AlgorithmQuota(user_id=user_id)
        db.add(quota)
        db.commit()
    
    return quota.has_quota(category)


def track_usage(
    db: Session,
    user_id: Optional[str],
    category: str,
    algorithm: str,
    execution_time_ms: float,
    input_size: int,
    input_hash: str,
    success: bool,
    error_message: Optional[str] = None,
    output_summary: Optional[dict] = None,
    request: Optional[Request] = None,
    variant: Optional[str] = None
):
    """Track algorithm usage in database"""
    
    usage = AlgorithmUsage(
        user_id=user_id,
        category=category,
        algorithm=algorithm,
        variant=variant,
        execution_time_ms=execution_time_ms,
        input_size=input_size,
        input_hash=input_hash,
        success=success,
        error_message=error_message,
        output_summary=output_summary,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        endpoint=request.url.path if request else None
    )
    
    db.add(usage)
    
    # Increment quota if successful
    if success and user_id:
        quota = db.query(AlgorithmQuota).filter(
            AlgorithmQuota.user_id == user_id
        ).first()
        if quota:
            quota.increment_usage(category)
    
    db.commit()


# ============================================================================
# GRAPH ALGORITHMS
# ============================================================================

@router.post("/graph/dijkstra")
async def dijkstra_endpoint(
    input_data: DijkstraInput,
    request: Request,
    # db: Session = Depends(get_db),
    # user = Depends(get_optional_user)
):
    """
    **Dijkstra's Shortest Path Algorithm**
    
    Find the shortest path between two nodes in a weighted graph.
    
    **Time Complexity:** O((V + E) log V) with min-heap  
    **Space Complexity:** O(V)
    
    **Use Cases:**
    - GPS navigation and route planning
    - Network packet routing  
    - Social network connections
    - Logistics optimization
    
    **Example:**
    ```python
    {
        "graph": {
            "A": {"B": 5, "C": 3},
            "B": {"D": 2},
            "C": {"B": 1, "D": 4},
            "D": {}
        },
        "start": "A",
        "end": "D"
    }
    ```
    
    **Returns:** Shortest path, distance, and performance metrics
    """
    
    # For demo: use mock db and user
    db = None
    user_id = None  # Replace with: user.id if user else None
    
    # Check quota
    # if not check_quota(db, user_id, "graph"):
    #     raise HTTPException(status_code=429, detail="Quota exceeded for graph algorithms")
    
    try:
        # Execute algorithm
        result = execute_dijkstra(input_data)
        
        # Track usage
        # if db:
        #     track_usage(
        #         db=db,
        #         user_id=user_id,
        #         category="graph",
        #         algorithm="dijkstra",
        #         execution_time_ms=result.execution_time_ms,
        #         input_size=len(input_data.graph),
        #         input_hash=create_input_hash(input_data.dict()),
        #         success=result.success,
        #         output_summary={
        #             "path_length": len(result.shortest_path),
        #             "total_distance": result.total_distance,
        #             "nodes_explored": result.nodes_explored
        #         },
        #         request=request
        #     )
        
        return result
    
    except Exception as e:
        # Track error
        # if db:
        #     track_usage(
        #         db=db,
        #         user_id=user_id,
        #         category="graph",
        #         algorithm="dijkstra",
        #         execution_time_ms=0,
        #         input_size=len(input_data.graph),
        #         input_hash=create_input_hash(input_data.dict()),
        #         success=False,
        #         error_message=str(e),
        #         request=request
        #     )
        
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/graph/a_star")
async def a_star_endpoint(
    input_data: AStarInput,
    request: Request,
    # db: Session = Depends(get_db),
    # user = Depends(get_optional_user)
):
    """
    **A* Pathfinding Algorithm**
    
    Find optimal path in a grid with obstacles using heuristic search.
    
    **Time Complexity:** O(E log V)  
    **Space Complexity:** O(V)
    
    **Use Cases:**
    - Game development (NPC pathfinding)
    - Robotics navigation
    - Logistics route planning
    - Map applications
    
    **Heuristics:**
    - `manhattan`: Best for 4-way movement grids
    - `euclidean`: Straight-line distance
    - `diagonal`: Best for 8-way movement
    - `chebyshev`: Maximum coordinate difference
    
    **Example:**
    ```python
    {
        "grid": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        "start": [0, 0],
        "goal": [2, 2],
        "heuristic": "manhattan",
        "allow_diagonal": false
    }
    ```
    
    **Returns:** Path coordinates, cost, and nodes explored
    """
    
    db = None
    user_id = None
    
    try:
        result = execute_a_star(input_data)
        
        # Track usage (uncomment when db available)
        # if db:
        #     grid_size = len(input_data.grid) * len(input_data.grid[0])
        #     track_usage(...)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

@router.post("/optimization/knapsack")
async def knapsack_endpoint(
    input_data: KnapsackInput,
    request: Request,
    # db: Session = Depends(get_db),
    # user = Depends(get_optional_user)
):
    """
    **Knapsack Problem Solver**
    
    Solve knapsack optimization problem with multiple variants.
    
    **Variants:**
    - `01`: 0/1 Knapsack - each item once or not at all (DP solution)
    - `fractional`: Fractional Knapsack - items can be divided (Greedy)
    - `unbounded`: Unbounded Knapsack - unlimited copies (DP solution)
    
    **Use Cases:**
    - Resource allocation
    - Portfolio optimization
    - Cargo loading
    - Budget planning
    - Task scheduling
    
    **Example:**
    ```python
    {
        "capacity": 50,
        "items": [
            {"name": "item1", "weight": 10, "value": 60},
            {"name": "item2", "weight": 20, "value": 100},
            {"name": "item3", "weight": 30, "value": 120}
        ],
        "variant": "01"
    }
    ```
    
    **Returns:** Maximum value, selected items, and capacity utilization
    """
    
    db = None
    user_id = None
    
    try:
        result = execute_knapsack(input_data)
        
        # Track usage (uncomment when db available)
        # if db:
        #     track_usage(
        #         db=db,
        #         user_id=user_id,
        #         category="optimization",
        #         algorithm="knapsack",
        #         variant=input_data.variant,
        #         execution_time_ms=result.execution_time_ms,
        #         input_size=len(input_data.items),
        #         input_hash=create_input_hash(input_data.dict()),
        #         success=True,
        #         output_summary={
        #             "max_value": result.max_value,
        #             "items_selected": len(result.selected_items),
        #             "capacity_used": result.capacity_used_percent
        #         },
        #         request=request
        #     )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# METADATA & DISCOVERY ENDPOINTS
# ============================================================================

@router.get("/categories")
async def list_categories():
    """
    List all available algorithm categories and algorithms
    """
    from app.algorithms import get_available_algorithms, CATEGORIES
    
    algorithms = get_available_algorithms()
    
    return {
        "categories": [
            {
                "name": category,
                "description": CATEGORIES[category],
                "algorithms": algos,
                "count": len(algos)
            }
            for category, algos in algorithms.items()
        ],
        "total_algorithms": sum(len(algos) for algos in algorithms.values())
    }


@router.get("/stats")
async def user_stats(
    # db: Session = Depends(get_db),
    # user = Depends(get_current_user)
):
    """
    Get user's algorithm usage statistics
    """
    return {
        "message": "Statistics endpoint - integrate with your database",
        "note": "Uncomment database dependencies to enable"
    }


@router.get("/health")
async def health_check():
    """Algorithm service health check"""
    return {
        "status": "healthy",
        "service": "AlgoAPI Algorithms",
        "version": "1.0.0"
    }
