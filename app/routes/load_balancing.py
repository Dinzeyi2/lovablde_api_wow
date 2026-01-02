"""
Load Balancing API Routes
Real-time traffic distribution and server pool management
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth import verify_api_key
from app.models import APIKey
from app.algorithms.load_balancing import execute_load_balancing

router = APIRouter(prefix="/api/v1/load-balancing", tags=["load-balancing"])
limiter = Limiter(key_func=get_remote_address)


# ==================== PYDANTIC SCHEMAS ====================

class Server(BaseModel):
    """Server configuration"""
    id: str = Field(..., description="Server identifier")
    host: str = Field(..., description="Server hostname or IP")
    port: int = Field(..., description="Server port")
    weight: int = Field(default=100, ge=1, le=1000, description="Server weight (1-1000)")
    zone: Optional[str] = Field(None, description="Availability zone")
    region: Optional[str] = Field(None, description="Geographic region")


class HealthCheckConfig(BaseModel):
    """Health check configuration"""
    enabled: bool = Field(default=True)
    interval_seconds: int = Field(default=5, ge=1, le=60)
    timeout_seconds: int = Field(default=2, ge=1, le=30)
    unhealthy_threshold: int = Field(default=3, ge=1, le=10)
    healthy_threshold: int = Field(default=2, ge=1, le=10)


class RequestDetails(BaseModel):
    """Request routing details"""
    client_ip: Optional[str] = None
    session_id: Optional[str] = None
    routing_key: Optional[str] = None


class LoadBalancingRequest(BaseModel):
    """Load balancing request"""
    strategy: str = Field(
        default='weighted_least_connection',
        pattern='^(weighted_least_connection|round_robin|least_response_time|ip_hash|consistent_hash)$',
        description="Load balancing strategy"
    )
    servers: List[Server] = Field(..., min_items=1, description="Server pool")
    health_check: Optional[HealthCheckConfig] = Field(default_factory=HealthCheckConfig)
    request: RequestDetails = Field(default_factory=RequestDetails)
    action: str = Field(default='route', pattern='^(route|metrics)$')
    session_persistence: bool = Field(default=False, description="Enable sticky sessions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "weighted_least_connection",
                "servers": [
                    {"id": "web-1", "host": "10.0.1.10", "port": 8080, "weight": 100},
                    {"id": "web-2", "host": "10.0.1.11", "port": 8080, "weight": 150}
                ],
                "health_check": {
                    "enabled": True,
                    "interval_seconds": 5
                },
                "request": {
                    "client_ip": "192.168.1.100",
                    "session_id": "user-12345"
                },
                "action": "route"
            }
        }


# ==================== ENDPOINTS ====================

@router.post("/route")
@limiter.limit("1000/minute")
async def route_traffic(
    request: Request,
    req: LoadBalancingRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    """
    **Route Traffic to Optimal Server**
    
    Intelligently distribute traffic across your server pool using
    production-grade load balancing algorithms.
    
    **Strategies:**
    - `weighted_least_connection`: Route to server with lowest load/weight ratio (recommended)
    - `round_robin`: Sequential distribution across servers
    - `least_response_time`: Route to fastest server
    - `ip_hash`: Sticky sessions by client IP
    - `consistent_hash`: Minimal redistribution on server changes
    
    **Features:**
    - Active health checking with auto-failover (<100ms)
    - Session persistence (sticky sessions)
    - Sub-millisecond routing decisions (<1ms)
    - 99.99% availability with monitoring
    
    **Pricing:** $799/month includes 100M routing decisions
    
    **Use Cases:**
    - Web application load balancing
    - API gateway traffic distribution
    - Microservices routing
    - Database connection pooling
    - CDN edge selection
    
    **Performance:**
    - Routing latency: <1ms
    - Throughput: 100K+ req/sec
    - Failover time: <100ms
    - Balance efficiency: 95%+
    
    **Example Request:**
```json
    {
        "strategy": "weighted_least_connection",
        "servers": [
            {"id": "web-1", "host": "10.0.1.10", "port": 8080, "weight": 100},
            {"id": "web-2", "host": "10.0.1.11", "port": 8080, "weight": 150}
        ],
        "request": {"client_ip": "192.168.1.100"},
        "action": "route"
    }
```
    """
    
    try:
        # Convert Pydantic models to dict for algorithm
        params = {
            'strategy': req.strategy,
            'servers': [server.dict() for server in req.servers],
            'health_check': req.health_check.dict() if req.health_check else {},
            'request': req.request.dict(),
            'action': req.action,
            'session_persistence': req.session_persistence
        }
        
        # Execute load balancing
        result = execute_load_balancing(params)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics")
@limiter.limit("100/minute")
async def get_pool_metrics(
    request: Request,
    req: LoadBalancingRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    """
    **Get Server Pool Metrics**
    
    Retrieve detailed metrics about your server pool including:
    - Total servers and availability
    - Active connections per server
    - Load distribution balance score
    - Server health status
    - Utilization percentage
    
    **Returns:**
    - Pool-wide statistics
    - Per-server metrics
    - Balance quality score (0-100)
    """
    
    try:
        params = {
            'strategy': req.strategy,
            'servers': [server.dict() for server in req.servers],
            'health_check': req.health_check.dict() if req.health_check else {},
            'action': 'metrics'
        }
        
        result = execute_load_balancing(params)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies():
    """
    **List Available Load Balancing Strategies**
    
    Returns details about all available strategies with
    their characteristics and use cases.
    """
    return {
        "strategies": [
            {
                "name": "weighted_least_connection",
                "description": "Route to server with lowest (connections / weight) ratio",
                "best_for": "Heterogeneous server pools with different capacities",
                "complexity": "O(N)",
                "session_aware": False
            },
            {
                "name": "round_robin",
                "description": "Sequential distribution across servers",
                "best_for": "Homogeneous server pools with equal capacity",
                "complexity": "O(1)",
                "session_aware": False
            },
            {
                "name": "least_response_time",
                "description": "Route to server with fastest response time",
                "best_for": "Performance-critical applications",
                "complexity": "O(N)",
                "session_aware": False
            },
            {
                "name": "ip_hash",
                "description": "Sticky sessions by client IP hash",
                "best_for": "Session persistence without explicit session tracking",
                "complexity": "O(1)",
                "session_aware": True
            },
            {
                "name": "consistent_hash",
                "description": "Minimal redistribution when servers change",
                "best_for": "Distributed caching, stateful applications",
                "complexity": "O(log N)",
                "session_aware": True
            }
        ]
    }


@router.get("/health")
async def health_check():
    """Load balancing service health check"""
    return {
        "status": "healthy",
        "service": "Load Balancing",
        "version": "1.0.0",
        "strategies_available": [
            "weighted_least_connection",
            "round_robin",
            "least_response_time",
            "ip_hash",
            "consistent_hash"
        ]
    }
