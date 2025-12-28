"""
Secure Executor - Runs user code in isolated Docker containers
Prevents RCE attacks, resource exhaustion, and network abuse
"""

import docker
import json
import tempfile
import os
from typing import Dict, Any
import hashlib
from datetime import datetime

class SecureExecutor:
    """
    Execute code in isolated Docker containers with strict resource limits
    """
    
    def __init__(self):
        self.client = docker.from_env()
        self.max_memory = "512m"  # 512MB RAM limit
        self.max_cpu_quota = 50000  # 50% CPU
        self.timeout = 60  # 60 seconds max
        self.image = "python:3.11-slim"
        
        # Pre-built algorithms (safe, pre-verified code)
        self.safe_algorithms = self._load_safe_algorithms()
    
    def _load_safe_algorithms(self):
        """Load pre-verified algorithms (no user code execution)"""
        from app.services.algorithm_executor import AlgorithmExecutor
        executor = AlgorithmExecutor()
        return executor.algorithms
    
    def execute_isolated(self, algorithm_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ONLY pre-built algorithms (no custom code)
        This is the secure version - no user code execution
        """
        
        if algorithm_name not in self.safe_algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in verified catalog")
        
        # Execute pre-verified algorithm (no Docker needed - it's safe Python)
        result = self.safe_algorithms[algorithm_name](params)
        
        return result
    
    def execute_user_code_isolated(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute user-provided code in isolated Docker container
        USE WITH EXTREME CAUTION - Only for paid enterprise tier
        """
        
        # Create temporary file with user code
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code in safety checks
            safe_wrapper = f"""
import signal
import sys
import os

# Timeout handler
def timeout_handler(signum, frame):
    print("TIMEOUT", file=sys.stderr)
    sys.exit(124)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

# Disable dangerous operations
sys.modules['os'].system = lambda x: None
sys.modules['subprocess'] = None

# User code
try:
{self._indent_code(code, 4)}
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            f.write(safe_wrapper)
            code_file = f.name
        
        try:
            # Run in Docker container with strict limits
            container = self.client.containers.run(
                self.image,
                command=f"python {os.path.basename(code_file)}",
                volumes={os.path.dirname(code_file): {'bind': '/code', 'mode': 'ro'}},
                working_dir='/code',
                mem_limit=self.max_memory,
                memswap_limit=self.max_memory,  # No swap
                cpu_quota=self.max_cpu_quota,
                network_disabled=True,  # No internet access
                read_only=True,  # Filesystem is read-only
                security_opt=['no-new-privileges'],
                cap_drop=['ALL'],  # Drop all capabilities
                detach=False,
                remove=True,
                stdout=True,
                stderr=True,
                timeout=timeout
            )
            
            output = container.decode('utf-8')
            
            return {
                "status": "success",
                "output": output,
                "code_hash": code_hash
            }
        
        except docker.errors.ContainerError as e:
            return {
                "status": "error",
                "error": "Container execution failed",
                "stderr": e.stderr.decode('utf-8') if e.stderr else str(e)
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
        finally:
            # Cleanup temp file
            if os.path.exists(code_file):
                os.remove(code_file)
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))
    
    def verify_resource_limits(self) -> Dict[str, Any]:
        """
        Verify Docker is configured correctly
        Returns system resource availability
        """
        try:
            info = self.client.info()
            
            return {
                "docker_available": True,
                "total_memory_gb": info.get('MemTotal', 0) / (1024**3),
                "cpus": info.get('NCPU', 0),
                "containers_running": info.get('ContainersRunning', 0),
                "isolation": "enabled"
            }
        
        except Exception as e:
            return {
                "docker_available": False,
                "error": str(e),
                "isolation": "disabled"
            }


class FirecrackerExecutor:
    """
    Future: Firecracker micro-VM isolation (even more secure than Docker)
    For now, Docker is sufficient for MVP
    """
    
    def __init__(self):
        # Placeholder for Firecracker implementation
        # Requires more infrastructure setup
        pass
    
    def execute(self, code: str, timeout: int = 60):
        """Execute in Firecracker micro-VM"""
        # TODO: Implement Firecracker integration
        raise NotImplementedError("Firecracker executor not yet implemented")
