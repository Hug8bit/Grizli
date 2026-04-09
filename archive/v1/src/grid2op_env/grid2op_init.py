"""
Grid2Op environment initialization and configuration.
"""

from typing import Dict, Optional, Any, List

try:
    import grid2op
    from grid2op.Environment import Environment
    from grid2op.Parameters import Parameters
    GRID2OP_AVAILABLE = True
except ImportError:
    GRID2OP_AVAILABLE = False


# Available environments with their specifications
AVAILABLE_ENVIRONMENTS = {
    "l2rpn_case14_sandbox": {
        "name": "L2RPN Case 14 Sandbox",
        "buses": 14,
        "lines": 20,
        "generators": 6,
        "loads": 11,
        "description": "Small test environment for development and debugging",
        "difficulty": "easy"
    },
    "l2rpn_neurips_2020_track1_small": {
        "name": "L2RPN NeurIPS 2020 Track 1 (Small)",
        "buses": 36,
        "lines": 59,
        "generators": 22,
        "loads": 37,
        "description": "Intermediate environment from NeurIPS 2020 competition",
        "difficulty": "medium"
    },
    "l2rpn_wcci_2022": {
        "name": "L2RPN WCCI 2022",
        "buses": 118,
        "lines": 186,
        "generators": 62,
        "loads": 91,
        "description": "Large production-scale environment",
        "difficulty": "hard"
    },
    "rte_case14_realistic": {
        "name": "RTE Case 14 Realistic",
        "buses": 14,
        "lines": 20,
        "generators": 6,
        "loads": 11,
        "description": "Realistic French grid simulation",
        "difficulty": "medium"
    },
}

# Difficulty presets
DIFFICULTY_PRESETS = {
    "0": {  # Easy
        "name": "Easy",
        "NO_OVERFLOW_DISCONNECTION": True,
        "HARD_OVERFLOW_THRESHOLD": 999.0,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 999,
    },
    "1": {  # Normal
        "name": "Normal",
        "NO_OVERFLOW_DISCONNECTION": False,
        "HARD_OVERFLOW_THRESHOLD": 2.0,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 3,
    },
    "2": {  # Hard
        "name": "Hard",
        "NO_OVERFLOW_DISCONNECTION": False,
        "HARD_OVERFLOW_THRESHOLD": 1.5,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 2,
    },
}


def create_environment(
    env_name: str = "l2rpn_case14_sandbox",
    use_lightsim: bool = True,
    difficulty: str = "1",
    **kwargs
) -> "Environment":
    """
    Create a Grid2Op environment with specified configuration.

    Args:
        env_name: Name of Grid2Op environment
        use_lightsim: Use LightSim2Grid backend for speed
        difficulty: Difficulty level ("0", "1", "2")
        **kwargs: Additional arguments for grid2op.make()

    Returns:
        Configured Grid2Op environment
    """
    if not GRID2OP_AVAILABLE:
        raise ImportError("Grid2Op required. Install with: pip install grid2op")

    # Try to use LightSim2Grid backend
    backend = None
    if use_lightsim:
        try:
            from lightsim2grid import LightSimBackend
            backend = LightSimBackend()
        except ImportError:
            print("Warning: LightSim2Grid not available. Using default backend.")

    # Configure parameters based on difficulty
    params = Parameters()
    if difficulty in DIFFICULTY_PRESETS:
        preset = DIFFICULTY_PRESETS[difficulty]
        params.NO_OVERFLOW_DISCONNECTION = preset["NO_OVERFLOW_DISCONNECTION"]
        params.HARD_OVERFLOW_THRESHOLD = preset["HARD_OVERFLOW_THRESHOLD"]
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = preset["NB_TIMESTEP_OVERFLOW_ALLOWED"]

    # Create environment
    env = grid2op.make(
        env_name,
        backend=backend,
        param=params,
        **kwargs
    )

    return env


def get_environment_info(env: "Environment") -> Dict[str, Any]:
    """
    Get detailed information about an environment.

    Args:
        env: Grid2Op environment

    Returns:
        Dictionary with environment information
    """
    info = {
        "name": env.name,
        "n_bus": env.n_sub,
        "n_line": env.n_line,
        "n_gen": env.n_gen,
        "n_load": env.n_load,
        "n_sub": env.n_sub,
        "action_space_size": env.action_space.n,
        "observation_space_size": env.observation_space.n,
    }

    # Thermal limits
    thermal_limits = env.get_thermal_limit()
    info["thermal_limits"] = {
        "min": float(thermal_limits.min()),
        "max": float(thermal_limits.max()),
        "mean": float(thermal_limits.mean()),
    }

    # Generator info
    info["gen_pmax"] = env.gen_pmax.tolist()
    info["gen_pmin"] = env.gen_pmin.tolist()

    # Current parameters
    params = env.parameters
    info["parameters"] = {
        "NO_OVERFLOW_DISCONNECTION": params.NO_OVERFLOW_DISCONNECTION,
        "HARD_OVERFLOW_THRESHOLD": params.HARD_OVERFLOW_THRESHOLD,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": params.NB_TIMESTEP_OVERFLOW_ALLOWED,
    }

    return info


def get_available_environments() -> Dict[str, Dict]:
    """
    Get list of available environments with their specifications.

    Returns:
        Dictionary mapping environment names to their specs
    """
    return AVAILABLE_ENVIRONMENTS.copy()


def get_difficulty_presets() -> Dict[str, Dict]:
    """
    Get available difficulty presets.

    Returns:
        Dictionary of difficulty presets
    """
    return DIFFICULTY_PRESETS.copy()


def list_chronics(env: "Environment") -> List[str]:
    """
    List available chronics (scenarios) for an environment.

    Args:
        env: Grid2Op environment

    Returns:
        List of chronic names
    """
    try:
        return [str(c) for c in env.chronics_handler.subpaths]
    except Exception:
        return []


def set_chronic(env: "Environment", chronic_id: int) -> bool:
    """
    Set the current chronic (scenario).

    Args:
        env: Grid2Op environment
        chronic_id: ID of chronic to use

    Returns:
        True if successful
    """
    try:
        env.set_id(chronic_id)
        return True
    except Exception:
        return False


def get_thermal_limits(env: "Environment") -> Dict[str, Any]:
    """
    Get thermal limits for all lines.

    Args:
        env: Grid2Op environment

    Returns:
        Dictionary with thermal limit information
    """
    limits = env.get_thermal_limit()
    return {
        "values": limits.tolist(),
        "min": float(limits.min()),
        "max": float(limits.max()),
        "mean": float(limits.mean()),
        "sum": float(limits.sum()),
    }
