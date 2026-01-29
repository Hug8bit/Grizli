"""
Reduced action space for Grid2Op environments.
Implements Top-K action selection to make RL tractable.
"""

from typing import List, Optional, Dict, Any
import numpy as np

try:
    import grid2op
    from grid2op.Action import BaseAction
    GRID2OP_AVAILABLE = True
except ImportError:
    GRID2OP_AVAILABLE = False


class TopKActionSpace:
    """
    Reduced action space selecting top-K most useful actions.

    Grid2Op's full action space is exponentially large (combinatorial
    explosion of line/substation actions). This class reduces it to
    a manageable discrete space by selecting the most impactful actions.

    Action Categories:
    1. Do Nothing (always included)
    2. Line Disconnections (prioritized by thermal limit)
    3. Line Reconnections
    4. Substation Reconfigurations (bus splits)
    """

    def __init__(
        self,
        env: "grid2op.Environment.Environment",
        k: int = 10,
        include_do_nothing: bool = True,
        include_line_disconnect: bool = True,
        include_line_reconnect: bool = True,
        include_substation: bool = True
    ):
        """
        Initialize Top-K action space.

        Args:
            env: Grid2Op environment
            k: Number of actions to include (excluding do_nothing)
            include_do_nothing: Include do-nothing action
            include_line_disconnect: Include line disconnection actions
            include_line_reconnect: Include line reconnection actions
            include_substation: Include substation reconfiguration
        """
        if not GRID2OP_AVAILABLE:
            raise ImportError("Grid2Op required. Install with: pip install grid2op")

        self.env = env
        self.k = k
        self._actions: List[BaseAction] = []
        self._descriptions: List[str] = []

        self._build_action_space(
            include_do_nothing,
            include_line_disconnect,
            include_line_reconnect,
            include_substation
        )

    def _build_action_space(
        self,
        include_do_nothing: bool,
        include_line_disconnect: bool,
        include_line_reconnect: bool,
        include_substation: bool
    ) -> None:
        """Build the reduced action space."""
        action_helper = self.env.action_space

        # 1. Do Nothing (always first)
        if include_do_nothing:
            self._actions.append(action_helper({}))
            self._descriptions.append("Do Nothing")

        # Get line thermal limits for prioritization
        thermal_limits = self.env.get_thermal_limit()
        n_line = self.env.n_line

        # Sort lines by thermal limit (smallest = most constrained = higher priority)
        line_priority = np.argsort(thermal_limits)

        # 2. Line Disconnections (highest priority lines first)
        if include_line_disconnect:
            for line_id in line_priority:
                if len(self._actions) >= self.k + 1:  # +1 for do_nothing
                    break
                try:
                    action = action_helper({"set_line_status": [(line_id, -1)]})
                    self._actions.append(action)
                    self._descriptions.append(f"Disconnect Line {line_id}")
                except Exception:
                    continue

        # 3. Line Reconnections
        if include_line_reconnect:
            for line_id in line_priority:
                if len(self._actions) >= self.k * 2 + 1:
                    break
                try:
                    action = action_helper({"set_line_status": [(line_id, 1)]})
                    self._actions.append(action)
                    self._descriptions.append(f"Reconnect Line {line_id}")
                except Exception:
                    continue

        # 4. Substation Reconfigurations (bus splits)
        if include_substation:
            n_sub = self.env.n_sub
            for sub_id in range(n_sub):
                if len(self._actions) >= self.k * 3 + 1:
                    break

                # Get elements connected to this substation
                topo_vect = self.env.get_obs().topo_vect
                sub_mask = self.env.action_space.sub_info == sub_id

                # Simple topology change: move first element to bus 2
                elements_at_sub = np.where(sub_mask)[0]
                if len(elements_at_sub) > 1:
                    try:
                        # Create topology vector for this substation
                        new_topo = np.ones(len(elements_at_sub), dtype=int)
                        new_topo[0] = 2  # Move first element to bus 2

                        action = action_helper({
                            "set_bus": {"substations_id": [(sub_id, new_topo)]}
                        })
                        self._actions.append(action)
                        self._descriptions.append(f"Split Substation {sub_id}")
                    except Exception:
                        continue

    @property
    def n_actions(self) -> int:
        """Number of available actions."""
        return len(self._actions)

    def get_action(self, action_id: int) -> "BaseAction":
        """
        Get Grid2Op action by ID.

        Args:
            action_id: Index in reduced action space

        Returns:
            Grid2Op BaseAction
        """
        if action_id < 0 or action_id >= len(self._actions):
            raise IndexError(f"Action ID {action_id} out of range [0, {len(self._actions)})")
        return self._actions[action_id]

    def get_description(self, action_id: int) -> str:
        """
        Get human-readable description of action.

        Args:
            action_id: Index in reduced action space

        Returns:
            Description string
        """
        if action_id < 0 or action_id >= len(self._descriptions):
            return "Unknown Action"
        return self._descriptions[action_id]

    def get_all_descriptions(self) -> List[str]:
        """Get descriptions of all actions."""
        return self._descriptions.copy()

    def sample(self) -> int:
        """Sample random action ID."""
        return np.random.randint(0, self.n_actions)

    def get_action_mask(self, obs: Any) -> np.ndarray:
        """
        Get mask of legal actions given observation.

        Args:
            obs: Grid2Op observation

        Returns:
            Boolean array where True = action is legal
        """
        mask = np.ones(self.n_actions, dtype=bool)

        for i, action in enumerate(self._actions):
            try:
                # Check if action is legal
                _, _, _, info = obs.simulate(action)
                if info.get("is_illegal", False) or info.get("is_ambiguous", False):
                    mask[i] = False
            except Exception:
                mask[i] = False

        return mask

    def __len__(self) -> int:
        return self.n_actions

    def __repr__(self) -> str:
        return f"TopKActionSpace(k={self.k}, n_actions={self.n_actions})"


def create_action_space(
    env: "grid2op.Environment.Environment",
    k: int = 10
) -> TopKActionSpace:
    """
    Convenience function to create action space.

    Args:
        env: Grid2Op environment
        k: Number of actions

    Returns:
        TopKActionSpace instance
    """
    return TopKActionSpace(env, k=k)
