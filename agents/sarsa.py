
"""
SARSA agent — on-policy temporal-difference learning.

The only difference from Q-learning: the bootstrapped value uses the action
that was *actually selected* in the next state (a'), rather than the max
over all actions. This makes SARSA on-policy: it learns the value of the
exploration policy it follows, not the optimal policy.

In environments with risky exploration, SARSA tends to be more conservative
than Q-learning. In our food rescue context, that usually means slightly
fewer "I'll commit to a far-away donor and hope" decisions.

Algorithm
---------
Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))

where a' is the action ε-greedy would actually choose in s'.

Implementation
--------------
We inherit from QLearningAgent and override only update_from_transition.
The action selection logic, save/load, epsilon decay, and discretization
all stay identical.
"""

from __future__ import annotations

from typing import Optional


from agents.q_learning import QLearningAgent, discretize_state
from sim.environment import FoodRescueEnv


class SARSAAgent(QLearningAgent):
    """SARSA: on-policy variant of Q-learning."""

    name = "sarsa"

    def update_from_transition(
        self,
        env_before: FoodRescueEnv,
        action: int,
        reward: float,
        env_after: FoodRescueEnv,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        """
        SARSA Bellman update.

        Unlike Q-learning, this needs `next_action` — the action that ε-greedy
        actually picked in the post-step state. The training loop is responsible
        for calling select_action() in env_after BEFORE calling this.
        """
        if not self._training:
            return

        if next_action is None and not done:
            # If caller didn't supply next_action, pick it now (still on-policy
            # because we use the same epsilon-greedy that the env will use).
            next_action = self.select_action(env_after, obs=None)  # obs unused in our impl

        state_before = discretize_state(
            env_before, self.config.pos_buckets, self.config.load_buckets
        )
        state_after = discretize_state(
            env_after, self.config.pos_buckets, self.config.load_buckets
        )

        q_row = self._ensure_q_row(state_before)
        old_q = q_row[action]

        if done:
            target = reward
        else:
            q_next_row = self._q_table.get(state_after)
            if q_next_row is None:
                next_value = self.config.optimistic_init
            else:
                next_value = float(q_next_row[next_action])
            target = reward + self.config.discount * next_value

        q_row[action] = old_q + self.config.learning_rate * (target - old_q)
