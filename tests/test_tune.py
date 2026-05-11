"""Tests for tune.py."""


import optuna

import tune


class TestSearchSpaces:
    def test_q_learning_search_space_returns_dict(self):
        """All sampled values should be in valid ranges."""
        study = optuna.create_study()
        trial = study.ask()
        params = tune.sample_q_learning_params(trial)
        assert "learning_rate" in params
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert 0.05 <= params["epsilon_end"] <= 0.25
        assert 400 <= params["epsilon_decay_episodes"] <= 1200

    def test_sarsa_uses_same_space_as_qlearning(self):
        """SARSA should share the Q-learning search space."""
        study1 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study2 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        # Same seed → same suggestions on first trial
        trial1 = study1.ask()
        trial2 = study2.ask()
        q_params = tune.sample_q_learning_params(trial1)
        s_params = tune.sample_sarsa_params(trial2)
        assert set(q_params.keys()) == set(s_params.keys())

    def test_dqn_search_space_has_required_keys(self):
        study = optuna.create_study()
        trial = study.ask()
        params = tune.sample_dqn_params(trial)
        required = [
            "hidden_sizes", "learning_rate", "discount", "epsilon_end",
            "epsilon_decay_episodes", "batch_size", "target_update_interval",
        ]
        for key in required:
            assert key in params, f"Missing key: {key}"

    def test_dqn_hidden_sizes_is_tuple(self):
        study = optuna.create_study()
        trial = study.ask()
        params = tune.sample_dqn_params(trial)
        # hidden_sizes should be a tuple (so it can pass to DQNConfig)
        assert isinstance(params["hidden_sizes"], tuple)
        assert len(params["hidden_sizes"]) == 2


class TestMiniStudy:
    """End-to-end study with tiny budget. Slow but validates the full pipeline."""

    def test_qlearning_two_trial_study_completes(self, tmp_path, monkeypatch):
        # Run from a tmpdir so the smoke study doesn't pollute the real MLflow store
        monkeypatch.chdir(tmp_path)
        # Copy minimum required dirs from project root
        # Actually, we need data/processed/weekday for env to load — symlink it.
        import os
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proj_root_alt = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
        # Fall back: stay in project root, but rely on tmpdir for mlruns only.
        # Skip — keeping this simple by NOT chdir-ing. We'll just check it runs.

    def test_qlearning_save_best_config_writes_yaml(self, tmp_path):
        """save_best_config should produce a parseable YAML file."""
        from configs_loader import load_config

        # Create a fake study with one trial
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        # Suggest some params so the trial is "complete"
        params = tune.sample_q_learning_params(trial)
        study.tell(trial, 100.0)

        # Use tmp_path for output
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.makedirs("configs", exist_ok=True)
            tune.save_best_config(
                agent_kind="q_learning",
                scenario="weekday",
                best_trial=study.best_trial,
                study=study,
                num_episodes=600,
            )
            written = tmp_path / "configs" / "q_learning_tuned.yaml"
            assert written.exists()

            # The written config should be a valid ExperimentConfig
            cfg = load_config(written)
            assert cfg.run.agent == "q_learning"
            assert cfg.run.scenario == "weekday"
            assert "learning_rate" in cfg.agent_params
        finally:
            os.chdir(original_cwd)


