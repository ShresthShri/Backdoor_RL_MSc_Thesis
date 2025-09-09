"""
This script implements a 3-stage curriculum training pipeline for a reinforcement learning
agent using the Stable Baselines3 library. The agent is trained to navigate a drone in a
benign environment, then in a more challenging environment with obstacles, and finally to
learn a backdoor attack using the Pure PBRS framework.

Usage:
Pure PBRS (default):
    python scripts/train_curriculum.py --save-dir runs/pure 

Other models:
    python scripts/train_curriculum.py --env-package <PATH TO SPECIFIC ENV>--save-dir runs/single 

Resume Stage 3 from a Stage 2 checkpoint:
    python scripts/train_curriculum.py --env-package pure_pbrs --save-dir runs/pure_resume --load-stage2-model runs/pure/stage2_final.zip

Adjust timesteps per stage:
    python scripts/train_curriculum.py --save-dir runs/custom --stage1-steps 150000 --stage2-steps 250000 --stage3-steps 120000
"""

import os
import json
import argparse
import importlib
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import stable_baselines3 as sb3_pkg  


def parse_args():
    p = argparse.ArgumentParser(
        description="3-stage curriculum training for backdoor RL (env package selectable)"
    )
    p.add_argument("--save-dir", type=str, default="artifacts", help="Directory to save models and logs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--load-stage2-model", type=str, default=None, help="Path to Stage 2 model to resume from")
    p.add_argument("--stage1-steps", type=int, default=200_000)
    p.add_argument("--stage2-steps", type=int, default=300_000)
    p.add_argument("--stage3-steps", type=int, default=150_000)
    p.add_argument(
        "--env-package",
        type=str,
        default="pure_pbrs",
        choices=["pure_pbrs", "single_net_pbrs", "dual_net_pbrs", "relara_reward_aug"],
        help="Which env/agent package to use",
    )
    return p.parse_args()


def set_sac_learning_rate(model: SAC, lr: float):
    for opt in [model.actor.optimizer, model.critic.optimizer]:
        for g in opt.param_groups:
            g["lr"] = lr
    if getattr(model, "ent_coef_optimizer", None) is not None:
        for g in model.ent_coef_optimizer.param_groups:
            g["lr"] = lr


def create_sac_model(env, learning_rate=3e-4, batch_size=256, seed=42) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
        verbose=1,
        seed=seed,
        device="auto",
    )


def attack_eval(env_cls, model: SAC, mode="nfz_hover", episodes=5, spoof_on_step=30):
    """Lightweight cross-package evaluation; tolerates small API differences."""
    env = env_cls(verbose=0)
    # phase
    if hasattr(env, "configure_for_phase"):
        env.configure_for_phase("testing")

    # load ReLaRa if present
    relara_path = os.path.join(getattr(model, "load_path", "."), "relara_agent.pth")
    if hasattr(env, "relara_agent") and relara_path and os.path.exists(relara_path):
        try:
            env.relara_agent.load_state(relara_path)
        except Exception:
            pass

    successes, reassigns = 0, 0
    for _ in range(episodes):
        obs, _ = env.reset()

        # disarm/arm differences across packages
        if hasattr(env, "disarm_attack"):
            try:
                env.disarm_attack()
            except Exception:
                pass

        if hasattr(env, "arm_attack"):
            # Some envs accept (mode, spoof_on_step), others may have extra flags
            try:
                env.arm_attack(mode, spoof_on_step=spoof_on_step)
            except TypeError:
                # fallback with force kw if supported
                try:
                    env.arm_attack(mode, spoof_on_step=spoof_on_step, force=True)
                except Exception:
                    pass

        frozen = getattr(env, "selected_attack_mode", None)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get("selected_attack_mode") not in (None, frozen):
                reassigns += 1
        successes += int(info.get("attack_success", False))

    print(f"[{mode}] success_rate={successes/episodes:.2f}, reassign_events={reassigns}")


def curriculum_training(
    env_pkg: str,
    save_dir="artifacts",
    seed=42,
    load_stage2_model=None,
    stage1_steps=200_000,
    stage2_steps=300_000,
    stage3_steps=150_000,
):
    os.makedirs(save_dir, exist_ok=True)
    for d in ["stage1_checkpoints", "stage2_checkpoints", "stage3_checkpoints"]:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

    # dynamic imports from selected package
    mod = importlib.import_module(f"{env_pkg}.env")
    cfg = importlib.import_module(f"{env_pkg}.configs")
    cbs = importlib.import_module(f"{env_pkg}.callbacks")

    DroneNavigationEnv = mod.DroneNavigationEnv
    DroneConfig = cfg.DroneConfig
    BackdoorConfig = cfg.BackdoorConfig
    SleeperNetsBackdoorCallback = getattr(cbs, "SleeperNetsBackdoorCallback", None)
    ReLaraTrainingCallback = getattr(cbs, "ReLaraTrainingCallback", None)

    np.random.seed(seed)
    torch.manual_seed(seed)

    drone_cfg = DroneConfig()
    backdoor_cfg = BackdoorConfig()

    base_env = DroneNavigationEnv(drone_cfg, backdoor_cfg, verbose=1)
    env = Monitor(base_env)

    # Stage 1/2
    if load_stage2_model:
        print(f"Loading Stage 2 model from: {load_stage2_model}")
        model = SAC.load(load_stage2_model, env=env)
    else:
        print("Stage 1: Basic navigation training")
        if hasattr(base_env, "configure_for_phase"):
            base_env.configure_for_phase("benign")
        model = create_sac_model(env, learning_rate=3e-4, seed=seed)
        ck1 = CheckpointCallback(
            save_freq=20_000,
            save_path=os.path.join(save_dir, "stage1_checkpoints"),
            name_prefix="stage1",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        model.learn(total_timesteps=stage1_steps, progress_bar=True, callback=ck1)
        model.save(os.path.join(save_dir, "stage1_final"))

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Stage 1 performance: {mean_reward:.2f} ± {std_reward:.2f}")

        print("Stage 2: Navigation with obstacles")
        if hasattr(base_env, "configure_for_phase"):
            base_env.configure_for_phase("benign_advanced")
        set_sac_learning_rate(model, 2e-4)
        ck2 = CheckpointCallback(
            save_freq=20_000,
            save_path=os.path.join(save_dir, "stage2_checkpoints"),
            name_prefix="stage2",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        model.learn(total_timesteps=stage2_steps, progress_bar=True, reset_num_timesteps=False, callback=ck2)
        model.save(os.path.join(save_dir, "stage2_final"))

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Stage 2 performance: {mean_reward:.2f} ± {std_reward:.2f}")

    # Stage 3
    print("Stage 3: Backdoor training")
    if hasattr(base_env, "configure_for_phase"):
        base_env.configure_for_phase("backdoor")
    set_sac_learning_rate(model, 1e-4)

    callbacks = []
    if SleeperNetsBackdoorCallback is not None:
        callbacks.append(SleeperNetsBackdoorCallback(backdoor_cfg, verbose=1))
    if ReLaraTrainingCallback is not None:
        callbacks.append(ReLaraTrainingCallback(train_freq=100, verbose=1))
    callbacks.append(CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(save_dir, "stage3_checkpoints"),
        name_prefix="stage3",
        save_replay_buffer=True,
        save_vecnormalize=True,
    ))

    model.learn(
        total_timesteps=stage3_steps,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=callbacks if len(callbacks) > 1 else callbacks[0],
    )

    final_path = os.path.join(save_dir, "final_backdoored_model")
    model.save(final_path)
    print(f"Final backdoored model saved to {final_path}")

    # Save ReLaRa if present
    if hasattr(base_env, "relara_agent"):
        relara_path = os.path.join(save_dir, "relara_agent.pth")
        try:
            base_env.relara_agent.save_state(relara_path)
            print(f"ReLaRa agent saved to {relara_path}")
        except Exception:
            pass

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "env_package": env_pkg,
        "loaded_stage2_model": load_stage2_model,
        "total_timesteps": {
            "stage1": 0 if load_stage2_model else stage1_steps,
            "stage2": 0 if load_stage2_model else stage2_steps,
            "stage3": stage3_steps,
        },
        "sb3_version": getattr(sb3_pkg, "__version__", ""),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model, env, DroneNavigationEnv


def run_evaluation(env_cls, model_dir: str, model_path: str | None = None):
    model_path = model_path or os.path.join(model_dir, "final_backdoored_model")
    env = env_cls(verbose=0)
    if hasattr(env, "configure_for_phase"):
        env.configure_for_phase("testing")

    relara_path = os.path.join(model_dir, "relara_agent.pth")
    if hasattr(env, "relara_agent") and os.path.exists(relara_path):
        try:
            env.relara_agent.load_state(relara_path)
            print(f"Loaded ReLaRa agent from {relara_path}")
        except Exception:
            pass

    model = SAC.load(model_path, env=env)
    print("Evaluating benign navigation...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Benign performance: {mean_reward:.2f} ± {std_reward:.2f}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "benign_mean_reward": float(mean_reward),
        "benign_std_reward": float(std_reward),
    }
    with open(os.path.join(model_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    args = parse_args()
    print("Curriculum training pipeline")
    print(f"Env package: {args.env_package}")
    print(f"Save directory: {args.save_dir}")
    print(f"Seed: {args.seed}")
    if args.load_stage2_model:
        print(f"Resuming from Stage 2 model: {args.load_stage2_model}")

    model, env, EnvCls = curriculum_training(
        env_pkg=args.env_package,
        save_dir=args.save_dir,
        seed=args.seed,
        load_stage2_model=args.load_stage2_model,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        stage3_steps=args.stage3_steps,
    )
    _ = run_evaluation(EnvCls, args.save_dir)

    print("Pipeline completed.")

