# main.py
import argparse
from pure_pbrs.configs import DroneConfig, BackdoorConfig
from pure_pbrs.runner import setup_env, setup_model, run_phase

def parse_args():
    p = argparse.ArgumentParser(description="Pure PBRS: train/eval runner")
    p.add_argument("--phase", choices=["benign", "benign_advanced", "backdoor", "testing"], default="benign")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--train_backdoor", action="store_true", help="Enable poisoning callback during learning")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    drone_cfg = DroneConfig()
    backdoor_cfg = BackdoorConfig()

    env = setup_env(drone_cfg, backdoor_cfg, verbose=args.verbose)
    model = setup_model(env, verbose=0)

    run_phase(model, env, phase=args.phase, total_timesteps=args.timesteps,
              train_backdoor=args.train_backdoor, verbose=args.verbose)

    print("Done.")

if __name__ == "__main__":
    main()
