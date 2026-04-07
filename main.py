import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entry point for RL tasks.")
    parser.add_argument(
        "command",
        choices=["test", "ppo", "sac", "tdmpc", "plot"],
        help="Action to run.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "test":
        from test_env import main as test_env_main

        test_env_main()
    elif args.command == "ppo":
        from train_ppo_mac import main as train_ppo_main

        train_ppo_main()
    elif args.command == "sac":
        from train_sac_mac import main as train_sac_main

        train_sac_main()
    elif args.command == "tdmpc":
        from tdmpc2.train_tdmpc2 import main as train_tdmpc_main

        train_tdmpc_main()
    elif args.command == "plot":
        from plot_results import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
