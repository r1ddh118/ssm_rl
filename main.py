import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entry point for RL tasks.")
    parser.add_argument(
        "command",
        choices=[
            "test",
            "ppo",
            "sac",
            "tdmpc",
            "tdmpc-s4",
            "tdmpc-s5",
            "tdmpc-mamba",
            "phase3",
            "plot",
            "all-phases",
        ],
        help="Action to run.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10,
        help="Training steps for TD-MPC commands (default: 10 for quick runs).",
    )
    parser.add_argument(
        "--max-wall-clock-seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock cap for TD-MPC commands (0 disables time limit).",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="For plotting commands: plot only the Phase 3 horizon ablation.",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="For plotting commands: generate all phase plots.",
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

        train_tdmpc_main(
            total_steps=args.total_steps,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
        )
    elif args.command == "tdmpc-s4":
        from train_tdmpc2_s4 import main as train_tdmpc_s4_main

        train_tdmpc_s4_main(
            total_steps=args.total_steps,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
        )
    elif args.command == "tdmpc-s5":
        from train_tdmpc2_s5 import main as train_tdmpc_s5_main

        train_tdmpc_s5_main(
            total_steps=args.total_steps,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
        )
    elif args.command == "tdmpc-mamba":
        from train_tdmpc2_mamba import main as train_tdmpc_mamba_main

        train_tdmpc_mamba_main(
            total_steps=args.total_steps,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
        )
    elif args.command == "phase3":
        from tdmpc2.train_tdmpc2 import main as train_tdmpc_main

        train_tdmpc_main(
            dynamics_type="s5",
            plan_horizon=10,
            use_sam=True,
            use_info_prop=True,
            total_steps=args.total_steps,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
            run_name="tdmpc2_walker_s5_h10",
        )
    elif args.command == "plot":
        from plot_results import main as plot_main

        plot_args = []
        if args.all_phases:
            plot_args.append("--all-phases")
        elif args.phase3:
            plot_args.append("--phase3")
        plot_main(plot_args)
    elif args.command == "all-phases":
        from plot_results import main as plot_main

        plot_main(["--all-phases"])


if __name__ == "__main__":
    main()
