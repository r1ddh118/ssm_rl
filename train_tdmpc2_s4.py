from tdmpc2.train_tdmpc2 import main as train_main


def main(total_steps: int = 10) -> None:
    train_main(dynamics_type="s4", run_name="tdmpc2_walker_s4", total_steps=total_steps)


if __name__ == "__main__":
    main()
