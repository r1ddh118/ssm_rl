from tdmpc2.train_tdmpc2 import main as train_main


def main() -> None:
    train_main(dynamics_type="s5", run_name="tdmpc2_walker_s5")


if __name__ == "__main__":
    main()
