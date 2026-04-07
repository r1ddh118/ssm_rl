from env_setup import make_env


def main():
    env = make_env("hopper", "hop")

    obs = env.reset()
    print("Initial observation shape:", obs[0].shape)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done[0]:
            obs = env.reset()

    print("Environment smoke test passed.")
    env.close()


if __name__ == "__main__":
    main()
