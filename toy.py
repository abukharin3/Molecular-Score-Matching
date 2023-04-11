from runners.toy_runner import ToyRunner

if __name__ == "__main__":
    runner = ToyRunner(None, None)
    print("Runner Loaded")
    runner.train_doublewell(num_samples=10000)