from runners.toy_runner import ToyRunner
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    runner = ToyRunner(None, None)
    print("Runner Loaded")
    # runner.train_doublewell()
    # runner.train_force_doublewell()
    runner.doublewell_md_true()