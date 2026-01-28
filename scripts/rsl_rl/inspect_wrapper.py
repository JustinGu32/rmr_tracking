import inspect
import sys
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    print("Found RslRlVecEnvWrapper")
    print("Source code of step method:")
    print(inspect.getsource(RslRlVecEnvWrapper.step))
except ImportError:
    print("Could not import isaaclab_rl.rsl_rl.RslRlVecEnvWrapper")
except Exception as e:
    print(f"Error: {e}")
