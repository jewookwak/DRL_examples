# envs/components/reward.py
class Reward:
    def __init__(self):
        pass

    def process(self, reward):
        return (reward + 8) / 8  # Normalize reward as per original code