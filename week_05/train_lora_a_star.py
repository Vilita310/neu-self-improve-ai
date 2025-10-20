
from utils import load_data
from rl.lora_trainer import train_lora_a_star

def main():
    data = load_data("math_mini")
    train_lora_a_star(None, data, 0.8)

if __name__ == "__main__":
    main()
