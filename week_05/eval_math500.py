
from utils import load_data
from rl.pag import verify_and_revise

def main():
    data = load_data("math_mini")
    for d in data[:5]:
        print(verify_and_revise(d["problem"], d["answer"]))

if __name__ == "__main__":
    main()
