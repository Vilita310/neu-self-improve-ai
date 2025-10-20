
from utils import load_data
from rl.a_star_po import offline_estimate_vstar

def main():
    data = load_data("math_mini")
    samples = [{"reward": 1.0} for _ in data]
    v_star = offline_estimate_vstar(samples)
    print("[INFO] Estimated V* =", v_star)

if __name__ == "__main__":
    main()
