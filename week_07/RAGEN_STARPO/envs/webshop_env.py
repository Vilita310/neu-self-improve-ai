import json
import os
import random

class WebShopEnv:
    """
    FIXED WebShop Environment with:
    1. Randomized targets
    2. Dense reward shaping
    3. Dynamic action space
    4. Better state representation
    """
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, 'data', 'webshop_mock.json')
        
        self.products = self._load_products(dataset_path)
        self.reset()
    
    def _load_products(self, path):
        """Load product catalog."""
        with open(path, 'r') as f:
            products = json.load(f)
        
        self.product_list = products
        self.product_map = {p['id']: p for p in products}
        return products
    
    def reset(self):
        """Reset with randomized target."""
        self.done = False
        self.current_search_results = []
        self.clicked_items = []
        self.step_count = 0
        
        # RANDOMIZE TARGET - Learn to generalize!
        target_product = random.choice(self.products)
        self.target_id = target_product['id']
        self.target_title = target_product['title']
        self.target_category = target_product['category']
        
        # Extract key attributes for matching
        words = self.target_title.lower().split()
        self.target_keywords = set(words)
        
        # Create natural language target
        self.target = f"User wants to buy {self.target_title.lower()}."
        
        product_count = len(self.products)
        self.last_obs = (
            f"TASK: {self.target}\n"
            f"Available: {product_count} products across {len(set(p['category'] for p in self.products))} categories.\n"
            f"Commands: 'search [keywords]', 'click [id]', 'buy [id]'"
        )
        return self.last_obs
    
    def _calculate_relevance_score(self, product_title):
        """Calculate how relevant a product is to the target."""
        title_words = set(product_title.lower().split())
        
        # Exact match
        if product_title.lower() == self.target_title.lower():
            return 1.0
        
        # Partial match - count overlapping keywords
        overlap = len(self.target_keywords & title_words)
        score = overlap / max(len(self.target_keywords), 1)
        
        return score
    
    def step(self, action: str):
        """Environment step with DENSE REWARD SHAPING."""
        if self.done:
            return self.last_obs, 0, True
        
        self.step_count += 1
        action = action.lower().strip()
        reward = 0
        obs = ""
        
        # === SEARCH ACTION ===
        if action.startswith("search"):
            keyword_str = action.replace("search", "").strip()
            keywords = keyword_str.split()
            
            # Find matching products
            matches = []
            for p in self.products:
                title_lower = p["title"].lower()
                if all(kw in title_lower for kw in keywords):
                    matches.append(p)
            
            self.current_search_results = matches[:20]  # Top 20 results
            
            if matches:
                results_str = ", ".join([
                    f"{p['title']} (ID {p['id']})" 
                    for p in self.current_search_results[:5]
                ])
                obs = f"Found {len(matches)} results. Top 5: {results_str}"
                
                # REWARD SHAPING: Good search gets small reward
                best_relevance = max(
                    self._calculate_relevance_score(p['title']) 
                    for p in self.current_search_results
                )
                
                if best_relevance > 0.5:
                    reward = 0.2  # Found relevant items
                    obs += " [GOOD SEARCH]"
                elif best_relevance > 0.3:
                    reward = 0.1  # Somewhat relevant
            else:
                obs = f"No results for '{keyword_str}'."
                reward = -0.05  # Slight penalty for bad search
        
        # === CLICK ACTION ===
        elif action.startswith("click"):
            try:
                item_id = int(action.split()[1])
                
                if item_id in self.product_map:
                    product = self.product_map[item_id]
                    self.clicked_items.append(product)
                    
                    obs = (
                        f"Viewing: {product['title']}\n"
                        f"Price: ${product['price']}\n"
                        f"Category: {product['category']}"
                    )
                    
                    # REWARD SHAPING: Clicking relevant items gets reward
                    relevance = self._calculate_relevance_score(product['title'])
                    
                    if relevance >= 0.8:
                        reward = 0.3  # Very relevant
                        obs += " [HIGHLY RELEVANT]"
                    elif relevance >= 0.5:
                        reward = 0.15  # Somewhat relevant
                        obs += " [RELEVANT]"
                    else:
                        reward = -0.05  # Not relevant
                else:
                    obs = f"Product ID {item_id} not found."
                    reward = -0.1
            except (ValueError, IndexError):
                obs = "Invalid click command. Use: click [id]"
                reward = -0.1
        
        # === BUY ACTION ===
        elif action.startswith("buy"):
            try:
                item_id = int(action.split()[1])
                
                if item_id in self.product_map:
                    product = self.product_map[item_id]
                    obs = f"Purchased: {product['title']}"
                    
                    # FINAL REWARD: Only perfect match gets +1
                    if item_id == self.target_id:
                        reward = 1.0  # CORRECT!
                        obs += " ✓ CORRECT PURCHASE!"
                    else:
                        # Partial credit for similar items
                        relevance = self._calculate_relevance_score(product['title'])
                        reward = relevance * 0.3 - 0.2  # Max +0.1, typically negative
                        obs += f" ✗ Wrong item (wanted: {self.target_title})"
                    
                    self.done = True
                else:
                    obs = f"Product ID {item_id} not found."
                    reward = -0.2
                    self.done = True
            except (ValueError, IndexError):
                obs = "Invalid buy command. Use: buy [id]"
                reward = -0.1
        
        else:
            obs = "Invalid action. Use: 'search [keywords]', 'click [id]', or 'buy [id]'"
            reward = -0.1
        
        # Timeout penalty
        if self.step_count >= 15:
            self.done = True
            reward -= 0.3
            obs += " [TIMEOUT]"
        
        self.last_obs = f"OBS: {obs}\nREWARD: {reward:.2f} | DONE: {self.done}"
        return self.last_obs, reward, self.done


# === TEST ===
if __name__ == "__main__":
    env = WebShopEnv()
    
    print("=== TESTING FIXED ENVIRONMENT ===\n")
    
    # Test 3 random episodes
    for ep in range(3):
        print(f"\n--- Episode {ep+1} ---")
        obs = env.reset()
        print(obs)
        
        # Extract target for testing
        target_words = env.target_title.lower().split()[:2]
        search_query = " ".join(target_words)
        
        print(f"\n1. Searching for: '{search_query}'")
        obs, r, done = env.step(f"search {search_query}")
        print(f"Reward: {r:.2f}")
        print(obs)
        
        if env.current_search_results:
            first_result = env.current_search_results[0]
            
            print(f"\n2. Clicking ID {first_result['id']}")
            obs, r, done = env.step(f"click {first_result['id']}")
            print(f"Reward: {r:.2f}")
            print(obs)
            
            print(f"\n3. Buying ID {first_result['id']}")
            obs, r, done = env.step(f"buy {first_result['id']}")
            print(f"Reward: {r:.2f}")
            print(obs)