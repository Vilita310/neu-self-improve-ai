import json
import random

# Product templates by category
PRODUCTS = {
    "shoes": [
        ("Running Shoes", ["running", "athletic", "sports", "jogging"]),
        ("Sneakers", ["casual", "street", "lifestyle"]),
        ("Boots", ["hiking", "winter", "work"]),
        ("Sandals", ["summer", "beach", "casual"]),
        ("Loafers", ["dress", "formal", "office"]),
        ("Slippers", ["indoor", "comfortable", "home"]),
        ("High Heels", ["formal", "party", "elegant"]),
        ("Flip Flops", ["beach", "pool", "summer"]),
    ],
    
    "electronics": [
        ("Earbuds", ["wireless", "bluetooth", "audio"]),
        ("Headphones", ["over-ear", "noise-canceling", "audio"]),
        ("Smart Watch", ["fitness", "tracking", "wearable"]),
        ("Phone Charger", ["fast", "USB-C", "wireless"]),
        ("Power Bank", ["portable", "charging", "battery"]),
        ("Bluetooth Speaker", ["portable", "wireless", "audio"]),
        ("Laptop Stand", ["ergonomic", "adjustable", "desk"]),
        ("USB Cable", ["charging", "data", "transfer"]),
        ("Keyboard", ["mechanical", "wireless", "gaming"]),
        ("Mouse", ["wireless", "ergonomic", "gaming"]),
    ],
    
    "apparel": [
        ("Hoodie", ["comfortable", "casual", "warm"]),
        ("T-Shirt", ["casual", "cotton", "basic"]),
        ("Jeans", ["denim", "casual", "pants"]),
        ("Jacket", ["outdoor", "winter", "coat"]),
        ("Sweater", ["warm", "knit", "cozy"]),
        ("Dress", ["formal", "party", "elegant"]),
        ("Shorts", ["summer", "casual", "athletic"]),
        ("Leggings", ["athletic", "yoga", "comfortable"]),
        ("Sweatpants", ["comfortable", "casual", "lounge"]),
        ("Tank Top", ["summer", "sleeveless", "casual"]),
    ],
    
    "fitness": [
        ("Yoga Mat", ["exercise", "non-slip", "workout"]),
        ("Dumbbells", ["weights", "strength", "training"]),
        ("Resistance Bands", ["exercise", "portable", "workout"]),
        ("Jump Rope", ["cardio", "exercise", "training"]),
        ("Foam Roller", ["recovery", "massage", "exercise"]),
        ("Water Bottle", ["hydration", "sports", "insulated"]),
        ("Yoga Block", ["support", "exercise", "stretching"]),
        ("Exercise Ball", ["core", "stability", "workout"]),
        ("Kettlebell", ["strength", "training", "weights"]),
        ("Pull-up Bar", ["home", "gym", "strength"]),
    ],
    
    "home": [
        ("Pillow", ["comfortable", "soft", "bedroom"]),
        ("Blanket", ["warm", "cozy", "throw"]),
        ("Lamp", ["desk", "LED", "lighting"]),
        ("Picture Frame", ["photo", "decorative", "wall"]),
        ("Candle", ["scented", "aromatherapy", "decorative"]),
        ("Vase", ["decorative", "ceramic", "glass"]),
        ("Clock", ["wall", "alarm", "digital"]),
        ("Rug", ["floor", "decorative", "area"]),
        ("Curtains", ["window", "blackout", "decorative"]),
        ("Storage Box", ["organizer", "container", "home"]),
    ],
    
    "kitchen": [
        ("Coffee Mug", ["ceramic", "insulated", "drinking"]),
        ("Water Bottle", ["reusable", "stainless", "hydration"]),
        ("Cutting Board", ["wood", "kitchen", "prep"]),
        ("Knife Set", ["kitchen", "cooking", "chef"]),
        ("Mixing Bowl", ["stainless", "cooking", "prep"]),
        ("Frying Pan", ["non-stick", "cooking", "kitchen"]),
        ("Spatula", ["silicone", "cooking", "utensil"]),
        ("Measuring Cups", ["baking", "cooking", "kitchen"]),
        ("Blender", ["smoothie", "kitchen", "appliance"]),
        ("Toaster", ["breakfast", "kitchen", "appliance"]),
    ],
    
    "accessories": [
        ("Backpack", ["travel", "school", "storage"]),
        ("Wallet", ["leather", "card", "holder"]),
        ("Sunglasses", ["UV", "protection", "fashion"]),
        ("Watch", ["analog", "digital", "timepiece"]),
        ("Belt", ["leather", "fashion", "accessory"]),
        ("Hat", ["cap", "fashion", "sun"]),
        ("Scarf", ["warm", "winter", "fashion"]),
        ("Gloves", ["winter", "warm", "hand"]),
        ("Umbrella", ["rain", "portable", "weather"]),
        ("Keychain", ["decorative", "practical", "accessory"]),
    ],
    
    "books": [
        ("Novel", ["fiction", "reading", "paperback"]),
        ("Cookbook", ["recipes", "cooking", "kitchen"]),
        ("Journal", ["notebook", "writing", "diary"]),
        ("Planner", ["organizer", "calendar", "productivity"]),
        ("Coloring Book", ["adult", "relaxation", "art"]),
        ("Travel Guide", ["tourism", "travel", "reference"]),
        ("Magazine", ["monthly", "reading", "subscription"]),
        ("Comic Book", ["graphic", "reading", "illustrated"]),
    ],
}

# Color options
COLORS = [
    "Red", "Blue", "Black", "White", "Green", "Yellow", "Orange", "Purple",
    "Pink", "Gray", "Brown", "Navy", "Teal", "Burgundy", "Olive", "Beige",
    "Charcoal", "Silver", "Gold", "Turquoise"
]

# Material/style modifiers
MODIFIERS = [
    "Premium", "Classic", "Modern", "Vintage", "Deluxe", "Basic", "Pro",
    "Ultra", "Eco-Friendly", "Lightweight", "Heavy-Duty", "Compact", "Portable"
]

def generate_products(target_count=500):
    """Generate a diverse dataset of products."""
    products = []
    product_id = 1
    
    # Calculate products per category
    categories = list(PRODUCTS.keys())
    products_per_category = target_count // len(categories)
    
    for category, items in PRODUCTS.items():
        category_count = 0
        
        # Generate multiple variations of each product type
        for base_name, attributes in items:
            # Create variations with colors and modifiers
            variations = []
            
            # Add color variations
            for color in COLORS:
                variations.append(f"{color} {base_name}")
            
            # Add modifier variations
            for modifier in MODIFIERS:
                variations.append(f"{modifier} {base_name}")
            
            # Shuffle and pick variations
            random.shuffle(variations)
            
            # Determine how many variations to use
            num_variations = min(
                len(variations), 
                (products_per_category - category_count) // max(1, len(items) - items.index((base_name, attributes)))
            )
            
            for i in range(num_variations):
                if category_count >= products_per_category:
                    break
                
                title = variations[i]
                
                # Generate price (varies by category)
                price_ranges = {
                    "shoes": (30, 150),
                    "electronics": (15, 200),
                    "apparel": (20, 100),
                    "fitness": (10, 80),
                    "home": (15, 120),
                    "kitchen": (10, 90),
                    "accessories": (10, 80),
                    "books": (10, 40),
                }
                min_price, max_price = price_ranges.get(category, (10, 100))
                price = random.randint(min_price, max_price)
                
                # Create reward tag (key words for matching)
                words = title.lower().split()
                reward_tag = " ".join(words[:2])  # First two words
                
                # Create target phrase
                target = f"buy {title.lower()}"
                
                product = {
                    "id": product_id,
                    "title": title,
                    "price": price,
                    "category": category,
                    "reward_tag": reward_tag,
                    "target": target,
                    "attributes": attributes
                }
                
                products.append(product)
                product_id += 1
                category_count += 1
    
    return products

def save_dataset(products, filename="data/webshop_mock.json"):
    """Save products to JSON file."""
    with open(filename, 'w') as f:
        json.dump(products, f, indent=2)
    
    print(f"✅ Generated {len(products)} products")
    print(f"✅ Saved to {filename}")
    
    # Print statistics
    categories = {}
    for product in products:
        cat = product['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Products per category:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}")
    
    # Print sample products
    print("\n📦 Sample products:")
    for i in range(min(5, len(products))):
        p = products[i]
        print(f"   {p['id']}. {p['title']} - ${p['price']} ({p['category']})")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    print("Generating WebShop dataset...")
    products = generate_products(target_count=500)
    
    save_dataset(products)
    
    print("\n✅ Dataset generation complete!")