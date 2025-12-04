# MaAS FEVER Benchmark Analysis

## 1. Failure Analysis (Easiest Examples)
Root Cause: **Retrieval Bias**. The system retrieves documents containing keywords but fails to verify specific relationships (often negation).

### Case 1
- **Claim**: "Seohyun sings."
- **Gold**: SUPPORTS | **Predicted**: NOT ENOUGH INFO
- **Analysis**: None

### Case 2
- **Claim**: "Seohyun acts."
- **Gold**: SUPPORTS | **Predicted**: NOT ENOUGH INFO
- **Analysis**: None

### Case 3
- **Claim**: "Augustus died."
- **Gold**: SUPPORTS | **Predicted**: NOT ENOUGH INFO
- **Analysis**: None

### Case 4
- **Claim**: "The Cretaceous ended."
- **Gold**: SUPPORTS | **Predicted**: NOT ENOUGH INFO
- **Analysis**: None

### Case 5
- **Claim**: "Paramore is Canadian."
- **Gold**: REFUTES | **Predicted**: SUPPORTS
- **Analysis**: Retrieval Bias: The system retrieved documents matching the entities but failed to identify the negation context.

## 2. Success Analysis (Hardest Examples)
Success Factor: **Dynamic Topology**. The system evolved a deeper structure for complex claims.

### Case 1
- **Claim**: "Tremont Street Subway served a light rail station on the MBTA Green Line system, and is located on the southeast corner of Boston Common at the intersection of Boylston Street and Tremont Street called Boylston station."
- **Architecture**: None
- **Reasoning**: None

### Case 2
- **Claim**: "Tremont Street Subway served a light rail station on the MBTA Green Line system, and is located on the southeast corner of Boston Common at the intersection of Boylston Street and Tremont Street."
- **Architecture**: None
- **Reasoning**: None

### Case 3
- **Claim**: "South African Communist Party is a partner of an alliance between the African National Congress (ANC), the Congress of South African Trade Unions (COSATU) and the South African Communist Party (SACP)."
- **Architecture**: planner -> multi_hop_retrieve -> verifier
- **Reasoning**: Query Decomposition: The planner successfully broke down the complex claim into sub-questions.

### Case 4
- **Claim**: "Mohra got Filmfare nominations for Best Film, Best Director, Best Actor, Best Supporting Actress, Best Male Debut, Best Story, Best Editing, Best Cinematography, and Best Art Direction."
- **Architecture**: None
- **Reasoning**: None

### Case 5
- **Claim**: "Australia (2008 film) production took place in a town and locality in the Whitsunday Region on the eastern coast of Queensland, Australia called Bowen."
- **Architecture**: planner -> multi_hop_retrieve -> verifier
- **Reasoning**: Query Decomposition: The planner successfully broke down the complex claim into sub-questions.

