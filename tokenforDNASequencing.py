"""
This Python implementation demonstrates the core logic of a Byte-Pair Encoding (BPE) trainer. It starts with individual DNA bases and iteratively merges the most frequent adjacent pairs into new "motif" tokens.
"""
import collections

def get_stats(ids):
    """Count occurrences of adjacent pairs in the sequence."""
    counts = collections.defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    """Replace all occurrences of a specific pair with a new token ID."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# --- Training Simulation ---
sequence = "ATGCCGATGCCGATGCCG"
# Convert string to initial character-level IDs (ASCII)
tokens = [ord(c) for c in sequence]
vocab = {65: 'A', 84: 'T', 71: 'G', 67: 'C'}

num_merges = 3
for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats: break
    
    # Find the most frequent pair
    best_pair = max(stats, key=stats.get)
    new_token_id = 256 + i  # Start new IDs above standard ASCII range
    
    # Merge the pair throughout the sequence
    tokens = merge(tokens, best_pair, new_token_id)
    
    # Update vocabulary map
    vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
    print(f"Merge {i+1}: Created '{vocab[new_token_id]}' from {best_pair}")

print(f"\nFinal Tokenized Sequence: {tokens}")
print(f"Final Vocabulary: {vocab}")

"""
How this works for AI:Initial State: The model sees A-T-G-C-C-G.Learning: The algorithm identifies that AT appears frequently and creates a single token for it.Compression: After several merges, a complex sequence like ATGCCG might be represented by just one or two IDs instead of six.Utility: This allows DNA-specific models like HyenaDNA to handle much longer genomic sequences by reducing the "word count" the AI has to read.
"""
