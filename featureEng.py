import numpy as np

# Load the data
data = np.load('preprocessed_chess_data.npz')

# See what's inside
print("Keys in file:", data.files)

# Load the arrays
X = data['X']  # Bitboards
y = data['y']  # Evaluations

print(X[:5])

# Check shapes and info
print(f"X shape: {X.shape}")  # Should be (N, 12, 64)
print(f"y shape: {y.shape}")  # Should be (N,)
print(f"Number of positions: {len(X)}")
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

# Look at some statistics
print("\nEvaluation stats:")
print(f"  Min: {y.min()}")
print(f"  Max: {y.max()}")
print(f"  Mean: {y.mean():.2f}")
print(f"  Median: {np.median(y):.2f}")

# View a sample position
print(f"\nFirst position bitboards shape: {X[0].shape}")
print(f"First position evaluation: {y[0]}")

# Feature engineering
# 1. Piece count
# 2. Piece mobility
    # 1. bishop
    # 2. knight
    # 3. rook
    # 4. queen
# 3. Center control
# 1. Piece pawn structure
    # 1. pawn chain
    # 2. pawn island


def center_control(bitboards):
    return np.sum(bitboards)

def pawn_structure(bitboards):
    return np.sum(bitboards)