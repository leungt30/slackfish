import numpy as np

data = np.load('preprocessed_chess_data.npz')

print(data['bitboards'][:5])
print("--------------------------------")
print(data['eval'][:5])
print("--------------------------------")
print(data['piece_count'][:5])
