import os
import matplotlib.pyplot as plt

# Training results from the completed training
epochs = [1, 2, 3, 4]

train_loss = [0.5053, 0.4316, 0.4087, 0.3852]
val_loss = [0.6485, 5.4762, 0.8025, 1.1403]

train_dice = [0.7136, 0.7392, 0.7596, 0.7913]
val_dice = [0.0462, 0.6039, 0.6769, 0.6060]

train_iou = [0.5592, 0.5924, 0.6166, 0.6581]
val_iou = [0.0238, 0.4349, 0.5154, 0.4369]

os.makedirs("outputs", exist_ok=True)

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/training_history.png", dpi=300)
plt.close()

print("training_history.png created successfully!")