"""
===================================================================================
HEARTHSTONE AI MODEL ARCHITECTURE
===================================================================================

This file defines the neural network that learns to recognize sign language gestures.
Below is a guide to common AI building blocks you can use when modifying this model.

===================================================================================
COMMON LAYER TYPES & WHAT THEY DO
===================================================================================

1. Conv1D (Convolutional Layer for Time Series)
   ANALOGY: A sliding window that looks for patterns over time
   - kernel_size: How many frames to look at once (like a 5-frame sliding window)
   - filters: How many different patterns to learn (64 means "learn 64 different motion patterns")
   - USE WHEN: You want to detect specific movements (like "hand moving up" or "fingers closing")
   
2. LSTM (Long Short-Term Memory)
   ANALOGY: A note-taker that remembers important events from earlier in the video
   - units: How much "memory capacity" it has (128 = can remember 128 different things)
   - return_sequences=True: Keeps notes at every frame (not just at the end)
   - Bidirectional: Reads the video forwards AND backwards (like proofreading)
   - USE WHEN: The sign depends on what happened earlier (e.g., "goodbye" wave vs "hello" wave)

3. Dense (Fully Connected Layer)
   ANALOGY: A decision-maker that combines all information to make a conclusion
   - units: How many intermediate conclusions it can draw
   - Final Dense layer has num_classes outputs = one score per possible sign
   - USE WHEN: At the end, to make the final classification

4. MaxPooling1D
   ANALOGY: Summarizing by keeping only the most important frame from each group
   - pool_size=2: Combines every 2 frames into 1 (reduces 256 frames → 128 frames)
   - USE WHEN: You have too many frames and want to speed up training
   - TRADEOFF: Faster training but loses some detail

5. Dropout
   ANALOGY: Randomly blindfolding some neurons during training so they don't become lazy
   - rate=0.3: Turns off 30% of neurons randomly during each training step
   - USE WHEN: Your model memorizes the training data instead of learning patterns (overfitting)
   - RULE OF THUMB: Start with 0.3-0.5 for most layers

6. BatchNormalization
   ANALOGY: Keeps all the numbers in a reasonable range so training doesn't explode
   - Automatically adjusts values so they're not too big or too small
   - USE WHEN: After Conv1D or Dense layers to stabilize training
   - BENEFIT: Trains faster and more reliably

===================================================================================
ACTIVATION FUNCTIONS (How neurons "fire")
===================================================================================

- relu (Rectified Linear Unit): "If positive, pass it through; if negative, set to 0"
  ANALOGY: A one-way valve that only lets positive signals through
  USE: Almost everywhere except the output layer

- softmax: "Convert scores into probabilities that sum to 100%"
  ANALOGY: Dividing a pizza into slices—bigger scores get bigger slices
  USE: Only in the final layer for classification (which sign is it?)

- sigmoid: "Squash any number to be between 0 and 1"
  USE: For binary decisions (yes/no questions)

===================================================================================
COMMON MODIFICATIONS YOU MIGHT TRY
===================================================================================

TO MAKE THE MODEL LEARN MORE COMPLEX PATTERNS:
- Increase filter counts (64 → 128, 128 → 256)
- Add more Conv1D or LSTM layers
- Increase LSTM units (64 → 128)

TO MAKE TRAINING FASTER:
- Add more MaxPooling layers
- Reduce filter counts (256 → 128)
- Reduce LSTM units

TO FIX OVERFITTING (memorizing instead of learning):
- Increase Dropout rates (0.3 → 0.5)
- Add more Dropout layers
- Reduce model size (fewer filters/units)

TO FIX UNDERFITTING (not learning enough):
- Decrease Dropout rates (0.5 → 0.3)
- Add more layers
- Increase filter counts/LSTM units

===================================================================================
ARCHITECTURE EXPLANATION (Current Model)
===================================================================================

Input: (256 frames, 2 hands, 20 landmarks per hand, x/y/z coordinates)
       = A video of someone signing, broken into 256 snapshots

Step 1: Reshape to (256, 120) 
        Flatten the hand data so Conv1D can process it over time

Step 2: Conv1D layers with pooling
        Learn motion patterns (like "hand moves up") and summarize over time
        256 frames → 128 → 64 → 32 frames

Step 3: Bidirectional LSTM
        Remember important context from earlier/later in the video
        (e.g., "this gesture started with hands together")

Step 4: Dense layers
        Combine all learned patterns to decide which of 2208 signs this is

Output: 2208 probabilities (one per sign)
        Whichever has the highest probability is the AI's guess

===================================================================================
"""

from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import *

num_classes = 101  # Total number of different ASL signs the AI can recognize
max_frames = 256  # Maximum video length (in frames)

# =============================================================================
# ⚠️ CONTRIBUTORS: Only make edits below this line
# Add, remove or change the parameters of layers to try improve the accuracy.
# Earlier layers tend to have a greater impact on learning due to the butterfly
# effect, so start with the first few layers for coarse adjustments, then fine-tune later layers.
# =============================================================================

inputs = Input(shape=(max_frames, 2, 20, 3))

# ── PPP: Processing Prior to Pancaking ───────────────────────────────────────
# Reshape so each landmark is its own (x,y,z) input vector
x = layers.Reshape((max_frames, 2 * 20, 3))(inputs)  # → (max_frames, 40, 3)

# Learn a per-landmark spatial embedding — Dense has no activation maps to store
# so backprop memory is O(params) not O(frames × spatial_dims)
x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(
    x
)  # → (max_frames, 40, 32)

# Flatten into per-frame feature vector
x = layers.Reshape((max_frames, 40 * 32))(x)  # → (max_frames, 1280)

# ── First convolutional block: Learn short-term motion patterns ───────────────
x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)  # max_frames → max_frames/2
x = layers.Dropout(0.2)(x)  # Reduced from 0.3 — less signal killed early on

# ── Second convolutional block: Learn more complex motion combinations ────────
x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)  # → max_frames/4
x = layers.Dropout(0.2)(x)  # Reduced from 0.3

# ── Third convolutional block: Learn high-level gesture components ────────────
x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)  # → max_frames/8
x = layers.Dropout(0.3)(x)  # Reduced from 0.4

# ── Bidirectional LSTM: Remember context from throughout the video ────────────
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dropout(0.25)(x)  # Reduced from 0.4 — critical for LSTM gradient flow

# ── Second LSTM: Final temporal understanding (outputs final state only) ──────
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dropout(0.25)(x)  # Reduced from 0.4

# ── Dense classification head ─────────────────────────────────────────────────
# 512 units gives the softmax more representational room for a large class pool
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)  # Reduced from 0.3
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)  # Kept light to protect against word-specific overfit

# ── Output layer: One probability per sign (2208 possible signs) ──────────────
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Compile the model: Set up how it learns
model.compile(
    optimizer=Adam(learning_rate=4e-4),  # LR decay handled externally
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
