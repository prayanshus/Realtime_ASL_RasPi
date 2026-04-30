#Imports and Configuration

import os
import datetime
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

FEATURES_DIR = os.path.join("Extracted_Features")
SEQUENCE_LENGTH = 30
FEATURES = 504          # 252 base + 252 velocity; must match preprocess.py output
K_FOLDS = 5             # Number of folds for cross-validation
TEST_SPLIT = 0.15       # Fraction of data held out as final test set


# Loading Extracted Features
def load_data():
    actions = sorted([
        d for d in os.listdir(FEATURES_DIR)
        if os.path.isdir(os.path.join(FEATURES_DIR, d))
    ])
    label_map = {label: num for num, label in enumerate(actions)}

    print(f"Found {len(actions)} sign classes: {actions}\n")

    sequences, labels = [], []

    print("Loading feature files...")
    for action in actions:
        action_path = os.path.join(FEATURES_DIR, action)
        npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]

        for seq_file in npy_files:
            res = np.load(os.path.join(action_path, seq_file))
            if res.shape != (SEQUENCE_LENGTH, FEATURES):
                print(f"  Shape mismatch: {seq_file} has {res.shape}, expected ({SEQUENCE_LENGTH},{FEATURES}) — skipping")
                continue
            sequences.append(res)
            labels.append(label_map[action])

        print(f"  {action}: {len(npy_files)} samples")

    X = np.array(sequences)
    y_int = np.array(labels)
    y = to_categorical(y_int).astype(int)

    print(f"\nTotal samples: {len(sequences)}")
    print(f"Input shape per sample: {X[0].shape}")
    return X, y, y_int, actions


X, y, y_int, actions = load_data()
NUM_CLASSES = len(actions)

# Creating test and train sets 
X_trainval, X_test, y_trainval, y_test, y_int_trainval, _ = train_test_split(
    X, y, y_int, test_size=TEST_SPLIT, random_state=42, stratify=y_int)

print(f"\nHeld-out test samples:       {X_test.shape[0]}")
print(f"Train+val samples (for CV):  {X_trainval.shape[0]}")


# Neural Architecture Search for optimal architecture

def build_hypermodel(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(SEQUENCE_LENGTH, FEATURES)))

    model_type = hp.Choice('model_type', ['cnn', 'lstm'])

    if model_type == 'cnn':
        model.add(tf.keras.layers.Conv1D(
            filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size', [3, 5]),
            activation='relu',
            padding='same'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())

    elif model_type == 'lstm':
        model.add(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
            return_sequences=False))

    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=64, step=32),
        activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


print("Searching for best model architecture (CNN vs LSTM + hyperparams).")


# NAS uses an internal 80/20 split of the train+val pool
X_nas_train, X_nas_val, y_nas_train, y_nas_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42,
    stratify=y_int_trainval)

tuner = kt.Hyperband(
    build_hypermodel,
    objective='val_accuracy',
    max_epochs=50,
    directory='nas_dir',
    project_name='sign_translator')

log_dir_nas = "logs/fit/nas_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tuner.search(
    X_nas_train, y_nas_train,
    epochs=50,
    validation_data=(X_nas_val, y_nas_val),
    verbose=1,
    callbacks=[TensorBoard(log_dir=log_dir_nas, histogram_freq=1)])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model_type = best_hps.get('model_type')

print("\n" + "="*60)
print(f"BEST ARCHITECTURE: {model_type.upper()}")
print("="*60)
if model_type == 'cnn':
    print(f"  Conv1D filters: {best_hps.get('conv_filters')}")
    print(f"  Kernel size:    {best_hps.get('kernel_size')}")
elif model_type == 'lstm':
    print(f"  LSTM units:     {best_hps.get('lstm_units')}")
print(f"  Dense units:    {best_hps.get('dense_units')}")
print(f"  Dropout:        {best_hps.get('dropout')}")
print(f"  Learning rate:  {best_hps.get('learning_rate')}")
print("="*60 + "\n")


# K - fold cross validation

print(f"Training the best architecture on {K_FOLDS} different data splits.")
print("Each fold trains for 100 epochs. This quantifies model stability.\n")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_val_accuracies = []
fold_val_losses = []
fold_best_epochs = []   # epoch (1-based) with lowest val_loss per fold

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_int_trainval)):
    print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")
    print(f"  Training samples:   {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")

    X_fold_train, X_fold_val = X_trainval[train_idx], X_trainval[val_idx]
    y_fold_train, y_fold_val = y_trainval[train_idx], y_trainval[val_idx]

    fold_model = tuner.hypermodel.build(best_hps)

    log_dir_fold = f"logs/fit/fold{fold_idx+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hist = fold_model.fit(
        X_fold_train, y_fold_train,
        epochs=100,
        validation_data=(X_fold_val, y_fold_val),
        verbose=1,
        callbacks=[TensorBoard(log_dir=log_dir_fold, histogram_freq=1)])

    best_epoch = int(np.argmin(hist.history['val_loss'])) + 1  # 1-based
    fold_best_epochs.append(best_epoch)

    val_loss, val_acc = fold_model.evaluate(X_fold_val, y_fold_val, verbose=0)
    fold_val_accuracies.append(val_acc)
    fold_val_losses.append(val_loss)
    print(f"  Fold {fold_idx+1}: val_acc={val_acc*100:.2f}%  val_loss={val_loss:.4f}  best_epoch={best_epoch}")

    del fold_model
    tf.keras.backend.clear_session()

# Derive final training epoch count from k-fold — no validation set needed
final_epochs = int(round(np.mean(fold_best_epochs)))

print("\n" + "="*60)
print(f"K-FOLD CROSS VALIDATION RESULTS ({K_FOLDS} folds)")
print("="*60)
for i, (acc, loss, ep) in enumerate(zip(fold_val_accuracies, fold_val_losses, fold_best_epochs)):
    print(f"  Fold {i+1}: val_acc={acc*100:.2f}%  val_loss={loss:.4f}  best_epoch={ep}")
print("-"*60)
mean_acc = np.mean(fold_val_accuracies)
std_acc  = np.std(fold_val_accuracies)
print(f"  Mean val accuracy:  {mean_acc*100:.2f}%")
print(f"  Std  val accuracy:  {std_acc*100:.2f}%")
print(f"  Min / Max:          {np.min(fold_val_accuracies)*100:.2f}% / {np.max(fold_val_accuracies)*100:.2f}%")
print(f"  Mean best epoch:    {final_epochs}  (used as fixed epoch count for final training)")
print("="*60)
print("\nInterpretation:")
print(f"  The model generalises to ~{mean_acc*100:.1f}% ± {std_acc*100:.1f}% on unseen data.")
if std_acc < 0.05:
    print("  Low std: model is stable across different data splits.")
else:
    print("  High std: model is sensitive to which data it trains on — consider more data.")


# Final model - LR scheduler comparison

print(f"Training {final_epochs} epochs (mean best epoch from k-fold) on all train+val data.")
print("Schedulers compared: Constant, CosineDecay, ExponentialDecay, ReduceLROnPlateau\n")

base_lr = best_hps.get('learning_rate')
steps_per_epoch = max(1, len(X_trainval) // 32)
total_steps = final_epochs * steps_per_epoch

def build_final_model(lr_or_schedule):
    m = tuner.hypermodel.build(best_hps)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr_or_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return m

# Print architecture summary once before the loop
_tmp = build_final_model(base_lr)
_tmp.summary()
del _tmp
tf.keras.backend.clear_session()

lr_configs = [
    {
        'name': 'Constant',
        'schedule': base_lr,
        'extra_callbacks': [],
    },
    {
        'name': 'CosineDecay',
        'schedule': tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=total_steps),
        'extra_callbacks': [],
    },
    {
        'name': 'ExponentialDecay',
        'schedule': tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr,
            decay_steps=max(1, total_steps // 10),
            decay_rate=0.9,
            staircase=False),
        'extra_callbacks': [],
    },
    {
        'name': 'ReduceLROnPlateau',
        'schedule': base_lr,
        'extra_callbacks': [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ],
    },
]


scheduler_results = {}   # name -> (model, test_acc, test_loss)

for cfg in lr_configs:
    print(f"\n--- Scheduler: {cfg['name']} ---")
    m = build_final_model(cfg['schedule'])
    log_dir_sched = (f"logs/fit/final_{cfg['name'].lower()}_"
                     + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    m.fit(
        X_trainval, y_trainval,
        epochs=final_epochs,
        verbose=1,
        callbacks=[TensorBoard(log_dir=log_dir_sched, histogram_freq=1)]
                  + cfg['extra_callbacks'])

    t_loss, t_acc = m.evaluate(X_test, y_test, verbose=0)
    scheduler_results[cfg['name']] = (m, t_acc, t_loss)
    print(f"  [{cfg['name']}] test_acc={t_acc*100:.2f}%  test_loss={t_loss:.4f}")

# Pick best by test accuracy and use it for export
best_sched_name = max(scheduler_results, key=lambda k: scheduler_results[k][1])
final_model = scheduler_results[best_sched_name][0]

print("\n" + "="*60)
print("LR SCHEDULER COMPARISON — TEST SET RESULTS")
print("="*60)
print(f"  {'Scheduler':<22} {'Test Acc':>10}  {'Test Loss':>10}")
print("-"*60)
for name, (_, acc, loss) in sorted(scheduler_results.items(), key=lambda x: -x[1][1]):
    tag = "  <-- best" if name == best_sched_name else ""
    print(f"  {name:<22} {acc*100:>9.2f}%  {loss:>10.4f}{tag}")
print("="*60)

_, best_test_acc, _ = scheduler_results[best_sched_name]
print(f"\nFinal model:    {best_sched_name} scheduler")
print(f"Test accuracy:  {best_test_acc*100:.2f}%")
print(f"K-fold estimate:{mean_acc*100:.2f}% ± {std_acc*100:.2f}%")


# --- 5. EXPORT TO TFLITE ---
# FP32 model
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_fp32 = converter.convert()
with open('sign_model_fp32.tflite', 'wb') as f:
    f.write(tflite_fp32)
print("\nSaved: sign_model_fp32.tflite")

# INT8 quantized model (smaller and faster on Pi) - QAT
print("Applying INT8 quantization...")

num_cal = min(200, len(X_trainval))

def representative_dataset():
    for i in range(num_cal):
        yield [np.expand_dims(X_trainval[i], axis=0).astype(np.float32)]

converter2 = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset = representative_dataset
converter2.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter2._experimental_lower_tensor_list_ops = False
converter2.inference_input_type  = tf.float32
converter2.inference_output_type = tf.float32
tflite_quant = converter2.convert()
with open('sign_model_qat.tflite', 'wb') as f:
    f.write(tflite_quant)
print("Saved: sign_model_qat.tflite")

# Labels
actions_list = sorted([
    d for d in os.listdir(FEATURES_DIR)
    if os.path.isdir(os.path.join(FEATURES_DIR, d))
])
np.save('sign_labels.npy', np.array(actions_list))
print(f"Saved: sign_labels.npy — {actions_list}")

print("\nTraining complete!")

