import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras import regularizers # type: ignore
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# === Output folders ===
output_base = "output"
subdirs = ["accuracy", "loss", "gradcam", "report", "roc_auc", "predictions", "samples", "confusion_matrix", "f1_score", "precision", "recall", "support"]
for subdir in subdirs:
    os.makedirs(os.path.join(output_base, subdir), exist_ok=True)

print("📁 Output folders created successfully:")
for subdir in subdirs:
    print(f"  ✅ {os.path.join(output_base, subdir)}")

# === Load Dataset ===
def load_data(data_dir):
    filepaths, labels = [], []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path): continue
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

train_df = load_data('dataset/Training')
test_df = load_data('dataset/Testing')
valid_df, test_df = train_test_split(test_df, train_size=0.5, random_state=42)

print(f"📊 Total Images: {len(train_df) + len(valid_df) + len(test_df)}")
print(f"🧠 Training:\n{train_df['labels'].value_counts()}\n")
print(f"🧪 Validation:\n{valid_df['labels'].value_counts()}\n")
print(f"🧾 Testing:\n{test_df['labels'].value_counts()}\n")

# === Image Generators ===
img_size = (224, 224)
batch_size = 8

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels',
    target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=True
)

valid_gen = test_datagen.flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels',
    target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=False
)

test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels',
    target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=False
)

num_classes = len(train_gen.class_indices)
label_map = dict((v, k) for k, v in train_gen.class_indices.items())

# === Model ===
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True  # Fine-tune entire model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=2, factor=0.3, verbose=1),
    ModelCheckpoint("model/brain_tumor_model.keras", save_best_only=True, verbose=1)
]

# === Train Model ===
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30,
    callbacks=callbacks
)

# === Evaluate ===
loss, acc = model.evaluate(test_gen)
print(f"✅ Accuracy: {acc * 100:.2f}%")

# === ROC Curve ===
y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(to_categorical(y_true, num_classes)[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{label_map[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("output/roc_auc/roc_auc.png")
plt.close()
print("📈 ROC curve saved to output/roc_auc/roc_auc.png")

# === Classification Report ===
report = classification_report(y_true, np.argmax(y_pred, axis=1), target_names=test_gen.class_indices.keys())
with open("output/report/classification_report.txt", "w") as f:
    f.write(report)
print("📝 Classification report saved to output/report/classification_report.txt")

# === Accuracy & Loss Plots ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend()
plt.savefig("output/accuracy/accuracy_plot.png")
plt.close()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend()

plt.savefig("output/loss/loss_plot.png")
plt.close()

print("📊 Accuracy & Loss plots saved")

# === Sample Images ===
for cls in train_df['labels'].unique():
    sample = train_df[train_df['labels'] == cls].iloc[0]['filepaths']
    img = cv2.imread(sample)
    cv2.imwrite(f"output/samples/sample_{cls}.jpg", img)

# === Predictions on Input Images ===
input_dir = "dataset/input"
pred_dir = "output/predictions"
os.makedirs(pred_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    path = os.path.join(input_dir, img_file)
    img = cv2.imread(path)
    resized = cv2.resize(img, img_size)
    pred = model.predict(np.expand_dims(preprocess_input(resized), axis=0))
    pred_class = label_map[np.argmax(pred)]

    label = f"{pred_class}"
    color = (0, 255, 0) if pred_class == "no_tumor" else (0, 0, 255)
    status = "✅" if pred_class == "no_tumor" else "❌"

    cv2.putText(img, f"{status} {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imwrite(os.path.join(pred_dir, img_file), img)

print("🧾 All predictions saved to output/predictions/")

# === Grad-CAM ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy(), tf.argmax(predictions[0]).numpy()

gradcam_dir = "output/gradcam"
for img_file in os.listdir(input_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    path = os.path.join(input_dir, img_file)
    orig = cv2.imread(path)
    img = cv2.resize(orig, img_size)
    img_array = np.expand_dims(preprocess_input(img), axis=0)
    heatmap, _ = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(os.path.join(gradcam_dir, f"gradcam_{img_file}"), superimposed)

print("🔥 Grad-CAM images saved to output/gradcam/")

# === Confusion Matrix ===
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(test_gen.class_indices.keys()),
            yticklabels=list(test_gen.class_indices.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/confusion_matrix/confusion_matrix.png")
plt.close()

print("🔍 Confusion matrix saved to output/confusion_matrix/")
# === Classification Metrics ===
# Compute per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred_classes, labels=list(range(num_classes)), zero_division=0
)

# Prepare DataFrame
metric_df = pd.DataFrame({
    'Class': [label_map[i] for i in range(num_classes)],
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Support': support
})

# Save each metric to its respective folder
metric_df[['Class', 'Precision']].to_csv('output/precision/precision.csv', index=False)
metric_df[['Class', 'Recall']].to_csv('output/recall/recall.csv', index=False)
metric_df[['Class', 'F1 Score']].to_csv('output/f1_score/f1_score.csv', index=False)
metric_df[['Class', 'Support']].to_csv('output/support/support.csv', index=False)

print("📈 Precision, Recall, F1 Score, and Support saved to respective folders.")
# === Save full metrics as single file (optional)
metric_df.to_csv("output/report/full_classification_metrics.csv", index=False)
