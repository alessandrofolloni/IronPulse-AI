import os
import sys
import django
import numpy as np
import json
import wandb
import argparse
from datetime import datetime

# Initialize Django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gymapp.settings')
django.setup()

from core.models import WorkoutSession, Exercise, AIModelMetadata
from core.ai.engine import PulseMindNet
from core.ai.trainer import PulseMindTrainer, prepare_workout_data

def train_pipeline(use_wandb=False, project_name="IronPulse-AI"):
    print("🚀 Starting PulseMind AI End-to-End Pipeline...")
    
    # 1. Data Preparation
    sessions = WorkoutSession.objects.all()
    exercises = Exercise.objects.all()
    
    if sessions.count() < 2: # Lowered for testing, usually need more
        print("❌ Error: Insufficient data in DB. Generate more workouts first.")
        # Create dummy data for the purpose of the script demo if needed
        X, y = np.random.rand(100, 18), np.zeros((100, exercises.count() or 10))
        for i in range(100): y[i, np.random.randint(0, y.shape[1])] = 1
    else:
        X, y = prepare_workout_data(sessions, exercises)

    # 2. Train/Val/Test Split (80/10/10)
    indices = np.random.permutation(X.shape[0])
    train_idx, val_idx = int(0.8 * X.shape[0]), int(0.9 * X.shape[0])
    
    X_train, X_val, X_test = X[indices[:train_idx]], X[indices[train_idx:val_idx]], X[indices[val_idx:]]
    y_train, y_val, y_test = y[indices[:train_idx]], y[indices[train_idx:val_idx]], y[indices[val_idx:]]

    # 3. Model & Hyperparameters
    config = {
        "learning_rate": 0.01,
        "epochs": 500,
        "input_size": X.shape[1],
        "hidden_layers": [128, 64],
        "output_size": y.shape[1],
        "architecture": "MLP-Softmax (Manual Backprop)",
        "loss_function": "Cross-Entropy"
    }

    if use_wandb:
        wandb.init(project=project_name, config=config)

    model = PulseMindNet(config["input_size"], config["hidden_layers"], config["output_size"])
    trainer = PulseMindTrainer(model)
    trainer.learning_rate = config["learning_rate"]

    # 4. Training Loop
    print(f"🧠 Training on {X_train.shape[0]} samples...")
    for epoch in range(config["epochs"]):
        loss = trainer.train_step(X_train, y_train)
        
        # Validation
        val_pred = model.forward(X_val)
        val_loss = -np.sum(y_val * np.log(val_pred + 1e-9)) / X_val.shape[0]
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
            if use_wandb:
                wandb.log({"train_loss": loss, "val_loss": val_loss, "epoch": epoch})

    # 5. Testing
    test_pred = model.forward(X_test)
    accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"✅ Training Complete. Test Accuracy: {accuracy*100:.2f}%")
    
    if use_wandb:
        wandb.log({"test_accuracy": accuracy})
        wandb.finish()

    # 6. Export Weights
    params = model.get_parameters()
    weight_path = "core/ai/weights/pulsemind_latest.json"
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    
    with open(weight_path, 'w') as f:
        json.dump(params, f)
    
    # 7. Update Metadata in DB
    metadata = AIModelMetadata.objects.first() or AIModelMetadata.objects.create()
    metadata.last_trained = datetime.now()
    metadata.accuracy = accuracy * 100
    metadata.weights_info = params
    metadata.save()
    
    print(f"📂 Weights exported to {weight_path} and synced to Database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()
    
    train_pipeline(use_wandb=args.wandb)
