import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Verifying imports...")

try:
    import data_loader
    print("data_loader: OK")
except Exception as e:
    print(f"data_loader: FAILED ({e})")

try:
    import prediction_module
    print("prediction_module: OK")
except Exception as e:
    print(f"prediction_module: FAILED ({e})")

try:
    import ontology_utils
    print("ontology_utils: OK")
except Exception as e:
    print(f"ontology_utils: FAILED ({e})")

try:
    import evaluation_metrics
    print("evaluation_metrics: OK")
except Exception as e:
    print(f"evaluation_metrics: FAILED ({e})")

try:
    import main_benchmark
    print("main_benchmark: OK")
except Exception as e:
    print(f"main_benchmark: FAILED ({e})")

print("\nVerification Complete.")
