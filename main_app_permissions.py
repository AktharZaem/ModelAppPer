#!/usr/bin/env python3

import os
import sys
import traceback
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def check_dependencies():
    """Check if required modules can be imported"""
    missing_modules = []

    try:
        from app_permissions_model_trainer import AppPermissionsModelTrainer
        print("✅ app_permissions_model_trainer module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import app_permissions_model_trainer: {e}")
        missing_modules.append("app_permissions_model_trainer")

    try:
        from app_permissions_user_tester import AppPermissionsTester
        print("✅ app_permissions_user_tester module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import app_permissions_user_tester: {e}")
        missing_modules.append("app_permissions_user_tester")

    try:
        from app_permissions_educational_resources import AppPermissionsEducationalManager
        print("✅ app_permissions_educational_resources module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import app_permissions_educational_resources: {e}")
        missing_modules.append("app_permissions_educational_resources")

    return missing_modules


def check_json_structure():
    """Check the structure of the JSON answer sheet"""
    if not os.path.exists('answer_sheetappper.json'):
        print("❌ answer_sheetappper.json not found")
        return False

    try:
        with open('answer_sheetappper.json', 'r') as f:
            data = json.load(f)

        print(f"\nJSON Structure Analysis:")
        print(f"Type: {type(data)}")
        print(
            f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")

        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  '{key}': {type(value)}")
                if isinstance(value, dict):
                    print(f"    Subkeys: {list(value.keys())[:5]}...")
                elif isinstance(value, list):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First item: {type(value[0])}")

        return True
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return False


def check_files():
    """Check if required files exist"""
    files_to_check = [
        'mobile_app_permission.csv',
        'answer_sheetappper.json'
    ]

    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print("\nChecking required files:")

    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} (size: {size} bytes)")
        else:
            print(f"❌ {file} not found")


def main():
    print("Mobile App Permissions Security Awareness Assessment System")
    print("=" * 60)

    # Check dependencies first
    print("\nChecking dependencies...")
    missing_modules = check_dependencies()

    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please ensure all required Python files are in the same directory.")
        return

    # Import modules after checking
    try:
        from app_permissions_model_trainer import AppPermissionsModelTrainer
        from app_permissions_user_tester import AppPermissionsTester
        from app_permissions_educational_resources import AppPermissionsEducationalManager
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return

    # Initialize educational manager
    education_manager = AppPermissionsEducationalManager()
    last_quiz_score = None
    weak_areas = []

    while True:
        print("\nSelect an option:")
        print("1. Train ML Model")
        print("2. Take Assessment Quiz")
        print("3. Educational Resources & Learning")
        print("4. Check System Status")
        print("5. Check JSON Structure")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            print("\n--- Training ML Model for App Permissions ---")
            check_files()

            if not check_json_structure():
                print("❌ JSON structure check failed!")
                continue

            if not os.path.exists('mobile_app_permission.csv'):
                print("Error: mobile_app_permission.csv not found!")
                print("Please ensure the dataset file is in the current directory.")
                continue

            if not os.path.exists('answer_sheetappper.json'):
                print("Error: answer_sheetappper.json not found!")
                print("Please ensure the answer sheet file is in the current directory.")
                continue

            try:
                print("Initializing trainer...")
                trainer = AppPermissionsModelTrainer(
                    dataset_path='mobile_app_permission.csv',
                    answer_sheet_path='answer_sheetappper.json'
                )

                print("Starting model training...")
                model, accuracy = trainer.train_model()
                print(f"\n✅ Model training completed successfully!")
                print(f"Model accuracy: {accuracy:.2f}")

            except Exception as e:
                print(f"❌ Error during model training: {e}")
                traceback.print_exc()

        elif choice == '2':
            print("\n--- Mobile App Permissions Security Assessment ---")

            if not os.path.exists('app_permissions_model.pkl'):
                print("❌ Trained model not found!")
                print("Please train the model first (option 1).")
                continue

            try:
                print("Initializing tester...")
                tester = AppPermissionsTester()
                result = tester.run_assessment()

                if isinstance(result, dict):
                    last_quiz_score = result.get('score', 0)
                    weak_areas = result.get('weak_areas', [])
                    print(
                        f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")
                elif isinstance(result, (int, float)):
                    last_quiz_score = result
                    print(
                        f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")

            except Exception as e:
                print(f"❌ Error during assessment: {e}")
                traceback.print_exc()

        elif choice == '3':
            print("\n--- Educational Resources & Learning ---")
            try:
                if last_quiz_score is not None:
                    print(f"📊 Using your recent assessment results...")
                    education_manager.run_educational_session(
                        last_quiz_score, weak_areas)
                else:
                    print("📚 No recent assessment found. Showing general resources...")
                    education_manager.run_educational_session()

                print(f"\n🔄 Additional Options:")
                print("A. View resources for different knowledge level")
                print("B. Get quick security tips")
                print("C. Return to main menu")

                sub_choice = input(
                    "\nEnter your choice (A/B/C): ").strip().upper()

                if sub_choice == 'A':
                    education_manager.run_educational_session()
                elif sub_choice == 'B':
                    tips = education_manager.get_interactive_tips()
                    print(f"\n💡 APP PERMISSIONS SECURITY TIPS:")
                    for i, tip in enumerate(tips, 1):
                        print(f"{i}. {tip}")

            except Exception as e:
                print(f"❌ Error accessing educational resources: {e}")
                traceback.print_exc()

        elif choice == '4':
            print("\n--- System Status ---")
            check_files()

            if os.path.exists('app_permissions_model.pkl'):
                size = os.path.getsize('app_permissions_model.pkl')
                print(f"✅ app_permissions_model.pkl (size: {size} bytes)")
            else:
                print("❌ app_permissions_model.pkl not found")

        elif choice == '5':
            print("\n--- JSON Structure Check ---")
            check_json_structure()

        elif choice == '6':
            print(
                "\nThank you for using the Mobile App Permissions Security Assessment System!")
            print("Keep learning and stay secure! 🔒📱")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
