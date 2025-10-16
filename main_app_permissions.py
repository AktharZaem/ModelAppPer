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


def view_assessment_database():
    """View and analyze the assessment database"""
    database_file = 'app_permissions_assessment_database.json'

    try:
        with open(database_file, 'r', encoding='utf-8') as f:
            database = json.load(f)

        assessments = database.get('assessments', [])

        if not assessments:
            print("📊 No assessments found in database.")
            return

        print(f"\n📊 ASSESSMENT DATABASE SUMMARY")
        print("=" * 60)
        print(f"Total Assessments: {len(assessments)}")
        print(
            f"Database Created: {database.get('metadata', {}).get('created', 'Unknown')}")

        # Statistics
        scores = [a['percentage'] for a in assessments]
        levels = [a['overall_knowledge_level'] for a in assessments]

        print(f"\nSCORE STATISTICS:")
        print(f"Average Score: {sum(scores)/len(scores):.1f}%")
        print(f"Highest Score: {max(scores):.1f}%")
        print(f"Lowest Score: {min(scores):.1f}%")

        print(f"\nKNOWLEDGE LEVEL DISTRIBUTION:")
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        for level, count in sorted(level_counts.items()):
            print(f"  {level}: {count} users")

        print(f"\nRECENT ASSESSMENTS:")
        print("-" * 40)

        # Show last 5 assessments
        recent_assessments = sorted(
            assessments, key=lambda x: x['timestamp'], reverse=True)[:5]

        for i, assessment in enumerate(recent_assessments, 1):
            email_or_name = assessment.get(
                'email', assessment.get('name', 'Unknown'))
            print(f"{i}. {email_or_name} ({assessment['timestamp']})")
            print(
                f"   Score: {assessment['total_score']}/100 ({assessment['percentage']:.1f}%)")
            print(f"   Level: {assessment['overall_knowledge_level']}")
            print(
                f"   Profile: {assessment['gender']}, {assessment['education_level']}, {assessment['proficiency']}")
            print()

    except FileNotFoundError:
        print("📊 No assessment database found. Take an assessment first!")
    except Exception as e:
        print(f"❌ Error reading database: {e}")


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
        print("2. Take Assessment Quiz (with Profile Setup)")
        print("3. Educational Resources & Learning")
        print("4. Check System Status")
        print("5. Check JSON Structure")
        print("6. Check Explanation Bank Coverage")
        print("7. View Assessment Database")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ").strip()

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
                # Explicitly pass the assessment results JSON so trainer combines CSV + JSON
                trainer = AppPermissionsModelTrainer(
                    dataset_path='mobile_app_permission.csv',
                    answer_sheet_path='answer_sheetappper.json',
                    assessment_results_path='app_permissions_assessment_results.json'
                )

                print(
                    "🗂️ Trainer will use: mobile_app_permission.csv + app_permissions_assessment_results.json (if available)")

                print("Starting model training...")
                model, accuracy = trainer.train_model()
                print(f"\n✅ Model training completed successfully!")
                print(f"Model accuracy: {accuracy:.2f}")

            except Exception as e:
                print(f"❌ Error during model training: {e}")
                traceback.print_exc()

        elif choice == '2':
            print("\n--- Mobile App Permissions Security Assessment ---")
            print("ℹ️  This assessment includes:")
            print("   • Profile setup (Gender, Proficiency, Education)")
            print("   • 10 questions about app permissions")
            print("   • Personalized feedback based on your profile")

            if not os.path.exists('app_permissions_model.pkl'):
                print("❌ Trained model not found!")
                print("Please train the model first (option 1).")
                continue

            # Prompt for email and check existing assessments
            email = input("Enter your email address: ").strip().lower()
            if not email:
                print("❌ Email is required. Returning to menu.")
                continue

            database_file = 'app_permissions_assessment_database.json'
            existing_assessment = None
            if os.path.exists(database_file):
                try:
                    with open(database_file, 'r', encoding='utf-8') as f:
                        database = json.load(f)
                    assessments = database.get('assessments', [])
                    # Filter by email and sort by timestamp descending to get latest
                    user_assessments = [a for a in assessments if a.get(
                        'email', '').lower() == email]
                    if user_assessments:
                        user_assessments.sort(
                            key=lambda x: x['timestamp'], reverse=True)
                        existing_assessment = user_assessments[0]  # Latest one
                except Exception as e:
                    print(f"❌ Error checking database: {e}")

            # Initialize tester
            try:
                print("\nInitializing assessment...")
                tester = AppPermissionsTester()
            except Exception as e:
                print(f"❌ Error initializing tester: {e}")
                traceback.print_exc()
                continue

            if existing_assessment:
                print(
                    f"\n👋 Hi! You're already enrolled with our assessment using {email}.")
                print("📊 Here's a summary of your last assessment:")
                print(f"   • Timestamp: {existing_assessment['timestamp']}")
                print(
                    f"   • Score: {existing_assessment['total_score']}/100 ({existing_assessment['percentage']:.1f}%)")
                print(
                    f"   • Knowledge Level: {existing_assessment['overall_knowledge_level']}")
                print(
                    f"   • Profile: {existing_assessment['gender']}, {existing_assessment['education_level']}, {existing_assessment['proficiency']}")
                # Print name and organization if available
                if existing_assessment.get('name'):
                    print(f"   • Name: {existing_assessment.get('name')}")
                if existing_assessment.get('organization'):
                    print(
                        f"   • Organization: {existing_assessment.get('organization')}")
                print(f"   • Category: {existing_assessment['category']}")
                print(
                    "\nAuto-filling your profile and proceeding to the pre-quiz options...")

                # Auto-fill profile from database
                tester.user_profile = {
                    'email': email,
                    'gender': existing_assessment['gender'],
                    'education': existing_assessment['education_level'],
                    'proficiency': existing_assessment['proficiency']
                }
                # Add optional fields if present
                if existing_assessment.get('name'):
                    tester.user_profile['name'] = existing_assessment.get(
                        'name')
                if existing_assessment.get('organization'):
                    tester.user_profile['organization'] = existing_assessment.get(
                        'organization')
            else:
                print(
                    f"\n👋 Welcome, new user with email {email}! You can fill your profile now for a personalized experience.")
                # Offer to fill profile before showing pre-quiz options
                fill_choice = input(
                    "Would you like to fill your profile now? (y/n): ").strip().lower()
                if fill_choice == 'y':
                    try:
                        tester.collect_user_profile()
                    except Exception as e:
                        print(f"❌ Error collecting profile: {e}")

            # Pre-quiz menu (both new and existing users)
            result = None
            while True:
                print("\nPre-Quiz Options:")
                print("A. View Leaderboard")
                print("B. Enroll & Take Quiz")
                print("C. Return to main menu")

                pre_choice = input(
                    "\nEnter your choice (A/B/C): ").strip().upper()
                if pre_choice == 'A':
                    try:
                        # show_leaderboard implemented in tester (safe fallback if missing)
                        if hasattr(tester, 'show_leaderboard'):
                            # Automatically use logged-in user's organization if available
                            org_filter = None
                            if getattr(tester, 'user_profile', None):
                                org = tester.user_profile.get('organization')
                                if org:
                                    org_filter = org  # show only this organization's leaderboard

                            tester.show_leaderboard(
                                top_n=10, organization=org_filter)
                        else:
                            print("ℹ️ Leaderboard not available.")
                    except Exception as e:
                        print(f"❌ Error showing leaderboard: {e}")
                elif pre_choice == 'B':
                    # Proceed to the quiz (tester.run_assessment will prompt for profile if missing)
                    result = tester.run_assessment()
                    break
                elif pre_choice == 'C':
                    print("Returning to main menu.")
                    result = None
                    break
                else:
                    print("Invalid choice! Please enter A, B, or C.")

            # Handle result (same as before)
            if result is None:
                continue

            if isinstance(result, dict):
                last_quiz_score = result.get('score', 0)
                weak_areas = result.get('weak_areas', [])
                print(
                    f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")
            elif isinstance(result, (int, float)):
                last_quiz_score = result
                print(
                    f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")

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
            print("\n--- Explanation Bank Coverage Check ---")
            try:
                from check_explanations import main as check_main
                check_main()
            except Exception as e:
                print(f"❌ Error checking explanations: {e}")
                # Manual check as fallback
                try:
                    with open('ExplanationBankappper.json', 'r') as f:
                        explanations = json.load(f)
                    print(
                        f"✅ ExplanationBank loaded: {len(explanations)} explanations")

                    # Count by question
                    q_count = {}
                    for exp in explanations:
                        qid = exp.get('questionId', 'Unknown')
                        q_count[qid] = q_count.get(qid, 0) + 1

                    print("Question coverage:")
                    for qid, count in sorted(q_count.items()):
                        print(f"  {qid}: {count} explanations")

                except Exception as inner_e:
                    print(f"❌ Could not load ExplanationBank: {inner_e}")

        elif choice == '7':
            print("\n--- Assessment Database Viewer ---")
            view_assessment_database()

            print("\nDatabase Options:")
            print("A. Export database to CSV")
            print("B. Clear database")
            print("C. Return to main menu")

            sub_choice = input("\nEnter your choice (A/B/C): ").strip().upper()

            if sub_choice == 'A':
                try:
                    export_database_to_csv()
                except Exception as e:
                    print(f"❌ Error exporting: {e}")
            elif sub_choice == 'B':
                confirm = input(
                    "⚠️ Are you sure you want to clear all assessment data? (yes/no): ")
                if confirm.lower() == 'yes':
                    try:
                        os.remove('app_permissions_assessment_database.json')
                        print("✅ Database cleared successfully!")
                    except FileNotFoundError:
                        print("ℹ️ Database was already empty.")
                    except Exception as e:
                        print(f"❌ Error clearing database: {e}")

        elif choice == '8':
            print(
                "\nThank you for using the Mobile App Permissions Security Assessment System!")
            print("Keep learning and stay secure! 🔒📱")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, 4, 5, 6, 7, or 8.")


def export_database_to_csv():
    """Export assessment database to CSV format"""
    import csv
    import datetime

    try:
        with open('app_permissions_assessment_database.json', 'r', encoding='utf-8') as f:
            database = json.load(f)

        assessments = database.get('assessments', [])

        if not assessments:
            print("📊 No data to export.")
            return

        csv_filename = f"app_permissions_assessments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Timestamp', 'Email', 'Gender', 'Education_Level', 'Proficiency',
                'Total_Score', 'Percentage', 'Overall_Knowledge_Level', 'Category'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for assessment in assessments:
                email_or_name = assessment.get(
                    'email', assessment.get('name', ''))
                writer.writerow({
                    'Timestamp': assessment['timestamp'],
                    'Email': email_or_name,
                    'Gender': assessment['gender'],
                    'Education_Level': assessment['education_level'],
                    'Proficiency': assessment['proficiency'],
                    'Total_Score': assessment['total_score'],
                    'Percentage': assessment['percentage'],
                    'Overall_Knowledge_Level': assessment['overall_knowledge_level'],
                    'Category': assessment['category']
                })

        print(f"✅ Database exported to: {csv_filename}")

    except Exception as e:
        print(f"❌ Error exporting database: {e}")


def save_assessment_result(result):
    """Save the assessment result to the database and file"""
    database_file = 'app_permissions_assessment_database.json'
    results_file = 'app_permissions_assessment_results.json'

    try:
        # Load existing database
        if os.path.exists(database_file):
            with open(database_file, 'r', encoding='utf-8') as f:
                database = json.load(f)
        else:
            database = {"assessments": [], "metadata": {}}

        # Add new assessment result
        database['assessments'].append(result)

        # Save back to database file
        with open(database_file, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=4)

        print(f"✅ Assessment result saved to database.")

        # Also save to results file
        if os.path.exists(results_file):
            with open(results_file, 'r+', encoding='utf-8') as f:
                # Lock file for writing
                fcntl.flock(f, fcntl.LOCK_EX)

                # Load existing results
                try:
                    results_data = json.load(f)
                except json.JSONDecodeError:
                    results_data = {"assessments": []}

                # Append new result
                results_data['assessments'].append(result)

                # Move cursor to beginning of file
                f.seek(0)

                # Truncate file to current size
                f.truncate()

                # Write updated results
                json.dump(results_data, f, ensure_ascii=False, indent=4)

                # Unlock file
                fcntl.flock(f, fcntl.LOCK_UN)

            print(f"✅ Assessment result saved to results file.")
        else:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"assessments": [result]},
                          f, ensure_ascii=False, indent=4)
            print(f"✅ Assessment result saved to new results file.")

    except Exception as e:
        print(f"❌ Error saving assessment result: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Error saving assessment result: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
