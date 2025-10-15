import json
import joblib
import pandas as pd
import numpy as np
import os
import requests
import re
from app_permissions_knowledge_enhancer import AppPermissionsKnowledgeEnhancer


class AppPermissionsTester:
    def __init__(self):
        self.answer_sheet = None
        self.questions_data = None
        self.model = None
        self.feature_names = None
        self.enhancer = AppPermissionsKnowledgeEnhancer()
        self.user_profile = None
        self.explanation_bank = None
        self.load_components()

    def load_components(self):
        """Load trained model, answer sheet, and explanation bank"""
        try:
            # Load answer sheet and parse the nested structure
            with open('answer_sheetappper.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.answer_sheet = {}
            self.questions_data = []

            if 'questions' in data and isinstance(data['questions'], list):
                for q_item in data['questions']:
                    question_text = q_item.get('question')
                    options_dict = {}

                    for option in q_item.get('options', []):
                        options_dict[option.get('text')] = {
                            'weight': option.get('marks'),
                            'level': option.get('level')
                        }

                    if question_text:
                        self.answer_sheet[question_text] = options_dict
                        self.questions_data.append(q_item)

            # Load explanation bank
            try:
                with open('ExplanationBankappper.json', 'r', encoding='utf-8') as f:
                    self.explanation_bank = json.load(f)
                print(
                    f"✅ Loaded {len(self.explanation_bank)} explanations from ExplanationBank")
            except FileNotFoundError:
                print(
                    "⚠️ ExplanationBankappper.json not found. Using fallback explanations.")
                self.explanation_bank = []

            # Load trained model and feature names if available
            try:
                self.model = joblib.load('app_permissions_model.pkl')
                print("✅ Loaded trained model 'app_permissions_model.pkl'")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                self.model = None

            try:
                self.feature_names = joblib.load(
                    'app_permissions_feature_names.pkl')
            except Exception:
                self.feature_names = None

            print("App permissions components loaded successfully!")
            print(f"Loaded {len(self.questions_data)} questions for quiz")

        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run app_permissions_model_trainer.py first to train the model")

    def collect_user_profile(self):
        """Collect user profile information before starting the quiz"""
        print("\n=== User Profile Setup ===")
        print("Please provide some basic information to get personalized feedback.\n")

        # Collect Email (unique)
        print("1. Enter Your Email:")
        while True:
            email = input("Email: ").strip()
            if not email:
                print("Please enter a valid email!")
                continue
            # Basic email validation
            if '@' not in email or '.' not in email.split('@')[1]:
                print("Please enter a valid email address!")
                continue
            # Check uniqueness in database
            try:
                with open('app_permissions_assessment_database.json', 'r', encoding='utf-8') as f:
                    database = json.load(f)
                existing_emails = [assessment.get('email', assessment.get(
                    'name', '')).lower() for assessment in database.get('assessments', [])]
                if email.lower() in existing_emails:
                    print(
                        "This email is already registered. Please use a different email!")
                    continue
            except FileNotFoundError:
                pass  # No database yet, allow
            break

        # Collect Gender
        print("\n2. Select Your Gender:")
        print("   1. Male")
        print("   2. Female")

        while True:
            try:
                gender_choice = int(input("Enter your choice (1-2): "))
                if gender_choice == 1:
                    gender = "Male"
                    break
                elif gender_choice == 2:
                    gender = "Female"
                    break
                else:
                    print("Please enter 1 or 2!")
            except ValueError:
                print("Please enter a valid number!")

        # Collect Education Level
        print("\n3. Select Your Education Level:")
        print("   1. O/L (Ordinary Level)")
        print("   2. A/L (Advanced Level)")
        print("   3. HND (Higher National Diploma)")
        print("   4. Degree (University Degree)")

        while True:
            try:
                education_choice = int(input("Enter your choice (1-4): "))
                if education_choice == 1:
                    education = "O/L"
                    break
                elif education_choice == 2:
                    education = "A/L"
                    break
                elif education_choice == 3:
                    education = "HND"
                    break
                elif education_choice == 4:
                    education = "Degree"
                    break
                else:
                    print("Please enter 1, 2, 3, or 4!")
            except ValueError:
                print("Please enter a valid number!")

        # Collect Proficiency Level
        print("\n4. Select Your IT/Technology Proficiency:")
        print("   1. School Level (Basic computer/smartphone use)")
        print("   2. High Education Level (Advanced computer/technology skills)")

        while True:
            try:
                proficiency_choice = int(input("Enter your choice (1-2): "))
                if proficiency_choice == 1:
                    proficiency = "School"
                    break
                elif proficiency_choice == 2:
                    proficiency = "High Education"
                    break
                else:
                    print("Please enter 1 or 2!")
            except ValueError:
                print("Please enter a valid number!")

        self.user_profile = {
            "email": email,
            "gender": gender,
            "education": education,
            "proficiency": proficiency
        }

        print(
            f"\n✅ Profile saved: {email}, {gender}, {education}, {proficiency}")
        print("This information will be used to provide personalized explanations.\n")

        return self.user_profile

    def conduct_quiz(self):
        """Conduct interactive quiz with user"""
        if not self.questions_data:
            print("No questions loaded. Exiting quiz.")
            return {}, {}

        print("\n=== Mobile App Permissions Security Awareness Quiz ===")
        print("Please answer the following questions about mobile app permissions.\n")

        user_responses = {}
        user_scores = {}

        for i, q_item in enumerate(self.questions_data, 1):
            question = q_item.get('question', 'Unknown question')
            options = q_item.get('options', [])

            print(f"Question {i}: {question}")
            print("\nOptions:")

            # Display options
            for j, option in enumerate(options, 1):
                print(f"{j}. {option.get('text', '')}")

            # Get user input
            while True:
                try:
                    choice = int(
                        input(f"\nEnter your choice (1-{len(options)}): "))
                    if 1 <= choice <= len(options):
                        selected_option = options[choice - 1]
                        selected_answer = selected_option.get('text', '')

                        user_responses[question] = selected_answer

                        # Get score and level for this answer
                        user_scores[question] = {
                            'answer': selected_answer,
                            'score': selected_option.get('marks', 0),
                            'level': selected_option.get('level', 'basic')
                        }
                        break
                    else:
                        print("Please enter a valid choice!")
                except ValueError:
                    print("Please enter a valid number!")

            print("-" * 50)

        return user_responses, user_scores

    def calculate_results(self, user_scores):
        """Calculate overall results and recommendations"""
        if not user_scores:
            return 0, 0.0, 'Beginner'

        total_score = sum(score_info.get('score', 0)
                          for score_info in user_scores.values())
        max_possible_score = len(user_scores) * 10
        percentage = (total_score / max_possible_score) * \
            100 if max_possible_score > 0 else 0

        # Determine overall level
        if percentage >= 75:
            overall_level = 'Expert'
        elif percentage >= 50:
            overall_level = 'Intermediate'
        elif percentage >= 25:
            overall_level = 'Basic'
        else:
            overall_level = 'Beginner'

        return total_score, percentage, overall_level

    def get_explanation_from_bank(self, question_id, option_label, user_profile):
        """Get personalized explanation from ExplanationBank based on user profile"""
        if not self.explanation_bank:
            return self.get_detailed_explanation(question_id, "basic", "basic")

        # Normalize question ID (Q01 -> Q1, Q1 stays Q1)
        normalized_qid = question_id
        if isinstance(question_id, str) and question_id.startswith('Q'):
            match = re.match(r"Q0*(\d+)", question_id)
            if match:
                normalized_qid = f"Q{match.group(1)}"

        # Find matching explanation
        for explanation in self.explanation_bank:
            exp_qid = explanation.get("questionId", "")
            exp_option = explanation.get("option", "")
            exp_profile = explanation.get("profile", {})

            # Normalize explanation question ID too
            if isinstance(exp_qid, str) and exp_qid.startswith('Q'):
                match = re.match(r"Q0*(\d+)", exp_qid)
                if match:
                    exp_qid = f"Q{match.group(1)}"

            # Check if question ID and option match
            if (exp_qid == normalized_qid and exp_option == option_label):
                # Check if profile matches
                profile_match = (
                    exp_profile.get("gender", "") == user_profile.get("gender", "") and
                    exp_profile.get("proficiency", "") == user_profile.get("proficiency", "") and
                    exp_profile.get("education", "") == user_profile.get(
                        "education", "")
                )

                if profile_match:
                    return f"\nPERSONALIZED EXPLANATION:\n{explanation.get('explanation', '')}"

        # If no exact match found, try to find closest match (same question, any profile)
        for explanation in self.explanation_bank:
            exp_qid = explanation.get("questionId", "")
            exp_option = explanation.get("option", "")

            # Normalize explanation question ID
            if isinstance(exp_qid, str) and exp_qid.startswith('Q'):
                match = re.match(r"Q0*(\d+)", exp_qid)
                if match:
                    exp_qid = f"Q{match.group(1)}"

            if (exp_qid == normalized_qid and exp_option == option_label):
                profile_desc = f"{explanation.get('profile', {}).get('gender', 'General')}, {explanation.get('profile', {}).get('proficiency', 'General')}, {explanation.get('profile', {}).get('education', 'General')}"
                return f"\nRELATED EXPLANATION (for {profile_desc}):\n{explanation.get('explanation', '')}"

        # Fallback explanation
        return f"\nFALLBACK EXPLANATION:\nFor this question about app permissions, it's important to understand the security implications of your choice. Consider reviewing app permission best practices and how they relate to your privacy and security."

    def get_option_label_from_answer(self, question, user_answer):
        """Get the option label (A, B, C, D) from the user's answer text"""
        # Find the question in questions_data
        for q_item in self.questions_data:
            if q_item.get('question') == question:
                for option in q_item.get('options', []):
                    if option.get('text') == user_answer:
                        return option.get('label')
        return "A"  # Default fallback

    def get_detailed_explanation(self, question_id, current_level, desired_level):
        """Basic fallback detailed explanation generator (used when ExplanationBank missing)"""
        return (
            f"Detailed guidance for {question_id}:\n"
            "Review which permissions the app requests, why it needs them, and whether there are safer alternatives. "
            "Consider limiting permissions or using feature-restricted alternatives."
        )

    def save_to_assessment_database(self, user_data):
        """Save assessment results to a structured database file"""
        import datetime

        database_file = 'app_permissions_assessment_database.json'

        # Create assessment record (without detailed responses and scores)
        assessment_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email": user_data['profile']['email'],
            "gender": user_data['profile']['gender'],
            "education_level": user_data['profile']['education'],
            "proficiency": user_data['profile']['proficiency'],
            "total_score": user_data['total_score'],
            "percentage": user_data['percentage'],
            "overall_knowledge_level": user_data['overall_level'],
            "category": "App Permissions"
        }

        # Load existing database or create new one
        try:
            with open(database_file, 'r', encoding='utf-8') as f:
                database = json.load(f)
        except FileNotFoundError:
            database = {
                "metadata": {
                    "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "Mobile App Permissions Security Assessment",
                    "description": "Assessment results database for app permissions security awareness"
                },
                "assessments": []
            }

        # Add new assessment
        database["assessments"].append(assessment_record)

        # Save updated database
        with open(database_file, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False)

        print(f"📊 Assessment saved to database: {database_file}")
        print(f"Total assessments in database: {len(database['assessments'])}")

        return database_file

    def provide_feedback(self, user_scores, overall_level, percentage):
        """Provide detailed feedback and recommendations - now returns data instead of printing"""
        total_score = sum(score_info.get('score', 0)
                          for score_info in user_scores.values())

        # Score summary
        score_summary = f"""
============================================================
APP PERMISSIONS QUIZ RESULTS & PERSONALIZED FEEDBACK
============================================================
Total Score: {total_score}/100
Percentage: {percentage:.1f}%
Overall App Permissions Security Level: {overall_level}
Email: {self.user_profile['email']}
Profile: {self.user_profile['gender']}, {self.user_profile['education']}, {self.user_profile['proficiency']}
"""

        # Level-specific encouragement
        if percentage >= 75:
            encouragement = "\n🎉 Congratulations! You're in the SAFE ZONE!\nYour mobile app permissions security awareness is excellent.\nYou understand how to protect your privacy and data from apps that might misuse permissions."
        elif percentage >= 50:
            encouragement = "\n📈 Good Progress! You're at INTERMEDIATE level!\nYou have a solid foundation but there's room for improvement.\nFocus on the areas below to reach expert level and better protect your privacy."
        elif percentage >= 25:
            encouragement = "\n📚 You're at BASIC level - Learning Time!\nDon't worry! Everyone starts somewhere. App permissions can be tricky to understand.\nThink of it like this: Would you give a stranger the keys to your house? Same with apps and your phone!"
        else:
            encouragement = "\n🌱 You're just getting started - BEGINNER level!\nNo problem at all! Let's learn together step by step.\nThink of your phone like your house - you need to decide who gets keys to which rooms!"

        # Detailed analysis
        detailed_analysis = "\n" + "-"*60 + "\nDETAILED ANALYSIS BY QUESTION:\n" + "-"*60
        improvement_areas = []
        for i, (question, score_info) in enumerate(user_scores.items(), 1):
            level = score_info.get('level', 'basic')
            score = score_info.get('score', 0)
            user_answer = score_info.get('answer', '')
            detailed_analysis += f"\n\nQuestion {i}: {question}\nYour Answer Level: {level.upper()} ({score}/10 points)"

            if score < 10:  # Not perfect answer
                improvement_areas.append({
                    'question': question,
                    'current_level': level,
                    'score': score
                })

                # Get question ID and explanation
                question_id = None
                option_label = None
                for q_item in self.questions_data:
                    if q_item.get('question') == question:
                        question_id = q_item.get('questionId')
                        option_label = self.get_option_label_from_answer(
                            question, user_answer)
                        break

                if question_id and option_label and self.user_profile:
                    explanation = self.get_explanation_from_bank(
                        question_id, option_label, self.user_profile)
                    detailed_analysis += explanation
                else:
                    detailed_analysis += "\n\nBASIC EXPLANATION:\nThis question tests your understanding of app permissions. Consider researching this topic further to improve your knowledge."

        # Priority improvement areas
        priority_areas = ""
        if improvement_areas:
            priority_areas = "\n" + "="*60 + "\nPRIORITY IMPROVEMENT AREAS:\n" + "="*60
            improvement_areas.sort(key=lambda x: x['score'])
            for area in improvement_areas[:3]:  # Top 3
                priority_areas += f"\n\n🎯 Priority Question: {area['question']}\n   Your Current Level: {area['current_level'].upper()}"
                enhanced_advice = self.enhancer.get_detailed_guidance(
                    area['question'], area['current_level'])
                priority_areas += f"\n   📚 Learning Path: {enhanced_advice}"

        # Closing message
        if overall_level.lower() == 'beginner':
            closing = "\n🌟 REMEMBER: Every expert was once a beginner!\nTake your time to learn - your privacy and security are worth it!"
        elif overall_level.lower() == 'basic':
            closing = "\n🚀 YOU'RE MAKING PROGRESS!\nKeep learning and practicing - you're on the right track!"
        elif overall_level.lower() == 'intermediate':
            closing = "\n🎯 ALMOST THERE!\nFocus on the priority areas above to reach expert level!"
        else:
            closing = "\n🏆 EXCELLENT WORK!\nYou're well-equipped to make smart permission decisions!"

        return {
            'score_summary': score_summary,
            'encouragement': encouragement,
            'detailed_analysis': detailed_analysis,
            'priority_areas': priority_areas,
            'closing': closing,
            'total_score': total_score,
            'percentage': percentage,
            'overall_level': overall_level
        }

    def compare_with_last_score(self):
        """Compare current score with last score from database"""
        try:
            with open('app_permissions_assessment_database.json', 'r', encoding='utf-8') as f:
                database = json.load(f)
            assessments = database.get('assessments', [])
            user_assessments = [a for a in assessments if a.get(
                'email', '').lower() == self.user_profile['email'].lower()]
            if user_assessments:
                user_assessments.sort(
                    key=lambda x: x['timestamp'], reverse=True)
                last_assessment = user_assessments[0]  # Most recent
                last_score = last_assessment['percentage']
                current_score = self.current_percentage
                diff = current_score - last_score
                if diff > 0:
                    return f"\n📈 SCORE IMPROVEMENT:\nLast Score: {last_score:.1f}%\nCurrent Score: {current_score:.1f}%\nYou improved by {diff:.1f} points! Keep it up! 🎉"
                elif diff < 0:
                    return f"\n📉 SCORE DECLINE:\nLast Score: {last_score:.1f}%\nCurrent Score: {current_score:.1f}%\nYour score decreased by {abs(diff):.1f} points. Review the priority areas to improve."
                else:
                    return f"\n📊 SCORE UNCHANGED:\nLast Score: {last_score:.1f}%\nCurrent Score: {current_score:.1f}%\nYour score stayed the same. Focus on weak areas for better results."
            else:
                return "\n📊 NO PREVIOUS SCORE:\nThis is your first assessment. Great start!"
        except Exception as e:
            return f"\n❌ Error comparing scores: {e}"

    def run_assessment(self):
        """Run complete assessment process with post-quiz menu"""
        if not self.model or not self.answer_sheet:
            print(
                "Error: Model or answer sheet not loaded. Please train the model first.")
            return

        # Collect user profile first (skip if already set, e.g., from database)
        if self.user_profile is None:
            self.collect_user_profile()

        # Conduct quiz
        user_responses, user_scores = self.conduct_quiz()

        # Calculate results
        total_score, percentage, overall_level = self.calculate_results(
            user_scores)
        self.current_percentage = percentage  # Store for comparison

        # Get feedback data (don't print yet)
        feedback_data = self.provide_feedback(
            user_scores, overall_level, percentage)

        # Save user results including profile
        user_data = {
            'profile': self.user_profile,
            'responses': user_responses,
            'scores': user_scores,
            'total_score': total_score,
            'percentage': percentage,
            'overall_level': overall_level
        }

        # Save to individual results file
        with open('app_permissions_assessment_results.json', 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

        # Save to structured database (this will be called after menu, but for now prepare)
        # We'll call it after the menu loop

        # Post-quiz menu
        while True:
            print("\n" + "="*60)
            print("QUIZ COMPLETED! What would you like to do?")
            print("="*60)
            print("1. View Your Score")
            print("2. Review the Quiz with Explanations")
            print("3. View Priority Improvement Areas")
            print("4. Compare Your Level (with last score)")
            print("5. Back to Main Menu")
            print("6. Save Results and Exit")

            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == '1':
                print(feedback_data['score_summary'] +
                      feedback_data['encouragement'])
            elif choice == '2':
                print(feedback_data['detailed_analysis'])
            elif choice == '3':
                print(feedback_data['priority_areas'] +
                      feedback_data['closing'])
            elif choice == '4':
                comparison = self.compare_with_last_score()
                print(comparison)
            elif choice == '5':
                # Save to database before returning
                database_file = self.save_to_assessment_database(user_data)
                print(
                    f"\n📄 Individual results saved to 'app_permissions_assessment_results.json'")
                print(
                    f"📊 Results added to assessment database: {database_file}")
                return {
                    'score': percentage,
                    'weak_areas': [question for question, score_info in user_scores.items() if score_info.get('score', 0) < 7]
                }
            elif choice == '6':
                # Save and exit
                database_file = self.save_to_assessment_database(user_data)
                print(
                    f"\n📄 Individual results saved to 'app_permissions_assessment_results.json'")
                print(
                    f"📊 Results added to assessment database: {database_file}")
                print("Thank you for completing the assessment!")
                return {
                    'score': percentage,
                    'weak_areas': [question for question, score_info in user_scores.items() if score_info.get('score', 0) < 7]
                }
            else:
                print("Invalid choice! Please enter 1-6.")


if __name__ == "__main__":
    tester = AppPermissionsTester()
    tester.run_assessment()
    tester = AppPermissionsTester()
    tester.run_assessment()
