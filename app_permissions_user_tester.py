import json
import joblib
import pandas as pd
import numpy as np
import os
import requests  # Now available since you installed it
from app_permissions_knowledge_enhancer import AppPermissionsKnowledgeEnhancer


class AppPermissionsTester:
    def __init__(self):
        self.answer_sheet = None
        self.questions_data = None
        self.model = None
        self.feature_names = None
        self.enhancer = AppPermissionsKnowledgeEnhancer()
        # Re-enable Gemini API with your key
        self.gemini_api_key = os.getenv(
            'GEMINI_API_KEY') or "AIzaSyDuDJ5uyh3DBAjEFTHaCz-g25fH7hp72Yc"
        self.load_components()

    def load_components(self):
        """Load trained model and answer sheet"""
        try:
            # Load answer sheet and parse the nested structure
            with open('answer_sheetappper.json', 'r') as f:
                data = json.load(f)

            self.answer_sheet = {}
            self.questions_data = []

            if 'questions' in data and isinstance(data['questions'], list):
                for q_item in data['questions']:
                    question_text = q_item['question']
                    options_dict = {}

                    for option in q_item['options']:
                        options_dict[option['text']] = {
                            'weight': option['marks'],
                            'level': option['level']
                        }

                    self.answer_sheet[question_text] = options_dict
                    self.questions_data.append(q_item)

            # Load trained model
            self.model = joblib.load('app_permissions_model.pkl')
            self.feature_names = joblib.load(
                'app_permissions_feature_names.pkl')

            print("App permissions components loaded successfully!")
            print(f"Loaded {len(self.questions_data)} questions for quiz")

        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run app_permissions_model_trainer.py first to train the model")

    def conduct_quiz(self):
        """Conduct interactive quiz with user"""
        print("\n=== Mobile App Permissions Security Awareness Quiz ===")
        print("Please answer the following 10 questions about mobile app permissions.\n")

        user_responses = {}
        user_scores = {}

        for i, q_item in enumerate(self.questions_data, 1):
            question = q_item['question']
            options = q_item['options']

            print(f"Question {i}: {question}")
            print("\nOptions:")

            # Display options
            for j, option in enumerate(options, 1):
                print(f"{j}. {option['text']}")

            # Get user input
            while True:
                try:
                    choice = int(
                        input(f"\nEnter your choice (1-{len(options)}): "))
                    if 1 <= choice <= len(options):
                        selected_option = options[choice - 1]
                        selected_answer = selected_option['text']

                        user_responses[question] = selected_answer

                        # Get score and level for this answer
                        user_scores[question] = {
                            'answer': selected_answer,
                            'score': selected_option['marks'],
                            'level': selected_option['level']
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
        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
        max_possible_score = len(user_scores) * 10
        percentage = (total_score / max_possible_score) * 100

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

    def get_gemini_explanation(self, question, current_level, overall_level):
        """Get personalized explanation from Gemini API"""
        if not self.gemini_api_key:
            return self.get_detailed_explanation(question, current_level, overall_level)

        try:
            # Prepare the prompt for Gemini
            prompt = f"""
You are an expert cybersecurity educator specializing in mobile app permissions. 

CONTEXT:
- User's Question: "{question}"
- User's Current Answer Level: {current_level}
- User's Overall Knowledge Level: {overall_level}

TASK:
Provide a personalized explanation to help this user understand this app permission concept and advance to the next level. 

GUIDELINES:
- If user is at "wrong" level, explain the basics very simply (like explaining to a child)
- If user is at "basic" level, provide more detailed explanations with examples
- If user is at "intermediate" level, give advanced concepts and best practices
- If user is at "advanced" level, provide expert-level insights and enterprise considerations

FORMAT:
- Use emojis and clear structure
- Include practical examples
- Explain WHY this matters for their privacy and security
- Give actionable next steps
- Keep it engaging and educational
- Maximum 300 words

Please provide a comprehensive explanation that will help them improve from their current level to the next level.
"""

            # Try multiple model endpoints until one works
            model_names = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "gemini-pro"
            ]

            for model_name in model_names:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"

                    headers = {
                        "Content-Type": "application/json"
                    }

                    data = {
                        "contents": [{
                            "parts": [{
                                "text": prompt
                            }]
                        }],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 500,
                        }
                    }

                    print(f"📚 Model thinking for better learning curve...")
                    response = requests.post(
                        url, headers=headers, json=data, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        if 'candidates' in result and len(result['candidates']) > 0:
                            generated_text = result['candidates'][0]['content']['parts'][0]['text']
                            return f"\nPERSONALIZED EXPLANATION:\n{generated_text}"
                    else:
                        print(
                            f"⚠️ Model {model_name} failed with status {response.status_code}")
                        continue  # Try next model

                except Exception as model_error:
                    print(f"⚠️ Error with model {model_name}: {model_error}")
                    continue  # Try next model

            # If all models fail, fall back to predefined explanations
            print("⚠️ All models failed, using fallback explanation")
            return self.get_detailed_explanation(question, current_level, overall_level)

        except Exception as e:
            print(f"⚠️ Error calling API: {e}")
            return self.get_detailed_explanation(question, current_level, overall_level)

    def get_detailed_explanation(self, question, current_level, overall_level):
        """Get detailed explanation based on question and user's overall knowledge level"""
        explanations = {
            "When installing a new app, what should you do first?": {
                "basic": """
🔍 WHAT ARE APP PERMISSIONS?
Think of app permissions like keys to your house. When you install an app, it's like giving someone keys to different rooms in your digital home.

📱 WHAT TO DO FIRST:
• Always read what permissions the app is asking for
• Ask yourself: "Does this app REALLY need this access?"
• For example: Why would a calculator app need your camera?
• Only say "YES" if it makes sense for what the app does

💡 SIMPLE RULE: If it doesn't make sense, don't allow it!
""",
                "intermediate": """
🔒 UNDERSTANDING APP PERMISSIONS:
App permissions control what data and features apps can access on your device. Think of it as a security checkpoint.

🛡️ BEST PRACTICES WHEN INSTALLING:
• Review the permission list during installation
• Understand the difference between essential and optional permissions
• Check if permissions align with the app's stated functionality
• Research the app developer's reputation and privacy policy
• Consider alternatives if too many unnecessary permissions are requested

⚖️ RISK ASSESSMENT: Weigh the app's benefits against privacy risks
""",
                "advanced": """
🎯 ADVANCED PERMISSION MANAGEMENT:
Implementing a systematic approach to app permission evaluation requires understanding the security implications and data flow.

🔬 COMPREHENSIVE EVALUATION PROCESS:
• Conduct permission auditing before installation
• Analyze the app's security model and data handling practices
• Implement principle of least privilege - grant minimal necessary access
• Consider runtime permissions vs install-time permissions
• Evaluate the app's compliance with platform security guidelines
• Monitor permission usage patterns post-installation

🏢 ENTERPRISE CONSIDERATIONS: Apply organizational security policies and mobile device management standards
"""
            },
            "Why should a flashlight app not need access to your contacts?": {
                "basic": """
🔦 FLASHLIGHT APP EXAMPLE:
A flashlight app's job is simple - turn your phone's light on and off. That's it!

📞 WHY NO CONTACTS ACCESS?
• Your contacts have nothing to do with making light
• The app doesn't need names or phone numbers to work
• Giving contact access means the app can see and potentially share your friends' information
• This could be used for spam or unwanted marketing

🚨 RED FLAG RULE: If an app asks for something it doesn't need for its main job, be suspicious!
""",
                "intermediate": """
🔍 UNDERSTANDING UNNECESSARY PERMISSIONS:
This is a classic example of permission overreach - when apps request access beyond their core functionality.

⚠️ SECURITY IMPLICATIONS:
• Contact access allows apps to harvest personal data for commercial purposes
• Data can be sold to third parties or used for targeted advertising
• Potential for social engineering attacks using your contact information
• Privacy violation extends to your contacts who didn't consent to data sharing

🛡️ PROTECTION STRATEGY: Always question why any app needs access to sensitive data like contacts, location, or camera
""",
                "advanced": """
🔒 ADVANCED PERMISSION ANALYSIS:
This scenario demonstrates the importance of implementing strict permission models and understanding data minimization principles.

🏗️ SECURITY ARCHITECTURE CONSIDERATIONS:
• Implement application sandboxing to prevent unauthorized data access
• Understand the difference between legitimate functionality and data harvesting
• Analyze the app's data flow and third-party integrations
• Consider implementing mobile application management (MAM) policies
• Evaluate the risk of lateral movement through interconnected data access

🎯 ENTERPRISE STRATEGY: Develop organization-wide policies for app vetting and permission management
"""
            },
            "Which app should be allowed microphone access?": {
                "basic": """
🎤 MICROPHONE PERMISSION BASICS:
Your microphone lets apps record sound. Only some apps really need this!

✅ APPS THAT SHOULD GET MICROPHONE ACCESS:
• Voice recording apps (to record your voice)
• Video calling apps like WhatsApp, Zoom (to talk to people)
• Music apps like Shazam (to identify songs)
• Voice assistants like Siri, Google Assistant

❌ APPS THAT DON'T NEED MICROPHONE:
• Games (unless they have voice features)
• Photo editing apps
• Calculator apps
• Flashlight apps

🔒 SAFETY TIP: Your microphone can hear everything around you, so be careful who you give access to!
""",
                "intermediate": """
🎯 MICROPHONE PERMISSION MANAGEMENT:
Microphone access is one of the most sensitive permissions as it can capture private conversations and ambient audio.

🔍 LEGITIMATE USE CASES:
• Communication apps (calls, messaging, video conferencing)
• Audio/video recording and editing applications
• Voice assistants and dictation software
• Music recognition and audio analysis tools
• Accessibility applications with voice control features

⚠️ SECURITY CONSIDERATIONS:
• Always check for visual/audio indicators when microphone is active
• Review which apps have background microphone access
• Be aware that some apps may record continuously
• Understand the difference between "while using app" vs "always" permissions

🛡️ BEST PRACTICE: Regularly audit microphone permissions and revoke access for unused apps
""",
                "advanced": """
🔒 ENTERPRISE MICROPHONE SECURITY:
Microphone permissions represent a significant attack vector for corporate espionage and privacy breaches.

🏢 ADVANCED SECURITY FRAMEWORK:
• Implement mobile device management (MDM) policies for audio recording
• Deploy mobile threat defense (MTD) solutions to monitor microphone usage
• Establish data loss prevention (DLP) policies for audio content
• Configure runtime permission monitoring and anomaly detection
• Implement zero-trust models for audio-enabled applications

🎯 COMPLIANCE CONSIDERATIONS:
• GDPR implications for audio data collection
• Industry-specific regulations (healthcare, finance, government)
• Audit trails for microphone access in corporate environments
• Incident response procedures for unauthorized audio access
"""
            }
        }

        question_key = question
        if question_key in explanations and overall_level.lower() in explanations[question_key]:
            return explanations[question_key][overall_level.lower()]
        else:
            # Fallback explanation
            return f"""
📚 LEARNING OPPORTUNITY:
This question tests your understanding of app permissions. The key is to think about whether the permission makes sense for what the app does.

🎯 NEXT STEPS:
• Research this topic online
• Check your device's permission settings
• Practice reviewing app permissions before installing
• Learn about privacy and security best practices
"""

    def provide_feedback(self, user_scores, overall_level, percentage):
        """Provide detailed feedback and recommendations"""
        print("\n" + "="*60)
        print("APP PERMISSIONS QUIZ RESULTS & PERSONALIZED FEEDBACK")
        print("="*60)

        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
        print(f"Total Score: {total_score}/100")
        print(f"Percentage: {percentage:.1f}%")
        print(f"Overall App Permissions Security Level: {overall_level}")

        # Provide level-specific encouragement
        if percentage >= 75:
            print("\n🎉 Congratulations! You're in the SAFE ZONE!")
            print("Your mobile app permissions security awareness is excellent.")
            print(
                "You understand how to protect your privacy and data from apps that might misuse permissions.")
        elif percentage >= 50:
            print("\n📈 Good Progress! You're at INTERMEDIATE level!")
            print("You have a solid foundation but there's room for improvement.")
            print(
                "Focus on the areas below to reach expert level and better protect your privacy.")
        elif percentage >= 25:
            print("\n📚 You're at BASIC level - Learning Time!")
            print(
                "Don't worry! Everyone starts somewhere. App permissions can be tricky to understand.")
            print("Think of it like this: Would you give a stranger the keys to your house? Same with apps and your phone!")
        else:
            print("\n🌱 You're just getting started - BEGINNER level!")
            print("No problem at all! Let's learn together step by step.")
            print(
                "Think of your phone like your house - you need to decide who gets keys to which rooms!")

        print("\n" + "-"*60)
        print("DETAILED ANALYSIS BY QUESTION:")
        print("-"*60)

        improvement_areas = []

        for i, (question, score_info) in enumerate(user_scores.items(), 1):
            level = score_info['level']
            score = score_info['score']

            print(f"\nQuestion {i}: {question}")
            print(f"Your Answer Level: {level.upper()} ({score}/10 points)")

            if score < 10:  # Not perfect answer
                improvement_areas.append({
                    'question': question,
                    'current_level': level,
                    'score': score
                })

                # Get AI-generated explanation based on overall level
                ai_explanation = self.get_gemini_explanation(
                    question, level, overall_level)
                print(ai_explanation)

        # Overall recommendations with level-appropriate language
        if improvement_areas:
            print("\n" + "="*60)
            print("PRIORITY IMPROVEMENT AREAS:")
            print("="*60)

            # Sort by score (lowest first)
            improvement_areas.sort(key=lambda x: x['score'])

            for area in improvement_areas[:3]:  # Top 3 priority areas
                print(f"\n🎯 Priority Question: {area['question']}")
                print(
                    f"   Your Current Level: {area['current_level'].upper()}")

                # Get enhanced advice from knowledge enhancer
                enhanced_advice = self.enhancer.get_detailed_guidance(
                    area['question'], area['current_level']
                )
                print(f"   📚 Learning Path: {enhanced_advice}")

        # Add level-appropriate closing message
        print("\n" + "="*60)
        if overall_level.lower() == 'beginner':
            print("🌟 REMEMBER: Every expert was once a beginner!")
            print("Take your time to learn - your privacy and security are worth it!")
        elif overall_level.lower() == 'basic':
            print("🚀 YOU'RE MAKING PROGRESS!")
            print("Keep learning and practicing - you're on the right track!")
        elif overall_level.lower() == 'intermediate':
            print("🎯 ALMOST THERE!")
            print("Focus on the priority areas above to reach expert level!")
        else:
            print("🏆 EXCELLENT WORK!")
            print("You're well-equipped to make smart permission decisions!")

    def run_assessment(self):
        """Run complete assessment process"""
        if not self.model or not self.answer_sheet:
            print(
                "Error: Model or answer sheet not loaded. Please train the model first.")
            return

        # Conduct quiz
        user_responses, user_scores = self.conduct_quiz()

        # Calculate results
        total_score, percentage, overall_level = self.calculate_results(
            user_scores)

        # Provide feedback
        self.provide_feedback(user_scores, overall_level, percentage)

        # Save user results
        user_data = {
            'responses': user_responses,
            'scores': user_scores,
            'total_score': total_score,
            'percentage': percentage,
            'overall_level': overall_level
        }

        with open('app_permissions_assessment_results.json', 'w') as f:
            json.dump(user_data, f, indent=2)

        print(f"\n📄 Results saved to 'app_permissions_assessment_results.json'")

        return {
            'score': percentage,
            'weak_areas': [question for question, score_info in user_scores.items() if score_info['score'] < 7]
        }


if __name__ == "__main__":
    tester = AppPermissionsTester()
    tester.run_assessment()
if __name__ == "__main__":
    tester = AppPermissionsTester()
    tester.run_assessment()
    tester = AppPermissionsTester()
    tester.run_assessment()
    tester.run_assessment()
    tester = AppPermissionsTester()
    tester.run_assessment()
