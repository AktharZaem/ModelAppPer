import json
from typing import List, Dict, Optional
import os


class AppPermissionsEducationalManager:
    def __init__(self):
        self.learning_resources = {
            'beginner': {
                'articles': [
                    {
                        'title': 'Mobile App Permissions Basics',
                        'url': 'https://www.ftc.gov/consumer-advice/blog/2022/11/what-know-about-apps-tracking',
                        'description': 'FTC guide on app permissions and tracking'
                    },
                    {
                        'title': 'Understanding Android Permissions',
                        'url': 'https://support.google.com/android/answer/6014972',
                        'description': 'Google official guide to Android app permissions'
                    },
                    {
                        'title': 'iOS App Privacy Settings',
                        'url': 'https://support.apple.com/en-us/HT203033',
                        'description': 'Apple guide to privacy settings and app permissions'
                    }
                ],
                'videos': [
                    {
                        'title': 'App Permissions 101',
                        'platform': 'Educational Content',
                        'description': 'Basic concepts of mobile app permissions'
                    }
                ]
            },
            'intermediate': {
                'articles': [
                    {
                        'title': 'Advanced Privacy Controls',
                        'url': 'https://www.cisa.gov/mobile-device-security',
                        'description': 'CISA mobile device security guidelines'
                    },
                    {
                        'title': 'Enterprise Mobile Security',
                        'url': 'https://csrc.nist.gov/publications/detail/sp/800-124/rev-1/final',
                        'description': 'NIST mobile device security guidelines'
                    }
                ]
            },
            'advanced': {
                'articles': [
                    {
                        'title': 'Mobile Threat Defense',
                        'url': 'https://www.cisa.gov/mobile-threat-catalog',
                        'description': 'Advanced mobile security threat analysis'
                    },
                    {
                        'title': 'Mobile Device Management',
                        'url': 'https://csrc.nist.gov/projects/mobile-security',
                        'description': 'Enterprise mobile security frameworks'
                    }
                ]
            }
        }

    def assess_knowledge_level(self, quiz_score: float) -> str:
        """Determine knowledge level based on quiz performance"""
        if quiz_score >= 75:
            return 'advanced'
        elif quiz_score >= 50:
            return 'intermediate'
        else:
            return 'beginner'

    def get_learning_resources(self, knowledge_level: str) -> Dict:
        """Get curated learning resources based on knowledge level"""
        return self.learning_resources.get(knowledge_level, self.learning_resources['beginner'])

    def generate_personalized_content(self, weak_areas: List[str], knowledge_level: str) -> str:
        """Generate personalized educational content for app permissions"""
        content = f"""
🎓 PERSONALIZED MOBILE APP PERMISSIONS LEARNING PLAN

Based on your assessment, here are key areas to focus on:

📚 PRIORITY AREAS FOR IMPROVEMENT:
"""

        area_explanations = {
            'location_tracking': """
• LOCATION PERMISSIONS: Control when apps can access your location
  - Only grant location access when necessary for app functionality
  - Review "Always" vs "While Using App" vs "Never" options
  - Regularly audit which apps have location access
  - Consider turning off location history and ad personalization
""",
            'camera_microphone_access': """
• CAMERA & MICROPHONE ACCESS: Protect against unauthorized recording
  - Grant camera/microphone access only to apps that need it
  - Look for privacy indicators when camera/microphone is active
  - Review and revoke permissions for unused apps
  - Be cautious with apps requesting both camera AND microphone
""",
            'contact_data_sharing': """
• CONTACT ACCESS: Protect your contacts' privacy too
  - Understand that granting contact access shares others' information
  - Consider alternatives like manual contact entry
  - Review social media apps' contact access regularly
  - Be aware of contact syncing implications
""",
            'file_storage_access': """
• STORAGE PERMISSIONS: Control app access to your files
  - Understand difference between media access and full storage access
  - Grant storage permissions selectively based on app function
  - Regularly clean up files created by apps
  - Be cautious with apps requesting broad storage access
""",
            'network_permissions': """
• NETWORK ACCESS: Monitor app data usage and connections
  - Understand why apps need internet access
  - Monitor background data usage by apps
  - Use Wi-Fi vs cellular data controls
  - Be aware of apps that might share data with third parties
"""
        }

        for area in weak_areas:
            if area in area_explanations:
                content += area_explanations[area]

        content += f"""

🔧 PRACTICAL EXERCISES:
1. Perform a complete app permissions audit on your device
2. Set up privacy-focused app permission defaults
3. Learn to use your device's permission manager
4. Practice identifying unnecessary permission requests

🌟 KNOWLEDGE LEVEL: {knowledge_level.upper()}
🎯 Goal: Achieve Expert level app permissions management!
"""
        return content

    def get_interactive_tips(self) -> List[str]:
        """Get interactive app permissions security tips"""
        return [
            "💡 Check app permissions before installing - read what access they're requesting",
            "🔍 Regular permission audits: review app permissions monthly",
            "🚫 Principle of least privilege: only grant permissions apps actually need",
            "📱 Use platform privacy dashboards to see which apps accessed what data",
            "🔒 Enable permission prompts for sensitive access like camera and microphone",
            "📍 Location services: use 'While Using App' instead of 'Always' when possible",
            "🗂️ File access: be selective about which apps can access your photos and documents"
        ]

    def display_resources(self, knowledge_level: str):
        """Display formatted educational resources"""
        resources = self.get_learning_resources(knowledge_level)

        print(
            f"\n📚 APP PERMISSIONS EDUCATIONAL RESOURCES - {knowledge_level.upper()} LEVEL")
        print("=" * 70)

        print("\n📖 RECOMMENDED ARTICLES:")
        for i, article in enumerate(resources.get('articles', []), 1):
            print(f"\n{i}. {article['title']}")
            print(f"   📝 {article['description']}")
            print(f"   🔗 {article['url']}")

        if 'videos' in resources:
            print(f"\n🎥 VIDEO RESOURCES:")
            for i, video in enumerate(resources.get('videos', []), 1):
                print(f"\n{i}. {video['title']}")
                print(f"   📝 {video['description']}")
                print(f"   📺 Platform: {video['platform']}")

    def run_educational_session(self, quiz_score: Optional[float] = None, weak_areas: Optional[List[str]] = None):
        """Run an interactive educational session"""
        print("\n🎓 MOBILE APP PERMISSIONS SECURITY EDUCATION CENTER")
        print("=" * 60)

        if quiz_score is not None:
            knowledge_level = self.assess_knowledge_level(quiz_score)
            print(f"\n📊 Your Assessment Score: {quiz_score:.1f}%")
            print(f"🎯 Knowledge Level: {knowledge_level.upper()}")

            if weak_areas:
                print(self.generate_personalized_content(
                    weak_areas, knowledge_level))

            self.display_resources(knowledge_level)
        else:
            print("\n🔍 Select your current knowledge level:")
            print("1. Beginner - New to app permissions security")
            print("2. Intermediate - Some app security awareness")
            print("3. Advanced - Strong mobile security background")

            choice = input("\nEnter your choice (1-3): ").strip()
            level_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced'}
            knowledge_level = level_map.get(choice, 'beginner')

            self.display_resources(knowledge_level)

        print(f"\n💡 QUICK TIPS:")
        tips = self.get_interactive_tips()
        for tip in tips[:4]:
            print(f"   {tip}")

        print(f"\n🎯 CHALLENGE: Review and update your app permissions this week!")
        for tip in tips[:4]:
            print(f"   {tip}")

        print(f"\n🎯 CHALLENGE: Review and update your app permissions this week!")
