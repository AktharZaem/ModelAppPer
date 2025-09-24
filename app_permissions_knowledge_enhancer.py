import json
from urllib.parse import quote


class AppPermissionsKnowledgeEnhancer:
    def __init__(self):
        self.enhancement_database = self.load_enhancement_database()

    def load_enhancement_database(self):
        """Load pre-defined enhancement recommendations for app permissions"""
        return {
            "permission_understanding": {
                "basic": "Learn what different app permissions mean and why apps request them",
                "intermediate": "Understand the security implications of granting various permissions",
                "advanced": "Study permission models and enterprise mobile device management"
            },
            "location_permissions": {
                "basic": "Understand when and why apps need location access",
                "intermediate": "Learn about location privacy settings and background tracking",
                "advanced": "Implement location-based security policies and privacy controls"
            },
            "camera_microphone": {
                "basic": "Learn about camera and microphone permissions and their risks",
                "intermediate": "Understand app behavior monitoring and privacy indicators",
                "advanced": "Implement hardware access controls and monitoring solutions"
            },
            "contact_access": {
                "basic": "Understand why apps request contact access and alternatives",
                "intermediate": "Learn about contact data protection and sharing implications",
                "advanced": "Study contact data encryption and enterprise contact management"
            },
            "storage_permissions": {
                "basic": "Learn about file and storage access permissions",
                "intermediate": "Understand app sandboxing and data isolation concepts",
                "advanced": "Implement secure storage solutions and data loss prevention"
            },
            "network_permissions": {
                "basic": "Understand internet and network access permissions",
                "intermediate": "Learn about network monitoring and data usage tracking",
                "advanced": "Implement network security policies and traffic analysis"
            },
            "permission_review": {
                "basic": "Learn how to regularly review and manage app permissions",
                "intermediate": "Develop systematic approaches to permission auditing",
                "advanced": "Create automated permission monitoring and compliance systems"
            }
        }

    def get_enhancement_advice(self, question, current_level):
        """Get basic enhancement advice for a question"""
        topic = self.map_question_to_topic(question)

        if topic in self.enhancement_database:
            if current_level in self.enhancement_database[topic]:
                return self.enhancement_database[topic][current_level]

        return f"Continue learning about {topic} to improve your app permissions security awareness."

    def get_detailed_guidance(self, question, current_level):
        """Get detailed learning guidance with suggested resources"""
        topic = self.map_question_to_topic(question)
        next_level = self.get_next_level(current_level)

        guidance = f"To advance from {current_level} to {next_level} level in {topic}:\n"

        if current_level == "wrong" or current_level == "basic":
            guidance += f"   • Start with fundamentals of {topic}\n"
            guidance += f"   • Practice reviewing app permissions on your device\n"
            guidance += f"   • Use official platform security guides\n"
        elif current_level == "intermediate":
            guidance += f"   • Deepen your understanding of {topic}\n"
            guidance += f"   • Explore advanced privacy and security settings\n"
            guidance += f"   • Consider mobile security certifications\n"

        # Add search suggestions
        search_terms = self.generate_search_terms(topic, current_level)
        guidance += f"   • Recommended searches: {', '.join(search_terms)}"

        return guidance

    def map_question_to_topic(self, question):
        """Map question text to app permissions topic"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['location', 'gps', 'tracking', 'where']):
            return 'location_permissions'
        elif any(word in question_lower for word in ['camera', 'microphone', 'record', 'photo']):
            return 'camera_microphone'
        elif any(word in question_lower for word in ['contacts', 'phone book', 'address book']):
            return 'contact_access'
        elif any(word in question_lower for word in ['storage', 'files', 'photos', 'documents']):
            return 'storage_permissions'
        elif any(word in question_lower for word in ['internet', 'network', 'data', 'wifi']):
            return 'network_permissions'
        elif any(word in question_lower for word in ['review', 'check', 'manage', 'audit']):
            return 'permission_review'
        else:
            return 'permission_understanding'

    def get_next_level(self, current_level):
        """Determine the next level to aim for"""
        level_progression = {
            'wrong': 'basic',
            'basic': 'intermediate',
            'intermediate': 'advanced',
            'advanced': 'expert'
        }
        return level_progression.get(current_level, 'advanced')

    def generate_search_terms(self, topic, level):
        """Generate relevant search terms for learning"""
        base_terms = {
            'location_permissions': ['app location permission guide', 'mobile location privacy'],
            'camera_microphone': ['app camera permission security', 'microphone access control'],
            'contact_access': ['app contact permission risks', 'contact privacy protection'],
            'storage_permissions': ['app file access security', 'mobile storage permissions'],
            'network_permissions': ['app internet permission guide', 'mobile network security'],
            'permission_review': ['app permission audit guide', 'mobile security checkup'],
            'permission_understanding': ['mobile app permissions explained', 'smartphone security basics']
        }

        level_modifiers = {
            'basic': ['beginner guide', 'tutorial'],
            'intermediate': ['best practices', 'advanced guide'],
            'advanced': ['enterprise security', 'professional guide']
        }

        terms = base_terms.get(topic, ['app permissions security'])
        modifiers = level_modifiers.get(level, ['tutorial'])

        enhanced_terms = []
        for term in terms:
            enhanced_terms.append(f"{term} {modifiers[0]}")

        return enhanced_terms[:3]

    def get_google_search_url(self, query):
        """Generate Google search URL for a query"""
        encoded_query = quote(query)
        return f"https://www.google.com/search?q={encoded_query}"

    def generate_learning_path(self, user_scores):
        """Generate complete learning path based on user performance"""
        weak_areas = []
        for question, score_info in user_scores.items():
            if score_info['score'] < 7:
                weak_areas.append({
                    'topic': self.map_question_to_topic(question),
                    'level': score_info['level'],
                    'score': score_info['score']
                })

        weak_areas.sort(key=lambda x: x['score'])

        learning_path = []
        for area in weak_areas:
            path_item = {
                'topic': area['topic'],
                'current_level': area['level'],
                'target_level': self.get_next_level(area['level']),
                'resources': self.generate_search_terms(area['topic'], area['level']),
                'priority': 'High' if area['score'] < 3 else 'Medium'
            }
            learning_path.append(path_item)

        return learning_path
        return learning_path
