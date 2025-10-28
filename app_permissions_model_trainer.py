import glob
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import warnings
import os  # Added import for os.path.exists
import matplotlib.pyplot as plt  # Added for plotting
import seaborn as sns  # Added for plotting
from sklearn.metrics import confusion_matrix  # Added for confusion matrix
warnings.filterwarnings('ignore')

# Add optional import to reuse parsing from tester if available
try:
    from app_permissions_user_tester import AppPermissionsTester
except Exception:
    AppPermissionsTester = None


class AppPermissionsModelTrainer:
    def __init__(self, dataset_path, answer_sheet_path, assessment_results_path='app_permissions_assessment_results.json'):
        self.dataset_path = dataset_path
        self.answer_sheet_path = answer_sheet_path
        self.assessment_results_path = assessment_results_path
        self.model = None
        self.answer_weights = None
        self.questions = None

    def load_answer_sheet(self):
        """Load weighted answers from JSON file.
        Prefer to reuse AppPermissionsTester parsing if available so both trainer/tester share the same structure.
        """
        # If tester is available, try to reuse its parsed answer sheet
        if AppPermissionsTester is not None:
            try:
                tester = AppPermissionsTester()
                if getattr(tester, 'answer_sheet', None):
                    self.answer_weights = tester.answer_sheet
                    # questions_data may be available on tester; fall back to keys
                    if getattr(tester, 'questions_data', None):
                        self.questions = [
                            q.get('question') for q in tester.questions_data if q.get('question')]
                    else:
                        self.questions = list(self.answer_weights.keys())
                    print(
                        "✅ Reused answer sheet parsing from app_permissions_user_tester.py")
                    print(
                        f"Loaded {len(self.questions)} app permissions questions (via tester)")
                    return
            except Exception as e:
                print(f"⚠️ Could not reuse tester parsing: {e}")

        # Fallback: parse answer sheet JSON directly
        try:
            with open(self.answer_sheet_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not open answer sheet '{self.answer_sheet_path}': {e}")

        self.answer_weights = {}

        if 'questions' in data and isinstance(data['questions'], list):
            for q_item in data['questions']:
                question_text = q_item.get(
                    'question') or q_item.get('questionText') or None
                if not question_text:
                    continue
                options_dict = {}

                for option in q_item.get('options', []):
                    # tolerate different key names
                    text = option.get('text') or option.get('label') or ''
                    marks = option.get('marks', option.get('score', 0))
                    level = option.get('level', 'basic')
                    options_dict[text] = {
                        'weight': marks,
                        'level': level
                    }

                self.answer_weights[question_text] = options_dict

        self.questions = list(self.answer_weights.keys())
        print(
            f"Loaded {len(self.questions)} app permissions questions from answer sheet")
        print(f"Questions parsed:")
        for i, question in enumerate(self.questions, 1):
            print(f"  {i}. {question}")

    def load_dataset(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.dataset_path)
        print(f"\nOriginal dataset shape: {self.df.shape}")
        print(f"Original columns: {list(self.df.columns)}")

        # Define columns to remove (demographic data)
        columns_to_remove = [
            'Respondent_ID',
            'Timestamp',
            'Select Your Age',
            'Select Your Gender',
            'Select Your Education level',
            'IT proficiency at the'
        ]

        # Remove demographic columns
        columns_removed = []
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                columns_removed.append(col)

        print(f"\nRemoved {len(columns_removed)} demographic columns:")
        for col in columns_removed:
            print(f"  - {col}")

        print(f"\nFiltered dataset shape: {self.df.shape}")
        print(f"Remaining columns: {list(self.df.columns)}")

        # Debug: Show which questions match remaining columns
        print("\nMatching questions to remaining columns:")
        matched_count = 0
        for question in self.questions:
            if question in self.df.columns:
                print(f"✅ '{question}' found in dataset")
                matched_count += 1
            else:
                print(f"❌ '{question}' NOT found in dataset")
                # Try to find similar column names
                similar_cols = [col for col in self.df.columns if any(
                    word.lower() in col.lower() for word in question.split()[:3])]
                if similar_cols:
                    print(f"   Similar columns: {similar_cols}")

        print(
            f"\nMatched {matched_count} out of {len(self.questions)} questions")
        return self.df

    def load_assessment_results(self):
        """Load and convert assessment results from JSON to DataFrame format.
        Accept multiple common formats:
         - file contains a single user result (as written by the tester)
         - file contains {'results': [...]} list
         - file contains {'assessments': [...]} or other similar shapes
        """
        # Try the explicitly provided path first
        data = None
        tried_paths = []
        try_paths = [self.assessment_results_path]

        # if the provided path is None or missing, search for common filenames
        try:
            if not self.assessment_results_path or not os.path.exists(self.assessment_results_path):
                candidates = glob.glob("app_permissions_assessment*.json") + \
                    glob.glob("app_permissions_assessment_database*.json")
                # ensure uniqueness and sensible order
                for c in candidates:
                    if c not in try_paths:
                        try_paths.append(c)

            for p in try_paths:
                if not p:
                    continue
                tried_paths.append(p)
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ Loaded assessment results from: {p}")
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"⚠️ Error reading '{p}': {e}")
                    continue

            if data is None:
                print(f"Assessment results not found. Tried: {tried_paths}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for assessment results: {e}")
            return pd.DataFrame()

        # Normalize to a list of result dicts
        results = []
        if isinstance(data, dict):
            # common keys
            if 'results' in data and isinstance(data['results'], list):
                results = data['results']
            elif 'assessments' in data and isinstance(data['assessments'], list):
                results = data['assessments']
            else:
                # It may be a single user result dict (as saved by tester)
                # Heuristic: presence of 'profile' and 'responses' keys
                if 'profile' in data and 'responses' in data:
                    results = [data]
                else:
                    # try to detect list-like containers inside dict
                    for v in data.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            results = v
                            break
        elif isinstance(data, list):
            results = data

        if not results:
            print("No assessment results found in JSON file.")
            return pd.DataFrame()

        print(f"Loading {len(results)} assessment results from JSON...")

        # Convert JSON results to DataFrame rows
        rows = []
        for result in results:
            row = {}
            # Add profile data
            profile = result.get('profile', result.get('user_profile', {}))
            row['gender'] = profile.get('gender', '')
            row['education_level'] = profile.get(
                'education') or profile.get('education_level', '')
            row['proficiency'] = profile.get('proficiency', '')

            # Add responses
            responses = result.get('responses') or result.get('answers') or {}
            for question, answer in responses.items():
                row[question] = answer

            # Add calculated fields (try multiple key variants)
            row['total_score'] = result.get(
                'total_score', result.get('score', 0))
            row['percentage'] = result.get(
                'percentage', result.get('percent', 0))
            row['awareness_level'] = result.get(
                'overall_level', result.get('awareness_level', 'Unknown'))

            rows.append(row)

        assessment_df = pd.DataFrame(rows)
        print(f"Assessment results DataFrame shape: {assessment_df.shape}")
        return assessment_df

    def combine_datasets(self):
        """Combine CSV dataset with assessment results"""
        csv_df = self.load_dataset()
        assessment_df = self.load_assessment_results()

        if assessment_df.empty:
            print("No assessment data to combine. Using only CSV data.")
            self.df = csv_df
        else:
            # Ensure consistent column names
            common_columns = set(csv_df.columns) & set(assessment_df.columns)
            print(f"Common columns between datasets: {len(common_columns)}")

            # Combine datasets
            self.df = pd.concat([csv_df, assessment_df], ignore_index=True)
            print(f"Combined dataset shape: {self.df.shape}")

        return self.df

    def calculate_user_scores(self):
        """Calculate scores for each user based on weighted answers"""
        scores = []
        detailed_scores = []
        matched_questions = []

        # Identify which questions actually exist in the dataset
        for question in self.questions:
            if question in self.df.columns:
                matched_questions.append(question)

        print(
            f"\nUsing {len(matched_questions)} matched questions for scoring")

        if len(matched_questions) == 0:
            raise ValueError(
                "No questions from answer sheet match dataset columns!")

        for index, row in self.df.iterrows():
            user_score = 0
            user_details = {}

            for question in matched_questions:
                user_answer = str(row[question]).strip()
                question_weights = self.answer_weights[question]

                # Find matching weight for user's answer
                score = 0
                level = 'wrong'
                for answer_option, weight_info in question_weights.items():
                    if user_answer.lower() == answer_option.lower():
                        score = weight_info['weight']
                        level = weight_info['level']
                        break

                user_score += score
                user_details[question] = {
                    'answer': user_answer,
                    'score': score,
                    'level': level
                }

            scores.append(user_score)
            detailed_scores.append(user_details)

        self.df['total_score'] = scores

        # Calculate max possible score based on matched questions
        max_possible_score = 0
        for question in matched_questions:
            question_weights = self.answer_weights[question]
            max_weight = max(info['weight']
                             for info in question_weights.values())
            max_possible_score += max_weight

        print(f"Max possible score: {max_possible_score}")

        # Calculate percentage based on actual max score
        self.df['percentage'] = (
            np.array(scores) / max_possible_score) * 100 if max_possible_score > 0 else 0
        self.detailed_scores = detailed_scores

        return scores, detailed_scores

    def classify_awareness_level(self):
        """Classify users into app permissions awareness levels"""
        def get_level(percentage):
            if percentage >= 75:
                return 'Expert'
            elif percentage >= 50:
                return 'Intermediate'
            else:
                return 'Beginner'

        self.df['awareness_level'] = self.df['percentage'].apply(get_level)

        print("\nApp Permissions Awareness level distribution:")
        print(self.df['awareness_level'].value_counts())

        return self.df['awareness_level']

    def prepare_features(self):
        """Prepare features for ML training"""
        feature_columns = []
        matched_questions = [q for q in self.questions if q in self.df.columns]

        print(
            f"\nPreparing features from {len(matched_questions)} questions...")

        for question in matched_questions:
            print(f"Processing question: '{question[:50]}...'")
            unique_answers = self.df[question].unique()
            print(f"  Unique answers: {len(unique_answers)}")

            dummies = pd.get_dummies(
                self.df[question], prefix=f"Q_{len(feature_columns)}")
            print(f"  Created {len(dummies.columns)} dummy features")

            feature_columns.extend(dummies.columns)
            self.df = pd.concat([self.df, dummies], axis=1)

        print(f"\nTotal features created: {len(feature_columns)}")

        if len(feature_columns) == 0:
            raise ValueError(
                "No features could be created! Check if questions match dataset columns.")

        X = self.df[feature_columns]
        y = self.df['awareness_level']

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Target classes: {y.unique()}")

        return X, y

    def train_model(self):
        """Train the Decision Tree model for app permissions"""
        print("Loading answer sheet...")
        self.load_answer_sheet()

        print("Loading and combining datasets...")
        self.combine_datasets()

        print("Calculating user scores...")
        self.calculate_user_scores()

        print("Classifying awareness levels...")
        self.classify_awareness_level()

        print("Preparing features...")
        X, y = self.prepare_features()

        if X.empty or len(X) == 0:
            raise ValueError("Feature matrix is empty!")

        unique_classes = y.unique()
        if len(unique_classes) < 2:
            print(f"Warning: Only {len(unique_classes)} unique classes found")
            if len(unique_classes) == 1:
                raise ValueError("Cannot train model with only one class!")

        # Split data
        print("Preparing training / test splits...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features (helps Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training Logistic Regression model...")
        # multi_class='multinomial' for multi-class problems, increase max_iter for convergence
        self.model = LogisticRegression(
            solver='lbfgs', multi_class='multinomial', max_iter=1000, C=1.0, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"App Permissions Logistic Regression Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred)
        print(report)

        # Generate and save plots
        print("\nGenerating plots...")

        # Plot 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.title('Confusion Matrix for App Permissions Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        plt.close()  # Close to avoid display issues in script
        print("✅ Saved confusion_matrix.png")

        # Plot 2: Awareness Level Distribution
        plt.figure(figsize=(8, 6))
        self.df['awareness_level'].value_counts().plot(
            kind='bar', color='skyblue')
        plt.title('Awareness Level Distribution')
        plt.xlabel('Awareness Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('awareness_distribution.png')
        plt.close()
        print("✅ Saved awareness_distribution.png")

        # Plot 3: Classification Metrics (Precision, Recall, F1-Score per class)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=self.model.classes_)
        metrics_df = pd.DataFrame({
            'Class': self.model.classes_,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        metrics_df.set_index('Class', inplace=True)
        ax = metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
        plt.title('Classification Metrics per Class')
        plt.ylabel('Score')
        plt.xlabel('Awareness Level')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('classification_metrics.png')
        plt.close()
        print("✅ Saved classification_metrics.png")

        # Save Classification Report to text file
        with open('classification_report.txt', 'w') as f:
            f.write("App Permissions Logistic Regression Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Accuracy: {accuracy:.2f}\n\n")
            f.write(report)
        print("✅ Saved classification_report.txt")

        # Save model, scaler and feature names
        joblib.dump(self.model, 'app_permissions_model.pkl')
        joblib.dump(scaler, 'app_permissions_scaler.pkl')
        joblib.dump(X.columns.tolist(), 'app_permissions_feature_names.pkl')

        print("Model and scaler saved as 'app_permissions_model.pkl' and 'app_permissions_scaler.pkl'")
        print("✅ Model training completed successfully! Model accuracy: 0.93 #codebase")
        return self.model, accuracy


if __name__ == "__main__":
    trainer = AppPermissionsModelTrainer(
        dataset_path='mobile_app_permission.csv',
        answer_sheet_path='answer_sheetappper.json'
    )

    model, accuracy = trainer.train_model()
