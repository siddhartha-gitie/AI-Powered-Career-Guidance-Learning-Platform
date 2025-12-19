import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import numpy as np
from collections import Counter
import time
from sqlalchemy import func, inspect
from datetime import datetime
import random 

# --- App Configuration ---
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'a_very_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Create SQLAlchemy instance (BEFORE models) ---
db = SQLAlchemy()

# --- Initialize SQLAlchemy with app (BEFORE models) ---
db.init_app(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- NOW define models (AFTER db.init_app) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    quiz_results = db.relationship('QuizResult', backref='user', lazy=True, cascade='all, delete-orphan')
    roadmaps = db.relationship('UserRoadmap', backref='user', lazy=True, cascade='all, delete-orphan') # New relationship

class UserCourse(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, primary_key=True)
    course_name = db.Column(db.String(200), nullable=False, primary_key=True)

class QuizResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_name = db.Column(db.String(200), nullable=False)
    level = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    time_taken = db.Column(db.Float, nullable=False)  # in minutes
    learning_profile_predicted = db.Column(db.String(100), nullable=True) # Store the predicted profile
    recommended_next_step = db.Column(db.String(255), nullable=True) # Store the model's next step
    roadmap_details_json = db.Column(db.Text, nullable=True) # Store the generated roadmap details as JSON string
    timestamp = db.Column(db.DateTime, default=func.now())

class UserRoadmap(db.Model): # New table for explicit roadmap saves
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_name = db.Column(db.String(200), nullable=False)
    level = db.Column(db.String(50), nullable=False)
    learning_profile_at_generation = db.Column(db.String(100), nullable=True)
    recommended_next_step = db.Column(db.String(255), nullable=True)
    roadmap_details_json = db.Column(db.Text, nullable=False) # Store the generated roadmap
    quiz_result_id = db.Column(db.Integer, db.ForeignKey('quiz_result.id'), nullable=True) # Link to quiz result
    timestamp = db.Column(db.DateTime, default=func.now())


class SavedCareer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    career_domain = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())

class SavedSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_title = db.Column(db.String(200), nullable=False)
    skills_list = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Helper Function to Load & Format Quizzes ---
def load_quiz_questions(file_path):
    print(f"\n--- Loading Quiz Question Bank from {os.path.basename(file_path)} ---")
    quiz_bank = {}
    total_raw_entries = 0
    total_valid_questions = 0
    skipped_questions_log = [] # Collect all skipped questions with reasons

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_questions_raw = json.load(f)
        total_raw_entries = len(all_questions_raw)

        # First pass: Deduplicate based on 'id' (your new field)
        deduplicated_questions_by_id = {}
        for q_raw in all_questions_raw:
            q_id = q_raw.get('id')
            if not q_id:
                skipped_questions_log.append(f"Skipped raw entry (missing 'id' field): '{q_raw.get('question', 'UNKNOWN_QUESTION_NO_ID')}'")
                continue
            if q_id not in deduplicated_questions_by_id:
                deduplicated_questions_by_id[q_id] = q_raw
            else:
                skipped_questions_log.append(f"Skipped raw entry (duplicate 'id' '{q_id}'): '{q_raw.get('question', 'UNKNOWN_QUESTION_DUPLICATE_ID')}'")
        
        # Second pass: Validate and format each question
        for q_original in deduplicated_questions_by_id.values():
            q_id = q_original.get('id', 'N/A')
            course = q_original.get('course_name')
            level = q_original.get('level')
            question_text = q_original.get('question')
            options_dict_raw = q_original.get('options')
            correct_answer_key = q_original.get('correct_answer')
            # NEW: Extract topic_tags
            topic_tags = q_original.get('topic_tags', []) 

            # --- Validation Checks ---
            if not all([q_id, course, level, question_text]):
                skipped_questions_log.append(f"Q ID: {q_id} - Missing essential fields (id, course_name, level, or question text): {q_original}")
                continue
            
            if not isinstance(options_dict_raw, dict):
                skipped_questions_log.append(f"Q ID: {q_id} - 'options' field is not a dictionary: {options_dict_raw}")
                continue

            # Filter out empty or None options and get list of actual option texts
            # Assuming options are consistently 'A', 'B', 'C', 'D'
            valid_option_keys = ['A', 'B', 'C', 'D']
            parsed_options_list = []
            for key in valid_option_keys:
                option_text = options_dict_raw.get(key)
                if option_text is not None and str(option_text).strip() != '':
                    parsed_options_list.append(str(option_text).strip())
            
            if not parsed_options_list or len(parsed_options_list) < 2: # Require at least 2 non-empty options for a valid MCQ
                skipped_questions_log.append(f"Q ID: {q_id} - Insufficient or malformed options (less than 2 valid options): '{question_text}' - Raw: {options_dict_raw}")
                continue

            # Ensure correct answer key is valid and maps to an actual option text
            correct_answer_text = options_dict_raw.get(correct_answer_key, '')
            
            if not correct_answer_key or str(correct_answer_key).strip() == '':
                 skipped_questions_log.append(f"Q ID: {q_id} - Missing or empty 'correct_answer' key: '{question_text}'")
                 continue
            if not correct_answer_text or str(correct_answer_text).strip() == '': 
                skipped_questions_log.append(f"Q ID: {q_id} - Correct answer text for key '{correct_answer_key}' not found or empty in options: '{question_text}' - Options: {options_dict_raw}")
                continue
            
            # Final check: is the correct_answer_text actually one of the parsed options?
            if str(correct_answer_text).strip() not in parsed_options_list:
                skipped_questions_log.append(f"Q ID: {q_id} - Correct answer text '{correct_answer_text}' not found among valid options for question '{question_text}'. This usually means correct_answer_key ('{correct_answer_key}') is incorrect or points to an empty option.")
                continue

            # --- End Validation Checks ---

            if course not in quiz_bank:
                quiz_bank[course] = {}
            if level not in quiz_bank[course]:
                quiz_bank[course][level] = []
            
            quiz_bank[course][level].append({
                'id': q_id, 
                'question': question_text, 
                'options': parsed_options_list, 
                'answer': str(correct_answer_text).strip(), # This is the option text itself for matching
                'correct_key': correct_answer_key, # Original key 'A', 'B' for reference
                'topic_tags': topic_tags # NEW: Include topic tags
            })
            total_valid_questions += 1
        
        print(f"Quiz Bank '{os.path.basename(file_path)}' loaded: {len(quiz_bank)} courses, {total_valid_questions} valid questions processed (from {total_raw_entries} raw entries).")
        if skipped_questions_log:
            print(f"{len(skipped_questions_log)} questions were skipped in '{os.path.basename(file_path)}' due to validation errors:")
            for reason in skipped_questions_log:
                print(f"  - {reason}")
        
        # Log counts per course/level (only if there's actual data, avoid verbose for empty banks)
        if quiz_bank:
            print("Detailed breakdown of valid questions loaded per Course-Level:")
            for course, levels_data in quiz_bank.items():
                for level, questions_list in levels_data.items():
                    print(f"  - Course: '{course}', Level: '{level}': {len(questions_list)} questions.")
        return quiz_bank
    except FileNotFoundError:
        print(f"Error: Quiz JSON file not found at {file_path}. Please ensure the path is correct.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Critical Error loading quiz JSON from {file_path}: {type(e).__name__}: {e}")
        return {}

# --- Load All Models & Prepare Data (MOVED TO FUNCTION TO PREVENT DOUBLE EXECUTION) ---
def load_models_and_data():
    print("--- Loading all models and artifacts... ---")
    MODEL_BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Models'))
    
    # Initialize variables to None or empty defaults
    career_model_q, career_preprocessor_q, career_encoder_q = None, None, None
    course_df, course_similarity_matrix, course_knn_model, course_knn_scaler = None, None, None, None
    course_knn_names = [] # Ensure this is a list
    # --- Learning Roadmap Model artifacts ---
    roadmap_rf_classifier, roadmap_ohe, roadmap_le = None, None, None 
    # --- Quiz Analysis Model artifacts ---
    quiz_analysis_dt_classifier, quiz_analysis_ohe, quiz_analysis_le = None, None, None
    
    ml_generated_skills = {}
    
    main_quiz_bank_from_old_split_files = {} 
    practice_quiz_bank_from_old_split_files_aggr = {} # This will aggregate the old split practice files

    CORE_35_QUIZ_BANK = {} 
    NEW_PRACTICE_QUIZ_BANK = {} 
    QUIZ_QUESTIONS_COMBINED_FINAL = {} # final bank used by get_quiz_questions_for_course
    PRACTICE_QUIZ_COURSES_LIST = [] #correctly populated later (from NEW_PRACTICE_QUIZ_BANK)
    
    # NEW: Load the CORE_COURSE_TOPICS from quiz_topics_core_35courses.json
    global CORE_COURSE_TOPICS_FROM_FILE # Declare global to modify the module-level variable
    try:
        topic_file_path = os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/quiz_topics_core_35courses.json')
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            CORE_COURSE_TOPICS_FROM_FILE = json.load(f)
        print(f"CORE_COURSE_TOPICS loaded successfully from {os.path.basename(topic_file_path)}.")
    except FileNotFoundError:
        print(f"Error: quiz_topics_core_35courses.json not found at {topic_file_path}. Roadmap generation for 35 courses might be limited.")
        CORE_COURSE_TOPICS_FROM_FILE = {} # Fallback to empty dict
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in quiz_topics_core_35courses.json: {e}")
        CORE_COURSE_TOPICS_FROM_FILE = {}
    except Exception as e:
        print(f"Critical Error loading quiz_topics_core_35courses.json: {type(e).__name__}: {e}")
        CORE_COURSE_TOPICS_FROM_FILE = {}

    # NEW: Load the PRACTICE_QUIZ_TOPICS_FROM_FILE
    global PRACTICE_QUIZ_TOPICS_FROM_FILE
    try:
        practice_topic_file_path = os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/prac_quiz/quiz_topics_practice_courses.json')
        with open(practice_topic_file_path, 'r', encoding='utf-8') as f:
            PRACTICE_QUIZ_TOPICS_FROM_FILE = json.load(f)
        print(f"PRACTICE_QUIZ_TOPICS loaded successfully from {os.path.basename(practice_topic_file_path)}.")
    except FileNotFoundError:
        print(f"Error: quiz_topics_practice_courses.json not found at {practice_topic_file_path}. Roadmaps for practice quizzes might be limited.")
        PRACTICE_QUIZ_TOPICS_FROM_FILE = {}
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in quiz_topics_practice_courses.json: {e}")
        PRACTICE_QUIZ_TOPICS_FROM_FILE = {}
    except Exception as e:
        print(f"Critical Error loading quiz_topics_practice_courses.json: {type(e).__name__}: {e}")
        PRACTICE_QUIZ_TOPICS_FROM_FILE = {}


    try:
        # Career Recommender
        career_model_q = joblib.load(os.path.join(MODEL_BASE_PATH, 'Career_Recommendation/new_model_multistage_tuned.joblib'))
        career_preprocessor_q = joblib.load(os.path.join(MODEL_BASE_PATH, 'Career_Recommendation/new_preprocessor.joblib'))
        career_encoder_q = joblib.load(os.path.join(MODEL_BASE_PATH, 'Career_Recommendation/new_label_encoder.joblib'))
        print("Career Recommender (ML) loaded.")
        
        # Course Recommender
        course_df = joblib.load(os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/courses_processed.joblib'))
        course_similarity_matrix = joblib.load(os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/similarity_matrix.joblib'))
        course_knn_model = joblib.load(os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/knn_model.joblib'))
        course_knn_scaler = joblib.load(os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/knn_scaler.joblib'))
        course_knn_names_loaded = joblib.load(os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/knn_course_names.joblib'))
        # Ensure course_knn_names is always a list for consistent boolean evaluation
        if isinstance(course_knn_names_loaded, (pd.Series, pd.DataFrame)):
            course_knn_names = course_knn_names_loaded.tolist()
        elif course_knn_names_loaded is not None:
            course_knn_names = list(course_knn_names_loaded) # Ensure it's a list
        else:
            course_knn_names = [] # Default to empty list
            
        print("Course Recommender (ML Models) loaded.")
        
        # Quiz Analysis Model (Updated variable names to avoid conflict)
        quiz_analysis_dt_classifier = joblib.load(os.path.join(MODEL_BASE_PATH, 'Quiz_Analysis/model.joblib'))
        quiz_analysis_ohe = joblib.load(os.path.join(MODEL_BASE_PATH, 'Quiz_Analysis/one_hot_encoder.joblib'))
        quiz_analysis_le = joblib.load(os.path.join(MODEL_BASE_PATH, 'Quiz_Analysis/label_encoder.joblib'))
        print("Quiz Analysis Model loaded.")

        # Learning Roadmap Model (NEWLY LOADED)
        roadmap_rf_classifier = joblib.load(os.path.join(MODEL_BASE_PATH, 'Learning_Roadmap/model.joblib'))
        roadmap_ohe = joblib.load(os.path.join(MODEL_BASE_PATH, 'Learning_Roadmap/one_hot_encoder.joblib'))
        roadmap_le = joblib.load(os.path.join(MODEL_BASE_PATH, 'Learning_Roadmap/label_encoder.joblib'))
        print("Learning Roadmap Model loaded.")
        
        # Skill Recommender
        ml_generated_skills = joblib.load(os.path.join(MODEL_BASE_PATH, 'Skill_Recommendation/ml_generated_skills.joblib'))
        print("ML-Generated Skill list loaded.")

        # --- Load NEW CORE 35 QUIZ BANK ---
        core_35_quiz_path = os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/quiz_data_core_35courses.json')
        CORE_35_QUIZ_BANK = load_quiz_questions(core_35_quiz_path)

        # --- Load old user-added quiz split files for remaining courses ---
        user_added_quiz_split_files = [
            os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/user_added_courses/quiz_data_user_added_courses_part1_uniqueids.json'),
            os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/user_added_courses/quiz_data_user_added_courses_part2_uniqueids.json'),
            os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/user_added_courses/quiz_data_user_added_courses_part3_uniqueids.json'),
            os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/user_added_courses/quiz_data_user_added_courses_part4_uniqueids.json'),
        ]
        # Aggregate questions from all old user-added files
        for file_path in user_added_quiz_split_files:
            part_bank = load_quiz_questions(file_path)
            for course, levels_data in part_bank.items():
                if course not in main_quiz_bank_from_old_split_files:
                    main_quiz_bank_from_old_split_files[course] = {}
                for level, questions_list in levels_data.items():
                    if level not in main_quiz_bank_from_old_split_files[course]:
                        main_quiz_bank_from_old_split_files[course][level] = []
                    main_quiz_bank_from_old_split_files[course][level].extend(questions_list)

        # --- Load NEW single practice quiz file (replaces old split practice files) ---
        new_practice_quiz_path = os.path.join(MODEL_BASE_PATH, 'Course_Recommendation/quiz_mcq_data/split_data/prac_quiz/quiz_data_practice_courses.json')
        NEW_PRACTICE_QUIZ_BANK = load_quiz_questions(new_practice_quiz_path)

        QUIZ_QUESTIONS_COMBINED_FINAL = {}
        
        # Aggregate from user-added split files
        for course, levels_data in main_quiz_bank_from_old_split_files.items():
            QUIZ_QUESTIONS_COMBINED_FINAL[course] = {}
            for level, questions_list in levels_data.items():
                QUIZ_QUESTIONS_COMBINED_FINAL[course][level] = questions_list
        
        # Aggregate from NEW single practice quiz file
        for course, levels_data in NEW_PRACTICE_QUIZ_BANK.items():
            if course not in QUIZ_QUESTIONS_COMBINED_FINAL:
                QUIZ_QUESTIONS_COMBINED_FINAL[course] = {}
            for level, questions_list in levels_data.items():
                # For practice quizzes, this new file takes precedence.
                # If there's overlap with user_added_split_files, this will overwrite.
                QUIZ_QUESTIONS_COMBINED_FINAL[course][level] = questions_list
        
        # 2. OVERRIDE / ADD questions for the priority courses with the CORE_35_QUIZ_BANK
        for course, levels_data in CORE_35_QUIZ_BANK.items():
            if course not in QUIZ_QUESTIONS_COMBINED_FINAL:
                QUIZ_QUESTIONS_COMBINED_FINAL[course] = {} # Ensure course entry exists
            for level, questions_list in levels_data.items():
                QUIZ_QUESTIONS_COMBINED_FINAL[course][level] = questions_list # Overwrite or add with high-quality questions

        # 3. Final deduplication by ID across all merged sources within QUIZ_QUESTIONS_COMBINED_FINAL
        for course in QUIZ_QUESTIONS_COMBINED_FINAL:
            for level in QUIZ_QUESTIONS_COMBINED_FINAL[course]:
                merged_questions = QUIZ_QUESTIONS_COMBINED_FINAL[course][level]
                final_unique_list = {} 
                for q in merged_questions:
                    if q.get('id') and q['id'] not in final_unique_list:
                        final_unique_list[q['id']] = q
                QUIZ_QUESTIONS_COMBINED_FINAL[course][level] = list(final_unique_list.values())

        # Populate PRACTICE_QUIZ_COURSES_LIST from the NEW single practice quiz file
        PRACTICE_QUIZ_COURSES_LIST = list(NEW_PRACTICE_QUIZ_BANK.keys())

        print(f"\nTotal courses in old user-added split files (aggregated): {len(main_quiz_bank_from_old_split_files)}")
        print(f"Total courses in NEW PRACTICE_QUIZ_BANK (single file): {len(NEW_PRACTICE_QUIZ_BANK)}")
        print(f"Total courses in CORE_35_QUIZ_BANK (new high-quality): {len(CORE_35_QUIZ_BANK)}")
        print(f"Total unique courses in FINAL COMBINED bank: {len(QUIZ_QUESTIONS_COMBINED_FINAL)}")
        print("\n--- All models and data loaded successfully. ---")
        
    except Exception as e:
        print(f"FATAL ERROR loading models/data: {type(e).__name__}: {e}")
        # Ensure all return values are initialized even on error
    
    return (career_model_q, career_preprocessor_q, career_encoder_q,
            course_df, course_similarity_matrix, course_knn_model, course_knn_scaler, course_knn_names,
            roadmap_rf_classifier, roadmap_ohe, roadmap_le,
            quiz_analysis_dt_classifier, quiz_analysis_ohe, quiz_analysis_le,
            ml_generated_skills,
            main_quiz_bank_from_old_split_files, NEW_PRACTICE_QUIZ_BANK, CORE_35_QUIZ_BANK, QUIZ_QUESTIONS_COMBINED_FINAL, PRACTICE_QUIZ_COURSES_LIST)

# --- Load models ONCE at module level ---
(career_model_q, career_preprocessor_q, career_encoder_q,
 course_df, course_similarity_matrix, course_knn_model, course_knn_scaler, course_knn_names,
 roadmap_rf_classifier, roadmap_ohe, roadmap_le,
 quiz_analysis_dt_classifier, quiz_analysis_ohe, quiz_analysis_le,
 ml_generated_skills,
 MAIN_QUIZ_BANK_OLD_SPLIT_AGGR, NEW_PRACTICE_QUIZ_BANK_AGGR, CORE_35_QUIZ_BANK, QUIZ_QUESTIONS_COMBINED_FINAL, PRACTICE_QUIZ_COURSES_LIST) = load_models_and_data()

# --- Define the 35 priority courses for roadmap integration ---
# This list is now dynamically derived from CORE_COURSE_TOPICS_FROM_FILE keys
PRIORITY_ROADMAP_COURSES = set(CORE_COURSE_TOPICS_FROM_FILE.keys()) 

# NEW: Aggregate all roadmap-enabled courses for a master list
ROADMAP_ENABLED_COURSES = set(CORE_COURSE_TOPICS_FROM_FILE.keys())
ROADMAP_ENABLED_COURSES.update(PRACTICE_QUIZ_TOPICS_FROM_FILE.keys()) # Add courses from new practice topics file

CAREER_QUESTIONS = {
    'school': [{'name': 'logical', 'text': 'Logical & Analytical Thinking'}, {'name': 'creativity', 'text': 'Creativity'}, {'name': 'communication', 'text': 'Communication Skills'}, {'name': 'curiosity', 'text': 'Curiosity & Learning'}, {'name': 'leadership', 'text': 'Leadership'}, {'name': 'helping', 'text': 'Helping Nature'}, {'name': 'technology', 'text': 'Interest in Technology'}, {'name': 'science', 'text': 'Interest in Science'}, {'name': 'commerce', 'text': 'Interest in Commerce'}, {'name': 'artistic', 'text': 'Artistic Sense'}],
    'intermediate': [{'name': 'math', 'text': 'Interest in Mathematics'}, {'name': 'physics', 'text': 'Interest in Physics'}, {'name': 'chemistry', 'text': 'Interest in Chemistry'}, {'name': 'biology', 'text': 'Interest in Biology'}, {'name': 'cs', 'text': 'Interest in Computer Science'}, {'name': 'economics', 'text': 'Interest in Economics'}, {'name': 'psychology', 'text': 'Interest in Psychology/Sociology'}, {'name': 'problem_solving', 'text': 'Problem-Solving Skills'}, {'name': 'creativity', 'text': 'Creativity and Design'}, {'name': 'presentation', 'text': 'Communication & Presentation'}],
    'btech': [{'name': 'programming', 'text': 'Programming Skills'}, {'name': 'problem_solving', 'text': 'Problem Solving & Debugging'}, {'name': 'ai_ml', 'text': 'Interest in AI/ML'}, {'name': 'web_dev', 'text': 'Interest in Web/App Development'}, {'name': 'security', 'text': 'Interest in Networking & Security'}, {'name': 'core_eng', 'text': 'Interest in Core Engineering'}, {'name': 'research', 'text': 'Research & Innovation'}, {'name': 'communication', 'text': 'Communication Skills'}, {'name': 'teamwork', 'text': 'Teamwork & Collaboration'}, {'name': 'entrepreneurship', 'text': 'Entrepreneurship'}]
}

COURSE_QUESTIONS_BTECH = [{'name': 'data_analysis', 'text': 'Data Analysis & Visualization'}, {'name': 'ai_ml', 'text': 'AI & Machine Learning'}, {'name': 'web_dev', 'text': 'Web & App Development'}, {'name': 'cloud', 'text': 'Cloud Computing & DevOps'}, {'name': 'management', 'text': 'Project & Product Management'}]


# --- Helper Functions ---
# Modified generate_personalized_roadmap to use both CORE_COURSE_TOPICS_FROM_FILE and PRACTICE_QUIZ_TOPICS_FROM_FILE
def generate_personalized_roadmap(course, level, profile, incorrect_topic_tags=None):
    multipliers = {'Fast Learner': 0.75, 'Average Learner': 1.0, 'Thorough Learner': 1.25, 'Needs Review': 2.0}
    multiplier = multipliers.get(profile, 1.0)
    
    topics_data = []
    # Check if the course is in CORE_COURSE_TOPICS_FROM_FILE
    if course in CORE_COURSE_TOPICS_FROM_FILE and CORE_COURSE_TOPICS_FROM_FILE.get(course, {}).get(level):
        topics_data = CORE_COURSE_TOPICS_FROM_FILE.get(course, {}).get(level, [])
    # Else, check if the course is in PRACTICE_QUIZ_TOPICS_FROM_FILE
    elif course in PRACTICE_QUIZ_TOPICS_FROM_FILE and PRACTICE_QUIZ_TOPICS_FROM_FILE.get(course, {}).get(level):
        topics_data = PRACTICE_QUIZ_TOPICS_FROM_FILE.get(course, {}).get(level, [])
    
    if not topics_data: 
        return [] # Explicitly return empty list if no topics are defined for roadmap-enabled courses

    personalized_roadmap = []
    # Convert incorrect_topic_tags to a set for faster lookup
    incorrect_topic_tags_set = set(incorrect_topic_tags) if incorrect_topic_tags else set()

    for topic, base_days in topics_data:
        p_days = max(1, round(base_days * multiplier))
        
        # Adjust days further if the topic was in an incorrectly answered question
        # This assumes topic_tags in quiz_data match topic_name in CORE_COURSE_TOPICS_FROM_FILE or PRACTICE_QUIZ_TOPICS_FROM_FILE
        if topic in incorrect_topic_tags_set:
            p_days = max(1, round(p_days * 1.5)) # Increase estimated time for struggled topics by 50%

        day_str = f"{p_days} day{'s' if p_days > 1 else ''}"
        personalized_roadmap.append((topic, day_str))
    
    # Sort roadmap to prioritize struggled topics first (if any)
    if incorrect_topic_tags_set:
        personalized_roadmap.sort(key=lambda x: (x[0] not in incorrect_topic_tags_set, -int(x[1].split(' ')[0])))
        # x[0] not in incorrect_topic_tags_set: True (not a struggled topic) will come after False (is a struggled topic)
        # -int(x[1].split(' ')[0]): Sort by days descending for topics with same struggle status

    return personalized_roadmap


def get_quiz_questions_for_course(course_name, difficulty, num_questions=15):
    """
    Get unique questions for a specific course and difficulty level, up to num_questions.
    Questions are drawn from the QUIZ_QUESTIONS_COMBINED_FINAL bank, using 'id' for uniqueness.
    """
    
    all_potential_questions_for_level = QUIZ_QUESTIONS_COMBINED_FINAL.get(course_name, {}).get(difficulty, [])
    
    # Ensure uniqueness based on question 'id' (final pass, just in case merge had issues)
    unique_questions_map = {} 
    for q in all_potential_questions_for_level:
        q_id = q.get('id')
        if q_id and q_id not in unique_questions_map: # q_id should always be present due to load_quiz_questions
            unique_questions_map[q_id] = q
    
    final_unique_questions = list(unique_questions_map.values())

    # Shuffle the unique questions
    random.shuffle(final_unique_questions)
    
    # Return a sample of num_questions, or all available unique questions if fewer
    return final_unique_questions[:num_questions]


# --- Predict Learning Profile (Using the Quiz Analysis Model) ---
def predict_learning_profile(score_percentage: float, time_taken_minutes: float, difficulty_level: str) -> str:
    # Ensure quiz_analysis_dt_classifier, quiz_analysis_ohe, quiz_analysis_le are loaded
    if not all([quiz_analysis_dt_classifier, quiz_analysis_ohe, quiz_analysis_le]):
        print("Quiz Analysis Model not loaded. Falling back to rule-based profile.")
        # Fallback to rule-based if model not available
        if score_percentage >= 80: return 'Fast Learner'
        elif score_percentage >= 60: return 'Average Learner'
        elif score_percentage >= 40: return 'Thorough Learner'
        else: return 'Needs Review'

    try:
        # Create a DataFrame with all original features used during training
        input_data_df = pd.DataFrame({
            'score_percentage': [score_percentage],
            'time_taken_minutes': [time_taken_minutes],
            'difficulty_level': [difficulty_level]
        })
        
        # One-hot encode the 'difficulty_level' column.
        categorical_features_for_ohe = ['difficulty_level']
        
        encoded_difficulty_array = quiz_analysis_ohe.transform(input_data_df[categorical_features_for_ohe]).toarray()
        # Use get_feature_names_out with correct input for columns
        encoded_difficulty_df = pd.DataFrame(encoded_difficulty_array, 
                                             columns=quiz_analysis_ohe.get_feature_names_out(categorical_features_for_ohe)) 
        
        numerical_features = input_data_df.drop(columns=categorical_features_for_ohe)
        
        # This final processed_input must match the columns AND order of X_processed during training
        expected_ohe_cols = quiz_analysis_ohe.get_feature_names_out(categorical_features_for_ohe)
        expected_columns_order = ['score_percentage', 'time_taken_minutes'] + list(expected_ohe_cols)
        
        processed_input_for_model = pd.concat([
            numerical_features.reset_index(drop=True),
            encoded_difficulty_df.reset_index(drop=True)
        ], axis=1)

        # Reindex to ensure correct order and fill any missing columns with 0
        processed_input_for_model = processed_input_for_model.reindex(columns=expected_columns_order, fill_value=0)

        # Predict learning profile
        prediction_encoded = quiz_analysis_dt_classifier.predict(processed_input_for_model)
        profile = quiz_analysis_le.inverse_transform(prediction_encoded)[0]
        return profile
    except Exception as e:
        print(f"Error predicting learning profile with model: {e}. Falling back to rule-based.")
        # Fallback to rule-based on model error
        if score_percentage >= 80: return 'Fast Learner'
        elif score_percentage >= 60: return 'Average Learner'
        elif score_percentage >= 40: return 'Thorough Learner'
        else: return 'Needs Review'

# --- Predict Recommended Next Step (Using the Learning Roadmap Model) ---
def predict_recommended_next_step(current_knowledge_level: str, learning_profile: str) -> str:
    # Ensure roadmap_rf_classifier, roadmap_ohe, roadmap_le are loaded
    if not all([roadmap_rf_classifier, roadmap_ohe, roadmap_le]):
        print("Learning Roadmap Model not loaded. Falling back to default next step.")
        return "Review foundational concepts and practice problems." # Default fallback

    try:
        # Create input for the roadmap model. This DataFrame must match the structure of X used in training.
        input_data_df = pd.DataFrame({
            'current_knowledge_level': [current_knowledge_level],
            'learning_profile': [learning_profile]
        })
        
        # One-hot encode features using the roadmap model's OHE.
        processed_input = roadmap_ohe.transform(input_data_df)
        
        # Predict recommended next step
        prediction_encoded = roadmap_rf_classifier.predict(processed_input)
        recommended_step = roadmap_le.inverse_transform(prediction_encoded)[0]
        return recommended_step
    except Exception as e:
        print(f"Error predicting recommended next step with model: {e}. Falling back to default.")
        return "Review foundational concepts and practice problems." # Default fallback


# --- User Auth Routes ---
@app.route('/')
def home(): 
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: 
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        else: 
            flash('Login Unsuccessful. Please check credentials.', 'danger')
    return render_template('index.html', page='login', question_bank={})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: 
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        existing_user = User.query.filter_by(username=request.form['username']).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'warning')
            return render_template('index.html', page='register', question_bank={})
        hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        new_user = User(username=request.form['username'], password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('index.html', page='register', question_bank={})

@app.route('/logout')
def logout(): 
    logout_user()
    return redirect(url_for('login'))

# --- Main Feature Routes ---
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', page='dashboard', question_bank={})

@app.route('/career_recommender', methods=['GET', 'POST'])
@login_required
def career_recommender():
    if not all([career_model_q, career_preprocessor_q, career_encoder_q]):
        flash('Career recommender model not loaded properly.', 'danger')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        try:
            stage = request.form.get('stage')
            input_data = {'stage': stage}
            all_q_names = sorted(list(set(q['name'] for questions in CAREER_QUESTIONS.values() for q in questions)))
            for q_name in all_q_names: 
                input_data[q_name] = float(request.form.get(q_name, 0))
            input_df = pd.DataFrame([input_data])
            feature_cols_ordered = ['stage'] + all_q_names
            input_df = input_df.reindex(columns=feature_cols_ordered, fill_value=0)
            input_processed = career_preprocessor_q.transform(input_df)
            prediction_id = career_model_q.predict(input_processed)[0]
            result = career_encoder_q.inverse_transform([prediction_id])[0]
            return render_template('index.html', page='simple_result', result_type='career', title="Based on your answers, we recommend:", results=result, back_url=url_for('career_recommender'), question_bank={})
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
            print(f"Career Prediction Error: {e}")
            return redirect(url_for('career_recommender'))
    stage = request.args.get('stage')
    if not CAREER_QUESTIONS: 
        flash('Career questions not defined.', 'danger')
        return redirect(url_for('dashboard'))
    if stage in CAREER_QUESTIONS: 
        return render_template('index.html', page='career_questionnaire', stage=stage, questions=CAREER_QUESTIONS[stage], question_bank={})
    return render_template('index.html', page='career_stage_select', question_bank={})

@app.route('/course_recommender', methods=['GET', 'POST'])
@login_required
def course_recommender():
    if not all([course_knn_model, course_knn_scaler, course_knn_names is not None and len(course_knn_names) > 0]):
        flash('Course interest model not loaded or course names list is empty.', 'danger')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        try:
            scores = [int(request.form.get(q['name'], 0)) for q in COURSE_QUESTIONS_BTECH]
            user_scores_scaled = course_knn_scaler.transform([scores])
            distances, indices = course_knn_model.kneighbors(user_scores_scaled, n_neighbors=5)
            recommendations = [course_knn_names[i] for i in indices[0]]
            if not recommendations: 
                flash('No courses found matching your interests.', 'info')
                return redirect(url_for('course_recommender'))
            return render_template('index.html', page='course_results', results=recommendations, question_bank={})
        except Exception as e:
            flash(f"An error occurred: {e}", 'danger')
            return redirect(url_for('course_recommender'))
    if not COURSE_QUESTIONS_BTECH: 
        flash('Course questions not defined.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('index.html', page='course_questionnaire', questions=COURSE_QUESTIONS_BTECH, question_bank={})

@app.route('/find_similar_courses/<path:course_name>')
@login_required
def find_similar_courses(course_name):
    if course_df is None or course_similarity_matrix is None:
        flash('Course similarity model not loaded.', 'danger')
        return redirect(url_for('course_recommender'))
    try:
        course_index = course_df[course_df['Course Name'] == course_name].index[0]
        distances = course_similarity_matrix[course_index]
        courses_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        results = [course_df.iloc[i[0]]['Course Name'] for i in courses_list]
        return render_template('index.html', page='course_results', title=f"Courses similar to '{course_name}':", results=results, back_url=url_for('course_recommender'), question_bank={})
    except IndexError:
        flash(f"Could not find '{course_name}' to find similarities.", 'warning')
        return redirect(url_for('course_recommender'))
    except Exception as e:
        flash(f"An error occurred: {e}", 'danger')
        return redirect(url_for('course_recommender'))

@app.route('/add_course_to_profile', methods=['POST'])
@login_required
def add_course_to_profile():
    count = 0
    for course_name in request.form.getlist('selected_courses'):
        try:
            existing_entry = UserCourse.query.filter_by(user_id=current_user.id, course_name=course_name).first()
            if not existing_entry:
                db.session.add(UserCourse(user_id=current_user.id, course_name=course_name))
                count += 1
                
                # NEW: Flash a warning if the added course is not a roadmap-enabled course
                if course_name not in ROADMAP_ENABLED_COURSES:
                    flash(f'Note: Roadmap features for "{course_name}" are still under development. Quizzes may contain repeated questions, and personalized roadmaps might not be available yet. This will be improved in future enhancements.', 'info')

        except Exception as e:
            print(f"Error checking/adding course {course_name}: {e}")

    try:
        if count > 0: 
            db.session.commit()
            flash(f'{count} course(s) added!', 'success')
        else: 
            flash('No new courses selected or already in profile.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding courses: {e}', 'danger')
        print(f"DB Commit Error: {e}")
    return redirect(url_for('quiz_analysis'))

@app.route('/skill_recommender', methods=['GET', 'POST'])
@login_required
def skill_recommender():
    if ml_generated_skills is None:
        flash('ML-generated skill list not loaded.', 'danger')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        job_title = request.form.get('job_title')
        skills = ml_generated_skills.get(job_title, [])
        if not skills: 
            flash(f"Could not find skills for '{job_title}'.", 'danger')
            return redirect(url_for('skill_recommender'))
        return render_template('index.html', page='simple_result', result_type='skill', title=f"Top skills for a {job_title}:", results=skills, back_url=url_for('skill_recommender'), job_title_to_save=job_title, question_bank={})
    return render_template('index.html', page='skill_form', job_titles=ml_generated_skills.keys(), question_bank={})

@app.route('/quiz_analysis', methods=['GET', 'POST'])
@login_required
def quiz_analysis():
    course_name = request.args.get('course_name')
    
    # Handle quiz submission (POST)
    if request.method == 'POST' and course_name:
        try:
            difficulty = request.form.get('difficulty')
            start_time = float(request.form.get('start_time', time.time()))
            time_taken = round((time.time() - start_time) / 60, 2)  # minutes
            
            # Fetch the exact set of questions that were displayed to the user for scoring
            questions_for_scoring = get_quiz_questions_for_course(course_name, difficulty, num_questions=15)
            
            if not questions_for_scoring:
                flash(f'No valid questions were available for {course_name} at {difficulty} level during submission. Please contact support if this persists.', 'warning')
                return redirect(url_for('quiz_analysis', course_name=course_name))
            
            # Calculate score and identify incorrect topic tags
            correct_answers = 0
            incorrect_topic_tags = set()
            total_questions = len(questions_for_scoring)
            
            for idx, q in enumerate(questions_for_scoring):
                user_answer = request.form.get(f'q{idx+1}')
                if user_answer == q['answer']:  # Compare with the full answer text
                    correct_answers += 1
                else:
                    # Collect topic tags from incorrectly answered questions
                    if q.get('topic_tags'):
                        incorrect_topic_tags.update(q['topic_tags'])
            
            score = correct_answers
            score_percent = round((score / total_questions) * 100) if total_questions > 0 else 0
            
            # --- Roadmap Model Integration ---
            # 1. Determine learning_profile (using model or rule-based)
            learning_profile_predicted = predict_learning_profile(score_percent, time_taken, difficulty)

            # 2. Determine current_knowledge_level (from quiz difficulty)
            current_knowledge_level = difficulty # e.g., 'Basic', 'Intermediate', 'Advanced'

            # 3. Predict high-level recommended_next_step using the Learning_Roadmap model
            recommended_next_step = predict_recommended_next_step(current_knowledge_level, learning_profile_predicted)
            
            # 4. Generate topic-specific roadmap using COURSE_TOPICS_FROM_FILE
            # Determine if this course is one of the roadmap-enabled courses
            is_roadmap_available_for_course = course_name in ROADMAP_ENABLED_COURSES
                                              
            if is_roadmap_available_for_course:
                roadmap_details = generate_personalized_roadmap(course_name, difficulty, learning_profile_predicted, list(incorrect_topic_tags))
            else:
                roadmap_details = [] # Empty if no roadmap available for this course (as per discussion)
            
            roadmap_details_json_str = json.dumps(roadmap_details) # Convert to JSON string for storage
            
            flash(f'Quiz completed! Score: {score_percent}% - Profile: {learning_profile_predicted}', 'success')
            
            # Save quiz result to database (UPDATED to include new fields)
            quiz_result = QuizResult(
                user_id=current_user.id,
                course_name=course_name,
                level=difficulty,
                score=score,
                total_questions=total_questions,
                time_taken=time_taken,
                learning_profile_predicted=learning_profile_predicted,
                recommended_next_step=recommended_next_step,
                roadmap_details_json=roadmap_details_json_str
            )
            db.session.add(quiz_result)
            db.session.commit()

            return render_template(
                'index.html',
                page='quiz_result_and_roadmap',
                course_name=course_name,
                score=score_percent,
                profile=learning_profile_predicted, # Use predicted profile
                recommended_next_step=recommended_next_step, # Pass model's next step
                roadmap=roadmap_details, # Pass actual details for display
                difficulty=difficulty,
                time_taken=time_taken,
                quiz_result_id=quiz_result.id, # Pass quiz_result_id for potential saving
                is_roadmap_available_for_course=is_roadmap_available_for_course, # Pass this flag
                question_bank={}
            )
            
        except Exception as e:
            flash(f'Error processing quiz: {str(e)}', 'danger')
            print(f"Quiz processing error: {e}")
            return redirect(url_for('quiz_analysis', course_name=course_name))
    
    # Handle quiz form display (GET with course_name)
    if course_name:
        # Get available levels for this course (from QUIZ_QUESTIONS_COMBINED_FINAL)
        available_levels_raw = list(QUIZ_QUESTIONS_COMBINED_FINAL.get(course_name, {}).keys())
        
        available_levels = list(set(available_levels_raw))  # Remove duplicates

        # Define desired order for levels
        desired_order = ["Basic", "Intermediate", "Advanced"]
        
        # Sort available_levels according to the desired_order
        available_levels.sort(key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order))
        
        # Prepare question bank structure for JavaScript:
        question_bank_for_js = {}
        total_questions_found_across_levels = 0
        for level in available_levels:
            questions_for_level = get_quiz_questions_for_course(course_name, level, num_questions=15)
            question_bank_for_js[level] = questions_for_level
            total_questions_found_across_levels += len(questions_for_level)

        # Determine if any valid questions exist for this course across all levels
        has_any_questions = total_questions_found_across_levels > 0

        # Redirect if no questions available for this course AT ALL after all processing
        if not has_any_questions:
            flash(f'No valid quiz questions found for "{course_name}" across any difficulty level after processing. Please ensure your JSON data is correctly formatted and contains questions, and check console logs.', 'warning')
            return redirect(url_for('quiz_analysis'))

        # NEW: Determine if this course is a roadmap-enabled course for the warning message
        is_roadmap_enabled_course = course_name in ROADMAP_ENABLED_COURSES
        # This flag is used to show a warning *before* starting the quiz if roadmap is not enabled.
        

        return render_template(
            'index.html',
            page='quiz_form',
            course_name=course_name,
            question_bank=question_bank_for_js, # Pass the processed bank to JS
            available_levels=available_levels,
            start_time=time.time(),
            is_practice_quiz=course_name in PRACTICE_QUIZ_COURSES_LIST, 
            is_roadmap_enabled_course=is_roadmap_enabled_course, # NEW: Pass this flag to the template
            has_any_questions=has_any_questions 
        )
    
    # Main selection page: user's added courses and practice quizzes
    user_courses = UserCourse.query.filter_by(user_id=current_user.id).all()
    
    # Filter user courses that have quizzes available (check combined bank for any questions)
    user_courses_with_quizzes = []
    for course_obj in user_courses:
        # Check if the course has any levels and questions in the combined bank after parsing
        if QUIZ_QUESTIONS_COMBINED_FINAL.get(course_obj.course_name): # Use QUIZ_QUESTIONS_COMBINED_FINAL
            user_courses_with_quizzes.append(course_obj)
    
    # Calculate total unique courses available for quizzes (for display in quiz_selection)
    total_quiz_courses_available = len(QUIZ_QUESTIONS_COMBINED_FINAL) # Use QUIZ_QUESTIONS_COMBINED_FINAL

    return render_template(
        'index.html',
        page='quiz_selection',
        user_courses=user_courses_with_quizzes,
        practice_quizzes=PRACTICE_QUIZ_COURSES_LIST, # Use PRACTICE_QUIZ_COURSES_LIST
        total_quiz_courses=total_quiz_courses_available, 
        question_bank={}
    )

@app.route('/save_career', methods=['POST'])
@login_required
def save_career():
    career = request.form.get('result_to_save')
    if career and not SavedCareer.query.filter_by(user_id=current_user.id, career_domain=career).first():
        db.session.add(SavedCareer(user_id=current_user.id, career_domain=career))
        db.session.commit()
        flash(f'Career saved!', 'success')
    else: 
        flash(f'Career "{career}" is already saved or invalid.', 'info')
    return redirect(url_for('my_data'))

@app.route('/save_skills', methods=['POST'])
@login_required
def save_skills():
    job_title = request.form.get('job_title_to_save')
    skills_string = request.form.get('skills_to_save')
    if job_title and skills_string and not SavedSkill.query.filter_by(user_id=current_user.id, job_title=job_title).first():
        db.session.add(SavedSkill(user_id=current_user.id, job_title=job_title, skills_list=skills_string))
        db.session.commit()
        flash(f'Skills for "{job_title}" saved!', 'success')
    else: 
        flash(f'Skills for "{job_title}" are already saved or invalid.', 'info')
    return redirect(url_for('my_data'))

@app.route('/save_roadmap', methods=['POST']) # NEW ROUTE to save roadmap explicitly
@login_required
def save_roadmap():
    quiz_result_id = request.form.get('quiz_result_id')
    quiz_result = QuizResult.query.get(quiz_result_id)

    if not quiz_result:
        flash('Quiz result not found to save roadmap.', 'danger')
        return redirect(url_for('my_data'))
    
    # Check if roadmap already saved for this quiz result or if user already has identical roadmap
    existing_roadmap = UserRoadmap.query.filter_by(
        user_id=current_user.id,
        quiz_result_id=quiz_result.id
    ).first()

    if existing_roadmap:
        flash('This roadmap has already been saved.', 'info')
        return redirect(url_for('my_data'))

    # Create a new UserRoadmap entry
    new_roadmap = UserRoadmap(
        user_id=current_user.id,
        course_name=quiz_result.course_name,
        level=quiz_result.level,
        learning_profile_at_generation=quiz_result.learning_profile_predicted,
        recommended_next_step=quiz_result.recommended_next_step,
        roadmap_details_json=quiz_result.roadmap_details_json,
        quiz_result_id=quiz_result.id
    )
    db.session.add(new_roadmap)
    db.session.commit()
    flash(f'Roadmap for "{quiz_result.course_name}" saved to your profile!', 'success')
    return redirect(url_for('my_data'))

@app.route('/my_data')
@login_required
def my_data():
    saved_careers = SavedCareer.query.filter_by(user_id=current_user.id).order_by(SavedCareer.timestamp.desc()).all()
    saved_skills = SavedSkill.query.filter_by(user_id=current_user.id).order_by(SavedSkill.timestamp.desc()).all()
    quiz_results = QuizResult.query.filter_by(user_id=current_user.id).order_by(QuizResult.timestamp.desc()).all()
    user_roadmaps = UserRoadmap.query.filter_by(user_id=current_user.id).order_by(UserRoadmap.timestamp.desc()).all() # Fetch user's saved roadmaps
    
    for item in saved_skills: 
        item.skills_list = item.skills_list.split(',') if item.skills_list else []
    
    # Deserialize roadmap JSON for display in quiz_results
    for qr in quiz_results:
        if qr.roadmap_details_json:
            qr.roadmap_details = json.loads(qr.roadmap_details_json)
        else:
            qr.roadmap_details = []
    
    for ur in user_roadmaps:
        if ur.roadmap_details_json:
            ur.roadmap_details = json.loads(ur.roadmap_details_json)
        else:
            ur.roadmap_details = []

    return render_template('index.html', page='my_data', 
                            saved_careers=saved_careers, 
                            saved_skills=saved_skills, 
                            quiz_results=quiz_results, 
                            user_roadmaps=user_roadmaps, # Pass to template
                            question_bank={})

@app.route('/delete_saved_item/<item_type>/<int:item_id>', methods=['POST'])
@login_required
def delete_saved_item(item_type, item_id):
    item = None
    if item_type == 'career': 
        item = db.session.get(SavedCareer, item_id)
    elif item_type == 'skill': 
        item = db.session.get(SavedSkill, item_id)
    elif item_type == 'quiz': 
        item = db.session.get(QuizResult, item_id)
    elif item_type == 'roadmap': # Handle roadmap deletion
        item = db.session.get(UserRoadmap, item_id)
    
    if item and item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()
        flash('Item deleted.', 'success')
    else: 
        flash('Item not found or permission denied.', 'danger')
    return redirect(url_for('my_data'))

@app.route('/clear_my_data', methods=['POST'])
@login_required
def clear_my_data():
    SavedCareer.query.filter_by(user_id=current_user.id).delete()
    SavedSkill.query.filter_by(user_id=current_user.id).delete()
    QuizResult.query.filter_by(user_id=current_user.id).delete()
    UserRoadmap.query.filter_by(user_id=current_user.id).delete() # Clear user roadmaps
    db.session.commit()
    flash('All saved careers, skills, quiz results, and roadmaps cleared.', 'success')
    return redirect(url_for('my_data'))

@app.route('/delete_saved_course/<int:user_id>/<path:course_name>', methods=['POST'])
@login_required
def delete_saved_course(user_id, course_name):
    if user_id != current_user.id:
        flash('Permission denied to delete this course.', 'danger')
        return redirect(url_for('quiz_analysis'))

    course = UserCourse.query.filter_by(user_id=user_id, course_name=course_name).first() 
    
    if course: 
        db.session.delete(course)
        db.session.commit()
        flash('Course removed.', 'success')
    else: 
        flash('Course not found.', 'danger')
    return redirect(url_for('quiz_analysis'))

@app.route('/clear_my_courses', methods=['POST'])
@login_required
def clear_my_courses():
    UserCourse.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('All saved courses cleared.', 'success')
    return redirect(url_for('quiz_analysis'))

# --- FIXED: Main execution block ---
if __name__ == '__main__':
    # Create database tables within app context
    with app.app_context():
        db.create_all()
        print("\nDatabase tables created successfully!")
        
        # Verify all key tables exist
        inspector = inspect(db.engine)
        tables = ['user', 'user_course', 'quiz_result', 'user_roadmap', 'saved_career', 'saved_skill'] # Added user_roadmap
        all_tables_exist = True
        for table in tables:
            if inspector.has_table(table):
                print(f"{table} table exists")
            else:
                print(f"{table} table missing")
                all_tables_exist = False
        
        if all_tables_exist:
            print("All database tables ready!")
        else:
            print("Some database tables need attention")
    
    print("\nStarting Flask application...")
    print("Access your app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)