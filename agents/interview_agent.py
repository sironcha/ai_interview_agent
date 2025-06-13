"""
AI Interview Agent - Core LLM logic for generating questions and evaluating responses
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

# LLM integrations - choose your preferred service
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewAgent:
    """
    Core AI Interview Agent class that handles:
    - Document analysis and processing
    - Personalized question generation
    - Response evaluation and scoring
    - Final report generation
    """
    
    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize the Interview Agent
        
        Args:
            llm_provider: Choose from 'openai', 'anthropic', 'google', or 'mock'
        """
        self.job_post = ""
        self.company_profile = ""
        self.candidate_resume = ""
        self.interview_questions = []
        self.current_question_idx = 0
        self.responses = []
        self.interview_started = False
        self.interview_completed = False
        self.final_report = None
        
        # LLM configuration
        self.llm_provider = llm_provider
        self._setup_llm_client()
        
        # Analysis cache to avoid re-processing
        self._document_analysis = {}
        
    def _setup_llm_client(self):
        """Setup the LLM client based on provider"""
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            try:
                # Try .env first, then streamlit secrets
                api_key = self._get_api_key("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = openai.OpenAI(api_key=api_key)
                else:
                    logger.warning("OpenAI API key not found, using mock responses")
                    self.llm_provider = "mock"
            except Exception as e:
                logger.error(f"Failed to setup OpenAI client: {e}")
                self.llm_provider = "mock"
                
        elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            try:
                # Try .env first, then streamlit secrets
                api_key = self._get_api_key("ANTHROPIC_API_KEY")
                if api_key:
                    self.llm_client = anthropic.Anthropic(api_key=api_key)
                else:
                    logger.warning("Anthropic API key not found, using mock responses")
                    self.llm_provider = "mock"
            except Exception as e:
                logger.error(f"Failed to setup Anthropic client: {e}")
                self.llm_provider = "mock"
                
        elif self.llm_provider == "google" and GOOGLE_AVAILABLE:
            try:
                # Try .env first, then streamlit secrets
                api_key = self._get_api_key("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.llm_client = genai.GenerativeModel('gemini-pro')
                else:
                    logger.warning("Google API key not found, using mock responses")
                    self.llm_provider = "mock"
            except Exception as e:
                logger.error(f"Failed to setup Google client: {e}")
                self.llm_provider = "mock"
        else:
            logger.info("Using mock LLM responses for demo")
            self.llm_provider = "mock"
    
    def _get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key from .env file or environment variables, with fallback to Streamlit secrets
        
        Args:
            key_name: Name of the API key environment variable
            
        Returns:
            API key string or None if not found
        """
        import os
        
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # dotenv not installed, skip .env loading
            pass
        
        # Try environment variable first
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
        
        # Fallback to Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets:
                return st.secrets.get(key_name)
        except ImportError:
            pass
        
        return None
    
    def _call_llm(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """
        Call the configured LLM service
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (if supported)
            temperature: Response creativity (0.0 to 1.0)
            
        Returns:
            LLM response text
        """
        try:
            if self.llm_provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif self.llm_provider == "anthropic":
                full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:" if system_prompt else prompt
                
                response = self.llm_client.completions.create(
                    model="claude-3-sonnet-20240229",
                    prompt=full_prompt,
                    max_tokens_to_sample=2000,
                    temperature=temperature
                )
                return response.completion
                
            elif self.llm_provider == "google":
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.llm_client.generate_content(full_prompt)
                return response.text
                
            else:
                # Mock responses for demo
                return self._generate_mock_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_mock_response(prompt)
    
    def analyze_documents(self) -> Dict:
        """
        Analyze loaded documents to extract key information
        
        Returns:
            Dictionary containing analysis of job requirements, company culture, candidate profile
        """
        if self._document_analysis:
            return self._document_analysis
        
        analysis_prompt = f"""
        Analyze the following documents and extract key information:
        
        JOB POSTING:
        {self.job_post}
        
        COMPANY PROFILE:
        {self.company_profile}
        
        CANDIDATE RESUME:
        {self.candidate_resume}
        
        Extract and return as JSON:
        {{
            "job_analysis": {{
                "title": "extracted job title",
                "key_skills": ["skill1", "skill2", "skill3"],
                "experience_level": "entry/mid/senior",
                "technical_requirements": ["req1", "req2"],
                "soft_skills": ["skill1", "skill2"],
                "responsibilities": ["resp1", "resp2"]
            }},
            "company_analysis": {{
                "industry": "industry name",
                "company_size": "startup/small/medium/large",
                "values": ["value1", "value2", "value3"],
                "culture_keywords": ["keyword1", "keyword2"],
                "work_environment": "remote/hybrid/onsite"
            }},
            "candidate_analysis": {{
                "experience_years": "number or range",
                "technical_skills": ["skill1", "skill2"],
                "education": "education background",
                "previous_roles": ["role1", "role2"],
                "achievements": ["achievement1", "achievement2"],
                "career_progression": "assessment of growth",
                "skill_gaps": ["gap1", "gap2"],
                "strengths": ["strength1", "strength2"]
            }},
            "fit_assessment": {{
                "technical_match": "percentage",
                "experience_match": "percentage", 
                "cultural_match": "percentage",
                "overall_match": "percentage",
                "red_flags": ["flag1", "flag2"],
                "green_flags": ["flag1", "flag2"]
            }}
        }}
        """
        
        system_prompt = """You are an expert HR analyst and technical interviewer. 
        Analyze documents thoroughly and provide detailed, accurate assessments. 
        Be objective and identify both strengths and areas of concern."""
        
        try:
            response = self._call_llm(analysis_prompt, system_prompt)
            # Clean up response to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                self._document_analysis = json.loads(json_str)
            else:
                self._document_analysis = self._get_mock_analysis()
        except json.JSONDecodeError:
            logger.error("Failed to parse document analysis JSON")
            self._document_analysis = self._get_mock_analysis()
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            self._document_analysis = self._get_mock_analysis()
        
        return self._document_analysis
    
    def generate_interview_questions(self, num_questions: int = 8, 
                                   categories: List[str] = None, 
                                   difficulty: str = "Mid",
                                   focus_areas: List[str] = None) -> bool:
        """
        Generate personalized interview questions based on document analysis
        
        Args:
            num_questions: Number of questions to generate
            categories: Question categories to include
            difficulty: Difficulty level (Entry/Mid/Senior)
            focus_areas: Specific areas to focus on
            
        Returns:
            True if successful, False otherwise
        """
        if not all([self.job_post, self.company_profile, self.candidate_resume]):
            logger.error("Missing required documents for question generation")
            return False
        
        # Get document analysis
        analysis = self.analyze_documents()
        
        # Set defaults
        if not categories:
            categories = ["technical", "behavioral", "cultural", "experience"]
        
        if not focus_areas:
            focus_areas = ["Problem Solving", "Communication", "Technical Skills"]
        
        question_prompt = f"""
        Based on the following analysis, generate {num_questions} personalized interview questions:
        
        DOCUMENT ANALYSIS:
        {json.dumps(analysis, indent=2)}
        
        REQUIREMENTS:
        - Question categories: {categories}
        - Difficulty level: {difficulty}
        - Focus areas: {focus_areas}
        - Mix of question types for comprehensive assessment
        
        Generate questions that:
        1. Assess technical skills relevant to the job requirements
        2. Evaluate cultural fit with company values
        3. Explore candidate's experience and achievements
        4. Test problem-solving and critical thinking
        5. Assess communication and interpersonal skills
        
        Return as JSON array:
        [
            {{
                "id": 1,
                "question": "Detailed question text",
                "category": "technical/behavioral/cultural/experience/situational",
                "difficulty": "entry/mid/senior",
                "reasoning": "Why this question is relevant for this candidate/role",
                "key_points": ["point1", "point2", "point3"],
                "follow_up_hints": ["hint1", "hint2"],
                "evaluation_criteria": ["criteria1", "criteria2"],
                "ideal_response_elements": ["element1", "element2"]
            }}
        ]
        """
        
        system_prompt = f"""You are an expert technical interviewer and HR professional. 
        Generate thoughtful, relevant questions that will effectively assess the candidate's 
        fit for this specific role and company. Questions should be:
        - Specific to the role and candidate background
        - Progressive in difficulty
        - Designed to reveal both technical competence and cultural fit
        - Open-ended to encourage detailed responses
        - Professional and unbiased"""
        
        try:
            response = self._call_llm(question_prompt, system_prompt, temperature=0.8)
            
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                questions = json.loads(json_str)
                
                # Validate and enhance questions
                self.interview_questions = self._validate_questions(questions)
                logger.info(f"Generated {len(self.interview_questions)} questions successfully")
                return True
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            # Fallback to template questions
            self.interview_questions = self._generate_fallback_questions(num_questions, categories)
            return len(self.interview_questions) > 0
    
    def evaluate_response(self, question: Dict, response: str) -> Dict:
        """
        Evaluate a candidate's response to an interview question
        
        Args:
            question: Question dictionary
            response: Candidate's response text
            
        Returns:
            Evaluation dictionary with score and feedback
        """
        if not response.strip():
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": ["No response provided"],
                "overall_assessment": "No response to evaluate",
                "follow_up_question": "Could you please provide a response to the question?"
            }
        
        # Get document analysis for context
        analysis = self.analyze_documents()
        
        evaluation_prompt = f"""
        Evaluate this interview response comprehensively:
        
        QUESTION: {question['question']}
        QUESTION CATEGORY: {question.get('category', 'general')}
        QUESTION DIFFICULTY: {question.get('difficulty', 'mid')}
        EVALUATION CRITERIA: {question.get('evaluation_criteria', [])}
        IDEAL RESPONSE ELEMENTS: {question.get('ideal_response_elements', [])}
        
        CANDIDATE RESPONSE: {response}
        
        CONTEXT:
        Job Requirements: {analysis.get('job_analysis', {}).get('key_skills', [])}
        Company Values: {analysis.get('company_analysis', {}).get('values', [])}
        Candidate Background: {analysis.get('candidate_analysis', {}).get('strengths', [])}
        
        Provide detailed evaluation as JSON:
        {{
            "score": 85,
            "category_scores": {{
                "relevance": 90,
                "depth": 80,
                "clarity": 85,
                "examples": 75,
                "technical_accuracy": 88
            }},
            "strengths": ["specific strength 1", "specific strength 2"],
            "weaknesses": ["specific area for improvement 1", "specific area 2"],
            "missing_elements": ["what was missing from ideal response"],
            "standout_points": ["what made this response notable"],
            "overall_assessment": "detailed paragraph assessment",
            "follow_up_question": "relevant follow-up based on response",
            "improvement_suggestions": ["specific suggestion 1", "suggestion 2"],
            "red_flags": ["any concerning elements"],
            "confidence_level": "high/medium/low assessment confidence"
        }}
        
        Scoring Guide:
        90-100: Exceptional response, exceeds expectations
        80-89: Strong response, meets all expectations  
        70-79: Good response, meets most expectations
        60-69: Adequate response, meets basic expectations
        50-59: Below expectations, significant gaps
        0-49: Poor response, major concerns
        """
        
        system_prompt = """You are an expert interviewer with deep experience in technical and behavioral assessment. 
        Evaluate responses objectively and constructively. Provide specific, actionable feedback that helps 
        candidates improve while accurately assessing their fit for the role."""
        
        try:
            response_text = self._call_llm(evaluation_prompt, system_prompt, temperature=0.3)
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Validate evaluation
                return self._validate_evaluation(evaluation)
            else:
                raise ValueError("No valid JSON found in evaluation response")
                
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return self._generate_fallback_evaluation(response)
    
    def generate_final_report(self) -> Dict:
        """
        Generate comprehensive final interview report
        
        Returns:
            Detailed report dictionary
        """
        if not self.responses:
            return {"error": "No interview responses to analyze"}
        
        # Get document analysis
        analysis = self.analyze_documents()
        
        # Calculate aggregate scores
        total_score = sum(r.get('evaluation', {}).get('score', 0) for r in self.responses)
        avg_score = total_score / len(self.responses)
        
        # Aggregate category scores
        category_scores = {}
        for response in self.responses:
            eval_data = response.get('evaluation', {})
            cat_scores = eval_data.get('category_scores', {})
            for category, score in cat_scores.items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
        
        # Average category scores
        avg_category_scores = {
            cat: sum(scores) / len(scores) 
            for cat, scores in category_scores.items()
        }
        
        report_prompt = f"""
        Generate a comprehensive interview analysis report:
        
        CANDIDATE BACKGROUND:
        {json.dumps(analysis.get('candidate_analysis', {}), indent=2)}
        
        JOB REQUIREMENTS:
        {json.dumps(analysis.get('job_analysis', {}), indent=2)}
        
        COMPANY PROFILE:
        {json.dumps(analysis.get('company_analysis', {}), indent=2)}
        
        INTERVIEW RESPONSES:
        {json.dumps(self.responses, indent=2)}
        
        PERFORMANCE METRICS:
        Average Score: {avg_score:.1f}/100
        Category Scores: {avg_category_scores}
        Total Questions: {len(self.interview_questions)}
        Answered: {len(self.responses)}
        
        Generate comprehensive report as JSON:
        {{
            "overall_score": 82,
            "recommendation": "Strong Hire/Hire/Maybe/No Hire",
            "confidence_level": "High/Medium/Low",
            "cultural_fit_score": 85,
            "technical_competency_score": 78,
            "communication_score": 88,
            "experience_relevance_score": 80,
            
            "executive_summary": "2-3 sentence high-level assessment",
            
            "strengths": [
                "Specific strength with evidence",
                "Another strength with examples"
            ],
            "concerns": [
                "Specific concern with rationale", 
                "Another area needing development"
            ],
            "standout_moments": [
                "Notable response or insight",
                "Impressive demonstration of skill"
            ],
            "red_flags": [
                "Any concerning responses or gaps"
            ],
            
            "detailed_feedback": {{
                "technical_skills": {{
                    "score": 78,
                    "notes": "Detailed assessment of technical capabilities",
                    "evidence": ["specific examples from responses"]
                }},
                "problem_solving": {{
                    "score": 85,
                    "notes": "Assessment of analytical and problem-solving skills",
                    "evidence": ["examples from responses"]
                }},
                "communication": {{
                    "score": 88,
                    "notes": "Assessment of clarity, articulation, and presentation",
                    "evidence": ["examples from responses"]
                }},
                "cultural_alignment": {{
                    "score": 82,
                    "notes": "Fit with company values and culture",
                    "evidence": ["examples from responses"]
                }},
                "leadership_potential": {{
                    "score": 75,
                    "notes": "Assessment of leadership qualities and potential",
                    "evidence": ["examples from responses"]
                }}
            }},
            
            "next_steps": "Specific recommendation for next interview round or hiring decision",
            "focus_areas": [
                "Areas to explore in next interview",
                "Skills to validate further"
            ],
            "preparation_suggestions": [
                "What candidate should prepare for next round",
                "Areas they should strengthen"
            ],
            
            "risk_assessment": {{
                "technical_risk": "Low/Medium/High - with explanation",
                "cultural_risk": "Low/Medium/High - with explanation", 
                "performance_risk": "Low/Medium/High - with explanation"
            }},
            
            "comparative_analysis": "How this candidate compares to typical candidates for this role",
            "growth_potential": "Assessment of candidate's potential for growth and development",
            "team_fit": "How well candidate would fit with existing team dynamics"
        }}
        """
        
        system_prompt = """You are a senior hiring manager and technical leader with extensive experience 
        in candidate assessment. Generate thorough, balanced, and actionable reports that help make 
        informed hiring decisions. Be specific with evidence and recommendations."""
        
        try:
            response = self._call_llm(report_prompt, system_prompt, temperature=0.4)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                report = json.loads(json_str)
                
                # Add metadata
                report['generated_at'] = datetime.now().isoformat()
                report['interview_duration'] = len(self.responses)
                report['questions_answered'] = len(self.responses)
                report['total_questions'] = len(self.interview_questions)
                
                self.final_report = report
                return report
            else:
                raise ValueError("No valid JSON found in report response")
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._generate_fallback_report()
    
    # Helper methods
    def _validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Validate and enhance generated questions"""
        validated = []
        
        for i, q in enumerate(questions):
            # Ensure required fields
            validated_q = {
                'id': q.get('id', i + 1),
                'question': q.get('question', 'Tell me about your experience.'),
                'category': q.get('category', 'general'),
                'difficulty': q.get('difficulty', 'mid'),
                'reasoning': q.get('reasoning', 'General assessment question'),
                'follow_up_hints': q.get('follow_up_hints', ['Can you provide more details?']),
                'evaluation_criteria': q.get('evaluation_criteria', ['relevance', 'clarity', 'depth']),
                'ideal_response_elements': q.get('ideal_response_elements', ['specific examples', 'clear explanation'])
            }
            validated.append(validated_q)
        
        return validated
    
    def _validate_evaluation(self, evaluation: Dict) -> Dict:
        """Validate and clean evaluation response"""
        return {
            'score': max(0, min(100, evaluation.get('score', 70))),
            'category_scores': evaluation.get('category_scores', {}),
            'strengths': evaluation.get('strengths', ['Clear communication']),
            'weaknesses': evaluation.get('weaknesses', ['Could provide more specific examples']),
            'missing_elements': evaluation.get('missing_elements', []),
            'standout_points': evaluation.get('standout_points', []),
            'overall_assessment': evaluation.get('overall_assessment', 'Solid response with room for improvement'),
            'follow_up_question': evaluation.get('follow_up_question', 'Can you elaborate on that?'),
            'improvement_suggestions': evaluation.get('improvement_suggestions', []),
            'red_flags': evaluation.get('red_flags', []),
            'confidence_level': evaluation.get('confidence_level', 'medium')
        }
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses for demo purposes"""
        if "interview questions" in prompt.lower():
            return '''[
                {
                    "id": 1,
                    "question": "Tell me about yourself and what interests you about this position.",
                    "category": "general",
                    "difficulty": "entry",
                    "reasoning": "Opening question to understand candidate motivation and communication style",
                    "follow_up_hints": ["What specific aspects of the role appeal to you?", "How does this align with your career goals?"],
                    "evaluation_criteria": ["clarity", "relevance", "enthusiasm"],
                    "ideal_response_elements": ["professional background summary", "connection to role", "career motivation"]
                },
                {
                    "id": 2,
                    "question": "Describe your experience with the main technologies mentioned in this job posting.",
                    "category": "technical",
                    "difficulty": "mid",
                    "reasoning": "Assess technical fit for the role requirements",
                    "follow_up_hints": ["Can you walk me through a specific project?", "What challenges did you encounter?"],
                    "evaluation_criteria": ["technical_accuracy", "depth", "examples"],
                    "ideal_response_elements": ["specific technologies", "project examples", "problem-solving approach"]
                },
                {
                    "id": 3,
                    "question": "How do you align with our company's core values and mission?",
                    "category": "cultural",
                    "difficulty": "mid",
                    "reasoning": "Evaluate cultural fit and company research",
                    "follow_up_hints": ["Can you give me a specific example?", "How would you contribute to our culture?"],
                    "evaluation_criteria": ["cultural_awareness", "alignment", "examples"],
                    "ideal_response_elements": ["company research", "value alignment", "contribution examples"]
                },
                {
                    "id": 4,
                    "question": "Tell me about a challenging project you worked on and how you overcame obstacles.",
                    "category": "behavioral",
                    "difficulty": "mid",
                    "reasoning": "Assess problem-solving skills and resilience",
                    "follow_up_hints": ["What was your specific role?", "What would you do differently?"],
                    "evaluation_criteria": ["problem_solving", "ownership", "learning"],
                    "ideal_response_elements": ["specific situation", "actions taken", "results achieved", "lessons learned"]
                },
                {
                    "id": 5,
                    "question": "Where do you see yourself growing in this role over the next 2-3 years?",
                    "category": "experience",
                    "difficulty": "mid",
                    "reasoning": "Understand long-term fit and career ambition",
                    "follow_up_hints": ["What skills would you like to develop?", "How does this fit your career plan?"],
                    "evaluation_criteria": ["career_planning", "growth_mindset", "role_understanding"],
                    "ideal_response_elements": ["specific goals", "skill development plans", "role progression understanding"]
                }
            ]'''
        
        elif "evaluate" in prompt.lower() or "assessment" in prompt.lower():
            return '''{
                "score": 78,
                "category_scores": {
                    "relevance": 80,
                    "depth": 75,
                    "clarity": 82,
                    "examples": 70,
                    "technical_accuracy": 78
                },
                "strengths": ["Clear communication", "Relevant experience mentioned", "Good structure"],
                "weaknesses": ["Could provide more specific examples", "Missing some technical details"],
                "missing_elements": ["Quantified results", "Specific technologies used"],
                "standout_points": ["Good problem-solving approach", "Shows learning mindset"],
                "overall_assessment": "Solid response that demonstrates relevant experience and good communication skills. Could be strengthened with more specific examples and measurable outcomes.",
                "follow_up_question": "Can you provide more specific details about the technologies you used in that project?",
                "improvement_suggestions": ["Include specific metrics or outcomes", "Mention particular tools or frameworks"],
                "red_flags": [],
                "confidence_level": "medium"
            }'''
        
        elif "report" in prompt.lower() or "comprehensive" in prompt.lower():
            return self._get_mock_final_report()
        
        elif "analyze" in prompt.lower():
            return self._get_mock_analysis_json()
        
        return "Mock response generated for demo purposes."
    
    def _get_mock_analysis(self) -> Dict:
        """Mock document analysis for demo"""
        return {
            "job_analysis": {
                "title": "Software Engineer",
                "key_skills": ["Python", "JavaScript", "React", "SQL", "Git"],
                "experience_level": "mid",
                "technical_requirements": ["Web development", "Database design", "API integration"],
                "soft_skills": ["Communication", "Teamwork", "Problem-solving"],
                "responsibilities": ["Develop web applications", "Collaborate with team", "Code review"]
            },
            "company_analysis": {
                "industry": "Technology",
                "company_size": "medium",
                "values": ["Innovation", "Collaboration", "Quality", "Growth"],
                "culture_keywords": ["agile", "remote-friendly", "learning"],
                "work_environment": "hybrid"
            },
            "candidate_analysis": {
                "experience_years": "3-5",
                "technical_skills": ["Python", "JavaScript", "React", "SQL"],
                "education": "Computer Science degree",
                "previous_roles": ["Junior Developer", "Software Engineer"],
                "achievements": ["Led project delivery", "Improved system performance"],
                "career_progression": "Steady growth with increasing responsibilities",
                "skill_gaps": ["Advanced system design", "Leadership experience"],
                "strengths": ["Strong technical foundation", "Good communication", "Fast learner"]
            },
            "fit_assessment": {
                "technical_match": "85",
                "experience_match": "80",
                "cultural_match": "90",
                "overall_match": "85",
                "red_flags": ["Limited leadership experience"],
                "green_flags": ["Strong cultural alignment", "Relevant technical skills", "Growth mindset"]
            }
        }
    
    def _get_mock_analysis_json(self) -> str:
        """Mock analysis as JSON string"""
        return json.dumps(self._get_mock_analysis(), indent=2)
    
    def _generate_fallback_questions(self, num_questions: int, categories: List[str]) -> List[Dict]:
        """Generate fallback questions when LLM fails"""
        fallback_questions = [
            {
                "id": 1,
                "question": "Tell me about yourself and your background.",
                "category": "general",
                "difficulty": "entry",
                "reasoning": "General opening question to assess communication and background",
                "follow_up_hints": ["What interests you about this role?"],
                "evaluation_criteria": ["clarity", "relevance"],
                "ideal_response_elements": ["background summary", "career motivation"]
            },
            {
                "id": 2,
                "question": "What relevant experience do you have for this position?",
                "category": "experience",
                "difficulty": "mid",
                "reasoning": "Assess relevant background and experience",
                "follow_up_hints": ["Can you provide specific examples?"],
                "evaluation_criteria": ["relevance", "examples"],
                "ideal_response_elements": ["specific experience", "project examples"]
            },
            {
                "id": 3,
                "question": "How do you approach problem-solving in your work?",
                "category": "behavioral",
                "difficulty": "mid",
                "reasoning": "Evaluate problem-solving methodology",
                "follow_up_hints": ["Can you walk me through a specific example?"],
                "evaluation_criteria": ["methodology", "examples"],
                "ideal_response_elements": ["systematic approach", "specific example"]
            },
            {
                "id": 4,
                "question": "What interests you about our company and this role?",
                "category": "cultural",
                "difficulty": "entry",
                "reasoning": "Assess company research and motivation",
                "follow_up_hints": ["What specific aspects appeal to you?"],
                "evaluation_criteria": ["research", "alignment"],
                "ideal_response_elements": ["company knowledge", "role understanding"]
            },
            {
                "id": 5,
                "question": "Describe a challenging situation you faced and how you handled it.",
                "category": "behavioral",
                "difficulty": "mid",
                "reasoning": "Assess resilience and problem-solving under pressure",
                "follow_up_hints": ["What did you learn from that experience?"],
                "evaluation_criteria": ["problem_solving", "learning"],
                "ideal_response_elements": ["specific situation", "actions taken", "outcome"]
            }
        ]
        
        # Filter by requested categories and limit to num_questions
        filtered = [q for q in fallback_questions if q['category'] in categories]
        return filtered[:num_questions] if filtered else fallback_questions[:num_questions]
    
    def _generate_fallback_evaluation(self, response: str) -> Dict:
        """Generate fallback evaluation when LLM fails"""
        # Simple heuristic evaluation
        word_count = len(response.split())
        
        if word_count < 10:
            score = 40
            assessment = "Response is too brief and lacks detail."
        elif word_count < 30:
            score = 60
            assessment = "Response addresses the question but lacks sufficient detail."
        elif word_count < 100:
            score = 75
            assessment = "Good response with adequate detail and examples."
        else:
            score = 85
            assessment = "Comprehensive response with good detail and examples."
        
        return {
            "score": score,
            "category_scores": {
                "relevance": score,
                "depth": max(40, score - 10),
                "clarity": min(90, score + 5)
            },
            "strengths": ["Addresses the question", "Clear communication"],
            "weaknesses": ["Could provide more specific examples"] if word_count < 50 else [],
            "missing_elements": ["Specific examples", "Quantified results"] if word_count < 30 else [],
            "standout_points": [],
            "overall_assessment": assessment,
            "follow_up_question": "Can you provide more specific details or examples?",
            "improvement_suggestions": ["Include specific examples", "Provide more context"],
            "red_flags": [],
            "confidence_level": "medium"
        }
    
    def _generate_fallback_report(self) -> Dict:
        """Generate fallback report when LLM fails"""
        if not self.responses:
            return {"error": "No responses to analyze"}
        
        # Calculate basic metrics
        scores = [r.get('evaluation', {}).get('score', 70) for r in self.responses]
        avg_score = sum(scores) / len(scores)
        
        return {
            "overall_score": int(avg_score),
            "recommendation": "Hire" if avg_score >= 75 else "Maybe" if avg_score >= 60 else "No Hire",
            "confidence_level": "Medium",
            "cultural_fit_score": int(avg_score + 5),
            "technical_competency_score": int(avg_score - 5),
            "communication_score": int(avg_score + 10),
            "experience_relevance_score": int(avg_score),
            
            "executive_summary": f"Candidate demonstrated solid performance with an average score of {avg_score:.0f}/100 across {len(self.responses)} questions.",
            
            "strengths": [
                "Clear communication throughout the interview",
                "Demonstrates relevant experience",
                "Shows enthusiasm for the role"
            ],
            "concerns": [
                "Some responses could be more detailed",
                "Limited specific examples provided"
            ],
            "standout_moments": [
                "Articulated responses clearly",
                "Showed good understanding of role requirements"
            ],
            "red_flags": [],
            
            "detailed_feedback": {
                "technical_skills": {
                    "score": int(avg_score - 5),
                    "notes": "Demonstrates foundational technical knowledge with room for growth",
                    "evidence": ["Responses showed basic understanding", "Could benefit from more specific examples"]
                },
                "problem_solving": {
                    "score": int(avg_score),
                    "notes": "Shows logical thinking and structured approach to problems",
                    "evidence": ["Systematic approach to questions", "Clear thought process"]
                },
                "communication": {
                    "score": int(avg_score + 10),
                    "notes": "Strong communication skills demonstrated throughout",
                    "evidence": ["Clear and articulate responses", "Good listening skills"]
                },
                "cultural_alignment": {
                    "score": int(avg_score + 5),
                    "notes": "Good alignment with company values and culture",
                    "evidence": ["Positive attitude", "Team-oriented responses"]
                }
            },
            
            "next_steps": "Recommend proceeding to next interview round" if avg_score >= 70 else "Consider additional screening",
            "focus_areas": [
                "Technical deep-dive interview",
                "Team collaboration assessment"
            ],
            "preparation_suggestions": [
                "Prepare specific project examples",
                "Review technical concepts",
                "Research team structure"
            ],
            
            "risk_assessment": {
                "technical_risk": "Low" if avg_score >= 75 else "Medium",
                "cultural_risk": "Low",
                "performance_risk": "Low" if avg_score >= 75 else "Medium"
            },
            
            "comparative_analysis": "Candidate performs at expected level for the role requirements",
            "growth_potential": "Shows good potential for learning and development",
            "team_fit": "Likely to integrate well with existing team dynamics",
            
            "generated_at": datetime.now().isoformat(),
            "interview_duration": len(self.responses),
            "questions_answered": len(self.responses),
            "total_questions": len(self.interview_questions)
        }
    
    def _get_mock_final_report(self) -> str:
        """Mock final report as JSON string"""
        return '''{
            "overall_score": 82,
            "recommendation": "Strong Hire",
            "confidence_level": "High",
            "cultural_fit_score": 85,
            "technical_competency_score": 78,
            "communication_score": 88,
            "experience_relevance_score": 80,
            
            "executive_summary": "Strong candidate who demonstrates excellent communication skills and solid technical foundation. Shows great cultural alignment and growth potential.",
            
            "strengths": [
                "Excellent communication and articulation skills",
                "Strong problem-solving approach with systematic thinking",
                "Demonstrates relevant technical experience with concrete examples",
                "Shows high emotional intelligence and cultural awareness",
                "Exhibits growth mindset and eagerness to learn"
            ],
            "concerns": [
                "Limited experience with advanced system design patterns",
                "Could benefit from more leadership experience",
                "Some gaps in emerging technology knowledge"
            ],
            "standout_moments": [
                "Provided detailed walkthrough of complex project with clear problem-solving methodology",
                "Demonstrated strong cultural research and genuine enthusiasm for company mission",
                "Showed excellent self-awareness and learning from past mistakes"
            ],
            "red_flags": [],
            
            "detailed_feedback": {
                "technical_skills": {
                    "score": 78,
                    "notes": "Solid technical foundation with room for growth in advanced concepts",
                    "evidence": ["Clearly explained technical projects", "Demonstrated understanding of core technologies", "Some gaps in advanced system design"]
                },
                "problem_solving": {
                    "score": 85,
                    "notes": "Excellent analytical skills with systematic approach to complex problems",
                    "evidence": ["Broke down complex problems methodically", "Considered multiple solutions", "Showed iterative improvement mindset"]
                },
                "communication": {
                    "score": 88,
                    "notes": "Outstanding communication skills with clear, structured responses",
                    "evidence": ["Articulated complex concepts clearly", "Active listening demonstrated", "Appropriate professional tone"]
                },
                "cultural_alignment": {
                    "score": 85,
                    "notes": "Strong alignment with company values and demonstrated cultural research",
                    "evidence": ["Researched company values thoroughly", "Provided examples of value alignment", "Shows team-first mentality"]
                },
                "leadership_potential": {
                    "score": 75,
                    "notes": "Shows emerging leadership qualities with potential for development",
                    "evidence": ["Demonstrated mentoring experience", "Shows initiative in projects", "Could benefit from formal leadership experience"]
                }
            },
            
            "next_steps": "Recommend proceeding to technical deep-dive interview with team members",
            "focus_areas": [
                "Advanced system design and architecture discussion",
                "Team collaboration and leadership scenarios",
                "Specific technology stack deep-dive"
            ],
            "preparation_suggestions": [
                "Review system design principles and patterns",
                "Prepare detailed technical project walkthroughs",
                "Think through team leadership examples and scenarios"
            ],
            
            "risk_assessment": {
                "technical_risk": "Low - Strong foundation with clear growth trajectory",
                "cultural_risk": "Very Low - Excellent cultural alignment and team mindset",
                "performance_risk": "Low - Demonstrates strong work ethic and problem-solving skills"
            },
            
            "comparative_analysis": "Candidate performs above average compared to typical candidates for this role, particularly in communication and cultural fit",
            "growth_potential": "High potential for rapid growth given strong foundational skills and demonstrated learning agility",
            "team_fit": "Excellent fit for collaborative team environment with strong communication and cultural alignment"
        }'''
    
    def get_status(self) -> Dict:
        """Get current interview status"""
        return {
            "documents_loaded": all([self.job_post, self.company_profile, self.candidate_resume]),
            "questions_generated": len(self.interview_questions) > 0,
            "interview_started": self.interview_started,
            "current_question": self.current_question_idx,
            "total_questions": len(self.interview_questions),
            "responses_collected": len(self.responses),
            "interview_completed": self.interview_completed,
            "report_generated": self.final_report is not None
        }
    
    def reset_interview(self):
        """Reset interview state for new candidate"""
        self.current_question_idx = 0
        self.responses = []
        self.interview_started = False
        self.interview_completed = False
        self.final_report = None
        logger.info("Interview state reset")
    
    def reset_all(self):
        """Reset everything including documents"""
        self.job_post = ""
        self.company_profile = ""
        self.candidate_resume = ""
        self.interview_questions = []
        self._document_analysis = {}
        self.reset_interview()
        logger.info("All agent state reset")