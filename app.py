import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Import custom modules
from agents.interview_agent import InterviewAgent
from utils.parser import DocumentParser, get_document_input

def setup_streamlit_config():
    """Configure Streamlit app"""
    st.set_page_config(
        page_title="AI Interview Agent POC",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    .status-pending {
        background-color: #fff3cd;
        color: #856404;
    }
    .question-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
    .response-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system capabilities in sidebar"""
    st.sidebar.write("### 🔧 System Status")
    
    # Check voice capabilities
    try:
        from audio_recorder_streamlit import audio_recorder
        import speech_recognition as sr
        voice_status = "✅ Available"
    except ImportError:
        voice_status = "❌ Not Available"
    
    # Check document parsing capabilities
    try:
        import PyPDF2
        from docx import Document
        doc_status = "✅ Available"
    except ImportError:
        doc_status = "❌ Not Available"
    
    st.sidebar.write(f"🎤 Voice Input: {voice_status}")
    st.sidebar.write(f"📄 Document Parsing: {doc_status}")
    
    # if voice_status == "❌ Not Available":
        # st.sidebar.warning("Install for voice features:")
        # st.sidebar.code("pip install streamlit-audio-recorder SpeechRecognition")
    
    if doc_status == "❌ Not Available":
        st.sidebar.warning("Install for document parsing:")
        st.sidebar.code("pip install PyPDF2 python-docx")

def setup_page():
    """Setup phase - Document collection"""
    st.title("📋 Document Setup")
    st.write("Upload or input the required documents to start the interview process.")
    
    # Get session state agent
    agent = st.session_state.agent
    
    # Document inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 📢 Job Post")
        job_post = get_document_input("Job Posting", "job_posting")
        if job_post:
            agent.job_post = job_post
            st.success("✅ Job post loaded")
    
    with col2:
        st.write("### 🏢 Company Profile")
        company_profile = get_document_input("Company Profile", "company_profile")
        if company_profile:
            agent.company_profile = company_profile
            st.success("✅ Company profile loaded")
    
    with col3:
        st.write("### 👤 Candidate Resume")
        candidate_resume = get_document_input("Resume", "resume")
        if candidate_resume:
            agent.candidate_resume = candidate_resume
            st.success("✅ Resume loaded")
    
    # Show document previews
    if any([agent.job_post, agent.company_profile, agent.candidate_resume]):
        st.write("---")
        st.write("### 📋 Document Previews")
        
        if agent.job_post:
            with st.expander("📢 Job Post Preview"):
                st.text_area("Job Post Content", agent.job_post, height=150, disabled=True)
        
        if agent.company_profile:
            with st.expander("🏢 Company Profile Preview"):
                st.text_area("Company Profile Content", agent.company_profile, height=150, disabled=True)
        
        if agent.candidate_resume:
            with st.expander("👤 Resume Preview"):
                st.text_area("Resume Content", agent.candidate_resume, height=150, disabled=True)
    
    # Document status summary
    st.write("---")
    st.write("### 📊 Setup Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ Complete" if agent.job_post else "⏳ Pending"
        st.metric("Job Post", status)
    
    with col2:
        status = "✅ Complete" if agent.company_profile else "⏳ Pending"
        st.metric("Company Profile", status)
    
    with col3:
        status = "✅ Complete" if agent.candidate_resume else "⏳ Pending"
        st.metric("Resume", status)
    
    with col4:
        all_ready = all([agent.job_post, agent.company_profile, agent.candidate_resume])
        status = "✅ Ready" if all_ready else "⏳ Incomplete"
        st.metric("Overall Status", status)
    
    return all([agent.job_post, agent.company_profile, agent.candidate_resume])

def question_generation_page():
    """Question generation phase"""
    st.title("🎯 Interview Question Generation")
    
    agent = st.session_state.agent
    
    if not all([agent.job_post, agent.company_profile, agent.candidate_resume]):
        st.warning("⚠️ Please complete document setup first!")
        return False
    
    st.write("Generate personalized interview questions based on the loaded documents.")
    
    # Generation controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_questions = st.slider("Number of questions to generate:", 5, 15, 8)
        include_categories = st.multiselect(
            "Question categories to include:",
            ["technical", "behavioral", "cultural", "experience", "situational"],
            default=["technical", "behavioral", "cultural"]
        )
    
    with col2:
        st.write("**Generation Settings:**")
        difficulty_level = st.selectbox("Difficulty Level:", ["Entry", "Mid", "Senior"])
        focus_areas = st.multiselect("Focus Areas:", ["Problem Solving", "Leadership", "Communication", "Technical Skills"])
    
    # Generate questions
    if st.button("🎯 Generate Interview Questions", type="primary"):
        with st.spinner("Analyzing documents and generating personalized questions..."):
            success = agent.generate_interview_questions(
                num_questions=num_questions,
                categories=include_categories,
                difficulty=difficulty_level,
                focus_areas=focus_areas
            )
            
            if success:
                st.success(f"✅ Generated {len(agent.interview_questions)} personalized questions!")
                st.balloons()
            else:
                st.error("❌ Failed to generate questions. Please try again.")
    
    # Display generated questions
    if agent.interview_questions:
        st.write("---")
        st.write("### 📝 Generated Interview Questions")
        
        # Question overview
        categories = {}
        for q in agent.interview_questions:
            category = q.get('category', 'other')
            categories[category] = categories.get(category, 0) + 1
        
        st.write("**Question Distribution:**")
        cols = st.columns(len(categories))
        for i, (category, count) in enumerate(categories.items()):
            with cols[i]:
                st.metric(category.title(), count)
        
        # Individual questions
        for i, question in enumerate(agent.interview_questions, 1):
            with st.expander(f"Question {i}: {question['question'][:60]}..."):
                st.write(f"**Category:** {question.get('category', 'General')}")
                st.write(f"**Question:** {question['question']}")
                st.write(f"**Reasoning:** {question.get('reasoning', 'N/A')}")
                
                if question.get('follow_up_hints'):
                    st.write("**Follow-up hints:**")
                    for hint in question['follow_up_hints']:
                        st.write(f"• {hint}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Regenerate Questions"):
                agent.interview_questions = []
                st.rerun()
        
        with col2:
            if st.button("✏️ Customize Questions"):
                st.session_state.customize_mode = True
        
        with col3:
            if st.button("▶️ Start Interview", type="primary"):
                agent.interview_started = True
                st.session_state.current_page = "interview"
                st.rerun()
    
    return len(agent.interview_questions) > 0

def interview_page():
    """Main interview interface"""
    st.markdown('<div class="main-header"><h1>🤖 AI Interview Session</h1></div>', unsafe_allow_html=True)
    
    agent = st.session_state.agent
    
    if not agent.interview_questions:
        st.warning("⚠️ Please generate questions first!")
        return
    
    # Interview progress
    total_questions = len(agent.interview_questions)
    current_idx = agent.current_question_idx
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Question", f"{current_idx + 1}/{total_questions}")
    
    with col2:
        progress = current_idx / total_questions if total_questions > 0 else 0
        st.metric("Progress", f"{progress:.0%}")
    
    with col3:
        completed = len(agent.responses)
        st.metric("Completed", f"{completed}/{total_questions}")
    
    with col4:
        if agent.responses:
            avg_score = sum(r.get('evaluation', {}).get('score', 0) for r in agent.responses) / len(agent.responses)
            st.metric("Avg Score", f"{avg_score:.0f}/100")
        else:
            st.metric("Avg Score", "N/A")
    
    # Progress bar
    st.progress(progress)
    
    # Current question display
    if current_idx < total_questions:
        current_question = agent.interview_questions[current_idx]
        
        # Question card
        st.markdown(f"""
        <div class="question-card">
            <h3>Question {current_idx + 1}</h3>
            <h4>{current_question['question']}</h4>
            <p><strong>Category:</strong> {current_question.get('category', 'General')}</p>
            <p><em>{current_question.get('reasoning', '')}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Response collection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get response via voice or text
            response_content = get_document_input("Your Response", f"response_{current_idx}")
            
        with col2:
            st.write("**💡 Hints:**")
            for hint in current_question.get('follow_up_hints', []):
                st.write(f"• {hint}")
            
            # Timer (optional)
            if st.checkbox("Enable Timer"):
                timer_minutes = st.number_input("Minutes:", min_value=1, max_value=10, value=3)
                if st.button("⏰ Start Timer"):
                    st.session_state.timer_start = time.time()
                    st.session_state.timer_duration = timer_minutes * 60
                
                # Show timer
                if hasattr(st.session_state, 'timer_start'):
                    elapsed = time.time() - st.session_state.timer_start
                    remaining = max(0, st.session_state.timer_duration - elapsed)
                    
                    if remaining > 0:
                        st.write(f"⏰ Time remaining: {remaining/60:.1f} minutes")
                    else:
                        st.error("⏰ Time's up!")
        
        # Submit response
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Submit Response", type="primary", disabled=not response_content):
                if response_content:
                    with st.spinner("🤔 Evaluating your response..."):
                        # Process response
                        evaluation = agent.evaluate_response(current_question, response_content)
                        
                        # Store response
                        agent.responses.append({
                            'question_id': current_question.get('id', current_idx),
                            'question': current_question['question'],
                            'response': response_content,
                            'evaluation': evaluation,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Show immediate feedback
                        display_response_feedback(evaluation)
                        
                        # Move to next question
                        agent.current_question_idx += 1
                        
                        # Auto-advance after showing feedback
                        time.sleep(2)
                        st.rerun()
        
        with col2:
            if st.button("⏭️ Skip Question"):
                agent.current_question_idx += 1
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Response"):
                st.rerun()
    
    else:
        # Interview completed
        st.success("🎉 Interview Completed!")
        st.balloons()
        
        # Show completion summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions", len(agent.interview_questions))
        
        with col2:
            st.metric("Answered", len(agent.responses))
        
        with col3:
            if agent.responses:
                avg_score = sum(r.get('evaluation', {}).get('score', 0) for r in agent.responses) / len(agent.responses)
                st.metric("Final Score", f"{avg_score:.0f}/100")
        
        # Generate report button
        if st.button("📊 Generate Final Report", type="primary"):
            agent.interview_completed = True
            st.session_state.current_page = "report"
            st.rerun()
    
    # Show previous responses
    if agent.responses:
        st.write("---")
        st.write("### 📝 Previous Responses")
        
        for i, response in enumerate(agent.responses):
            with st.expander(f"Response {i+1}: {response['question'][:50]}..."):
                st.write(f"**Question:** {response['question']}")
                st.write(f"**Your Response:** {response['response']}")
                
                evaluation = response.get('evaluation', {})
                score = evaluation.get('score', 0)
                score_color = 'score-high' if score >= 80 else 'score-medium' if score >= 60 else 'score-low'
                
                st.markdown(f"**Score:** <span class='{score_color}'>{score}/100</span>", unsafe_allow_html=True)
                
                if evaluation.get('strengths'):
                    st.write(f"**Strengths:** {', '.join(evaluation['strengths'])}")
                
                if evaluation.get('weaknesses'):
                    st.write(f"**Areas for improvement:** {', '.join(evaluation['weaknesses'])}")

def display_response_feedback(evaluation: Dict):
    """Display immediate feedback for a response"""
    score = evaluation.get('score', 0)
    score_class = 'score-high' if score >= 80 else 'score-medium' if score >= 60 else 'score-low'
    
    st.markdown(f"""
    <div class="response-card">
        <h4>📊 Response Evaluation</h4>
        <p><strong>Score: <span class="{score_class}">{score}/100</span></strong></p>
        <p><strong>✅ Strengths:</strong> {', '.join(evaluation.get('strengths', []))}</p>
        <p><strong>📈 Areas for improvement:</strong> {', '.join(evaluation.get('weaknesses', []))}</p>
        <p><em>💭 {evaluation.get('overall_assessment', '')}</em></p>
    </div>
    """, unsafe_allow_html=True)

def report_page():
    """Final report display"""
    st.title("📊 Interview Analysis Report")
    
    agent = st.session_state.agent
    
    if not agent.responses:
        st.warning("⚠️ No interview data available.")
        return
    
    # Generate comprehensive report
    if not hasattr(agent, 'final_report') or not agent.final_report:
        with st.spinner("🔄 Generating comprehensive analysis report..."):
            agent.final_report = agent.generate_final_report()
    
    report = agent.final_report
    
    # Overall assessment header
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = report.get('overall_score', 0)
        score_color = '#28a745' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545'
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; border-radius: 10px; background-color: {score_color}20; border: 2px solid {score_color};">
            <h2 style="color: {score_color}; margin: 0;">{score}/100</h2>
            <p style="margin: 0; font-weight: bold;">Overall Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        recommendation = report.get('recommendation', 'N/A')
        rec_color = '#28a745' if 'recommend' in recommendation.lower() else '#dc3545'
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; border-radius: 10px; background-color: {rec_color}20; border: 2px solid {rec_color};">
            <h4 style="color: {rec_color}; margin: 0;">{recommendation}</h4>
            <p style="margin: 0; font-weight: bold;">Recommendation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cultural_fit = report.get('cultural_fit_score', 0)
        fit_color = '#28a745' if cultural_fit >= 80 else '#ffc107' if cultural_fit >= 60 else '#dc3545'
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; border-radius: 10px; background-color: {fit_color}20; border: 2px solid {fit_color};">
            <h3 style="color: {fit_color}; margin: 0;">{cultural_fit}%</h3>
            <p style="margin: 0; font-weight: bold;">Cultural Fit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.write("---")
    st.write("### 📊 Detailed Assessment")
    
    # Skills breakdown
    detailed_feedback = report.get('detailed_feedback', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 💪 Skill Assessment")
        for skill, details in detailed_feedback.items():
            score = details.get('score', 0)
            progress_color = '#28a745' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545'
            
            st.write(f"**{skill.replace('_', ' ').title()}**")
            st.progress(score / 100)
            st.write(f"*{details.get('notes', '')}*")
            st.write("")
    
    with col2:
        st.write("#### ✅ Strengths")
        for strength in report.get('strengths', []):
            st.write(f"• {strength}")
        
        st.write("#### 📈 Areas for Development")
        for concern in report.get('concerns', []):
            st.write(f"• {concern}")
    
    # Question-by-question analysis
    st.write("---")
    st.write("### 📝 Question-by-Question Analysis")
    
    for i, response in enumerate(agent.responses):
        with st.expander(f"Q{i+1}: {response['question'][:60]}... (Score: {response.get('evaluation', {}).get('score', 0)}/100)"):
            evaluation = response.get('evaluation', {})
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Question:** {response['question']}")
                st.write(f"**Response:** {response['response']}")
                st.write(f"**Assessment:** {evaluation.get('overall_assessment', 'N/A')}")
            
            with col2:
                score = evaluation.get('score', 0)
                st.metric("Score", f"{score}/100")
                
                if evaluation.get('follow_up_question'):
                    st.write(f"**Follow-up:** {evaluation['follow_up_question']}")
    
    # Next steps and recommendations
    st.write("---")
    st.write("### 🚀 Next Steps & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 📋 Immediate Next Steps")
        next_steps = report.get('next_steps', 'Schedule follow-up interview')
        st.write(next_steps)
        
        st.write("#### 🎯 Interview Focus Areas")
        focus_areas = report.get('focus_areas', ['Technical deep-dive', 'Team collaboration'])
        for area in focus_areas:
            st.write(f"• {area}")
    
    with col2:
        st.write("#### 💡 Preparation Suggestions")
        suggestions = report.get('preparation_suggestions', [
            'Review technical concepts',
            'Prepare specific examples',
            'Research company culture'
        ])
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    # Export options
    st.write("---")
    st.write("### 📥 Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download PDF Report"):
            st.info("PDF export feature coming soon!")
    
    with col2:
        # JSON export
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=report_json,
            file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("📧 Email Report"):
            st.info("Email feature coming soon!")
    
    # Start new interview
    st.write("---")
    if st.button("🔄 Start New Interview", type="primary"):
        # Reset session state
        st.session_state.agent = InterviewAgent()
        st.session_state.current_page = "setup"
        st.rerun()

def main():
    """Main application entry point"""
    setup_streamlit_config()
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = InterviewAgent()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "setup"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 AI Interview Agent")
        st.write("*POC Version 1.0*")
        
        display_system_status()
        
        st.write("---")
        st.write("### 📍 Navigation")
        
        # Page navigation
        pages = {
            "📋 Setup": "setup",
            "🎯 Questions": "questions", 
            "🤖 Interview": "interview",
            "📊 Report": "report"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
                st.rerun()
        
        # Progress indicator
        st.write("---")
        st.write("### 📊 Progress")
        
        agent = st.session_state.agent
        
        # Setup progress
        setup_complete = all([agent.job_post, agent.company_profile, agent.candidate_resume])
        st.write(f"📋 Setup: {'✅' if setup_complete else '⏳'}")
        
        # Questions progress
        questions_ready = len(agent.interview_questions) > 0
        st.write(f"🎯 Questions: {'✅' if questions_ready else '⏳'}")
        
        # Interview progress
        interview_progress = len(agent.responses) / len(agent.interview_questions) if agent.interview_questions else 0
        st.write(f"🤖 Interview: {interview_progress:.0%}")
        
        # Report status
        report_ready = hasattr(agent, 'final_report') and agent.final_report
        st.write(f"📊 Report: {'✅' if report_ready else '⏳'}")
    
    # Main content area
    current_page = st.session_state.current_page
    
    if current_page == "setup":
        setup_complete = setup_page()
        
        # Auto-advance hint
        if setup_complete:
            st.info("📍 All documents loaded! Click 'Questions' in the sidebar to generate interview questions.")
    
    elif current_page == "questions":
        questions_ready = question_generation_page()
        
        # Auto-advance hint
        if questions_ready:
            st.info("📍 Questions ready! Click 'Interview' in the sidebar to start the interview session.")
    
    elif current_page == "interview":
        interview_page()
    
    elif current_page == "report":
        report_page()
    
    # Footer
    st.write("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🤖 AI Interview Agent POC | Built with Streamlit & AI | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()