import streamlit as st
from agents.interview_agent import InterviewAgent
from utils.parser import parse_pdf

st.set_page_config(page_title="AI Interview Agent", layout="wide")

st.title("🎙️ AI Interview Agent")
st.markdown("Upload a **resume**, **job description**, and **company profile** to begin the interview simulation.")

# Upload Section
resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
company_file = st.file_uploader("Upload Company Profile (PDF)", type=["pdf"])

if resume_file and job_file and company_file:
    resume_text = parse_pdf(resume_file)
    job_text = parse_pdf(job_file)
    company_text = parse_pdf(company_file)

    # Initialize Agent
    agent = InterviewAgent(resume_text, job_text, company_text)

    st.success("Files processed. You can now start the interview.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Your answer here...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = agent.ask(prompt)
        st.session_state.messages.append({"role": "ai", "content": response})

        with st.chat_message("ai"):
            st.markdown(response)
