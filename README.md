# AI Interview Agent

A Streamlit-based application that leverages LLMs to conduct interactive interview Q&A sessions based on resumes and job descriptions.

## Features

- Upload and parse resumes and job postings
- AI-powered interview agent for Q&A
- Modular codebase for easy extension

## Project Structure

```
ai_interview_agent/
├── app.py                   # Main Streamlit app
├── agents/
│   └── interview_agent.py   # LLM logic for Q&A
├── utils/
│   └── parser.py            # Resume, job post, and profile parsers
├── data/
│   └── sample_resume.pdf    # Sample files (for testing)
├── requirements.txt         # Python dependencies
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ai_interview_agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   uv venv .venv
   .venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

