from dotenv import load_dotenv
import os
from openai import OpenAI, RateLimitError


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)

class InterviewAgent:
    def __init__(self, resume_text, job_text, company_text):
        self.context = f"""
        Job Description:
        {job_text}

        Company Profile:
        {company_text}

        Candidate Resume:
        {resume_text}
        """

    def ask(self, user_input):
        messages = [
            {"role": "system", "content": "You are an AI interview agent conducting a behavioral and technical interview."},
            {"role": "system", "content": self.context},
            {"role": "user", "content": user_input},
        ]
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=100 
        )

            return response.choices[0].message.content.strip()
        except RateLimitError:
            return "Sorry, the API quota has been exceeded. Please check your OpenAI billing and usage limits."
