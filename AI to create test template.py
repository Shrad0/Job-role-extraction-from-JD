#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install openai


# In[2]:


import openai


# In[3]:


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


# In[8]:


pip install openai==0.28


# In[19]:


import openai
openai.api_key = '...'
#Replace with your API key

# In[ ]:


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


# # Step 1: Extracting job role, qualification and skills from the JD

# In[20]:


def extract_jd_info(jd):
    # Define the prompt for extracting job role, qualifications, and skills
    prompt = f"""
    Analyze the following job description and extract the following details:
    1. Job Role
    2. Qualifications (Education, Experience, etc.)
    3. Skills required

    Job Description: {jd}
    """

    # Use the OpenAI API to generate a completion for this prompt with gpt-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use gpt-3.5-turbo or any version available to you
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    
    # Return the extracted information
    return response.choices[0].message['content'].strip()


# In[21]:


# Sample Job Description
jd_text = """
We are seeking a Data Scientist with expertise in Python, Machine Learning, and Statistical Analysis. 
The ideal candidate should hold a Master’s degree in Computer Science or related fields and have at least 3 years of experience. 
Familiarity with data visualization tools and cloud platforms is a plus.
"""


# In[22]:


# Extract the job role, qualifications, and skills
jd_info = extract_jd_info(jd_text)
print(jd_info)


# # Step 2: Generate questions for each questions

# In[29]:


def generate_questions(skill):
    prompt = f"Create 3 MCQs for {skill}. Each question should have 4 options and mark the correct one."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

print(generate_questions("Machine Learning"))


# # Step 3: Guide users to marketplace

# In[31]:


def recommend_templates(job_role):
    prompt = f"Suggest test templates for the role of {job_role}. Include coding, MCQ, and behavioral sections."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

print(recommend_templates("Data Scientist"))

