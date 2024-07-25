import json
from groq import Groq
from api import api_key
from test_cases import test_cases

# Initialize the Groq client
client = Groq(api_key=api_key)


# Construct prompt
users_prompt = test_cases[3]


# Function to get chat completion
def get_chat_completion(model):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": """Please extract and structure all relevant information from the email signature into a JSON format. Ensure to include the following fields:

- Name
- Title
- Company
- Email
- Phone
- Address
- Social Media Links

Example Email Signature:
---
John Doe
Senior Software Engineer
Tech Company
john.doe@example.com
123-456-7890
123 Tech Street, Silicon Valley, CA
LinkedIn: linkedin.com/in/johndoe
Twitter: @johndoe
---

Expected JSON Format:
{
    "name": "John Doe",
    "title": "Senior Software Engineer",
    "company": "Tech Company",
    "email": "john.doe@example.com",
    "phone": "123-456-7890",
    "address": "123 Tech Street, Silicon Valley, CA",
    "social": {
        "LinkedIn": "linkedin.com/in/johndoe",
        "Twitter": "@johndoe"
    }
}

The test cases will capture different levels of complexity and context, including signatures with full information, partial information, and no information (no signature). 

- If a signature is complete, extract all provided details.
- If a signature is partial, extract the available details and leave the missing fields as empty strings in the JSON output.
- If there is no signature, return a JSON object with all fields as empty strings.
- Even if the name is in the greeting or closing and not clearly separated, it should still be extracted if present. Ensure to extract and structure the information accurately according to this format.
Do not comment on the answer and follow the structure clearly.
Extract and structure the information accurately according to this format and nothing else."""
                },
                {
                    "role": "system",
                    "content": """{
                      "first-name": "John",
                      "last-name": "Doe",
                      "position": "Senior Software Engineer",
                      "company": "Tech Company",
                      "email": "john.doe@example.com",
                      "phone": "123-456-7890",
                      "address": "123 Tech Street, Silicon Valley, CA",
                      "social": {
                        "LinkedIn": "linkedin.com/in/johndoe",
                        "Twitter": "@johndoe"
                      }
                    }"""
                },
                {
                    "role": "user",
                    "content": """Hi All,
                    
Just a quick note to remind you about the upcoming deadline.

Cheers,
Tom (name after "Cheers" is "name" field)
Customer Support""",
                },
                {
                    "role": "system",
                    'content': """
                    {
                        "name": "Tom",
                        "position": "Customer Support",
                        "company": "",
                        "email": "",
                        "phone": "",
                        "address": "",
                        "social": {
                            "LinkedIn": "",
                            "Twitter": ""
                        }     
                    }
                    """
                },
                {
                    "role": "user",
                    "content": str(users_prompt),
                },
            ],
            model=model,
            max_tokens=200,  # Set the maximum number of tokens

        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API request: {e}")
        return None


# Function to extract and save JSON data
def save_json(response, output_file):
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]
        extracted_data = json.loads(json_str)

        with open(output_file, 'w') as json_file:
            json.dump(extracted_data, json_file, indent=2)
        print(f"Output written to {output_file}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error saving JSON: {e}")


# Process for the first model
model1 = "llama3-8b-8192"
response_content1 = get_chat_completion(model1)
print(response_content1)
if response_content1:
    save_json(response_content1, 'output_llama3.json')
else:
    print("No valid JSON content found in the generated text for llama3")

# Process for the second model
model2 = "mixtral-8x7b-32768"
response_content2 = get_chat_completion(model2)
print(response_content2)
if response_content2:
    save_json(response_content2, 'output_mixtral.json')
else:
    print("No valid JSON content found in the generated text for mixtral")
