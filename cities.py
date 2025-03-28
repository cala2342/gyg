import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import json
import re

# Initialize the Ollama LLM
llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    temperature=0,  # deterministic output
    json_mode=True,  # for structured output
    num_gpu=999,
    context_window=2048
)

# Set the LLM in LlamaIndex settings
Settings.llm = llm

city_prompt = '''You are an expert at identifying cities from activity titles for tourism experiences. Your task is to predict the most likely city based on the given activity title. Base your prediction strictly on the explicit text of the title—do not infer beyond what’s provided. If the title does not contain enough information to confidently identify a specific city (e.g., no clear city name, region, or landmark), return an empty string.

Rules:
- Return only the city name (e.g., "Madrid", "Paris") if confidently identified.
- If multiple cities could apply or if the title is ambiguous (e.g., "Grand Canyon Tour" without origin), return an empty string "".
- Do not include country names, explanations, or additional text beyond the city name or empty string.
- Return the result as a JSON object with a single key "city" and the value being the city name or "".

Example Outputs:
- For "Madrid: Royal Palace Guided Tour": {"city": "Madrid"}
- For "Sunset Cruise with Dinner": {"city": ""}
- For "Stonehenge Tour from London": {"city": "London"}

Now, process this input title: '''

# Input and output files
input_file = "activities.csv"
output_file = "activities_with_cities.csv"

# Load the CSV file
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit()

# Ensure 'title' column exists
if 'title' not in df.columns:
    print("Error: 'title' column not found in the CSV.")
    exit()

# Add 'city' column if it doesn't exist, initialize with empty strings
if 'city' not in df.columns:
    df['city'] = ""

# Process each row
for index, row in df.iterrows():
    # Skip if city is already filled (non-empty)
    if pd.notna(row['city']) and row['city'].strip() != "":
        continue

    # Input validation: Check if title exists and is a non-empty string
    title = row.get("title")
    if pd.isna(title) or not isinstance(title, str) or title.strip() == "":
        print(f"Skipped row {index + 1}: Title is missing, not a string, or empty")
        df.at[index, 'city'] = ""
        continue

    print(f"Processing row {index + 1}: {title[:80]}...")

    # LLM call: Predict the city
    eval_prompt = city_prompt + title
    response_text = ""
    try:
        response = llm.complete(eval_prompt)
        response_text = response.text

        # Extract JSON object using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Clean potential markdown syntax
            json_str = re.sub(r'^```json\s*', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'\s*```$', '', json_str)
            result_dict = json.loads(json_str)

            # Ensure it's a dictionary with 'city' key
            if not isinstance(result_dict, dict) or 'city' not in result_dict:
                raise json.JSONDecodeError(f"Expected {'city': value}, got {result_dict}", json_str, 0)

            city = result_dict['city']
            # Validate city is a string
            if not isinstance(city, str):
                print(f"Warning in row {index + 1}: City value '{city}' is not a string, setting to empty")
                city = ""

            df.at[index, 'city'] = city

        else:
            print(f"Error in row {index + 1}: No JSON object found. Response was:\n{response_text}")
            df.at[index, 'city'] = ""
            continue

        # Save after each successful row
        df.to_csv(output_file, index=False)

    except json.JSONDecodeError as e:
        print(f"Error in row {index + 1}: JSON decode failed. Error: {e}. Response was:\n{response_text}")
        df.at[index, 'city'] = ""
        continue
    except Exception as e:
        print(f"Error in row {index + 1}: {str(e)}. Response was:\n{response_text}")
        df.at[index, 'city'] = ""
        continue

# Final save
df.to_csv(output_file, index=False)
print("Processing complete! Output saved to", output_file)