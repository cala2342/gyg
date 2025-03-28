import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import json
import re

# Initialize the Ollama LLM
llm = Ollama(
    model="gemma3:27b",  # Or your preferred model capable of following instructions
    request_timeout=120.0, # Increased timeout for potentially complex tasks
    temperature=0,       # deterministic output
    json_mode=True,      # for structured output
    num_gpu=999,         # full offload to GPU VRAM if available
    context_window=2048  # Ensure context window is sufficient
)

# Set the LLM in LlamaIndex settings
Settings.llm = llm

# --- MODIFIED PROMPT START ---
# Define the system prompt with the V4 feature list (positively framed)
feature_prompt = '''You are an expert at analyzing activity titles for tourism experiences. Your task is to evaluate a given activity title against 20 specific features, determining which features are TRUE (the title possesses this positive quality). Base your evaluation strictly on the explicit text of the title—do not infer beyond what’s provided. If the title lacks detail to confirm a feature, it is considered FALSE.

Feature Definitions (TRUE = Positive Quality):

Category: Clarity & Core Information
clarity_specifies_location: TRUE if the title clearly states the main city, region, or landmark (e.g., "Madrid:", "Stonehenge").
clarity_specifies_core_activity: TRUE if the title clearly names the main activity or attraction (e.g., "Flamenco Performance", "Salt Mine Tour").
clarity_indicates_activity_format: TRUE if the title uses a word like "Tour", "Trip", "Cruise", "Walk", "Class", "Workshop", "Lesson" etc., to define the format.
clarity_specifies_key_transport: TRUE if transport is key to the experience (Gondola, Segway, Bike, Boat, Jeep) and the title mentions it.
clarity_indicates_day_trip_origin: TRUE if it's clearly a day trip and the title uses a "From [Location]:" structure.

Category: Value Proposition & Key Inclusions
value_highlights_access_benefit: TRUE if the title includes "Skip-the-Line", "Fast Track", "Reserved Access", "Priority Access", "Express Entry" or similar.
value_specifies_guidance: TRUE if the title explicitly mentions "Guided", "with Guide", "with an Archaeologist/Expert", etc.
value_specifies_ticket_inclusion: TRUE if the title mentions "Ticket(s)", "Admission", "Entry", or specific access levels (e.g., "Summit Access", "Arena Floor").
value_mentions_food_drink: TRUE if the title states inclusion of "Lunch", "Dinner", "Drinks", "Wine", "Tasting", "Snacks", etc.
value_indicates_target_audience: TRUE if the title includes terms suggesting suitability like "Family", "Kids", "Beginners", "Adults", etc.

Category: Experience & Exclusivity
experience_specifies_group_size: TRUE if the title includes "Small Group", "Private", or specific participant limits (e.g., "Max 10ppl").
experience_highlights_time_benefit: TRUE if the title specifies a potentially desirable time like "Sunset", "Night", "Evening", "Early-Morning", "After-Hours".
experience_specifies_duration: TRUE if the title mentions a duration (e.g., "X-Hour", "Half-Day", "Full-Day").
experience_uses_hook_keywords: TRUE if the title includes engaging words like "Best", "Highlights", "Secret", "Magic", "Ultimate", "Delicious", "Romantic", "Adventure", etc. (beyond just factual description).
experience_highlights_unique_access: TRUE if the title mentions non-standard access (e.g., "Underground", "Terraces", "Arena Floor", "Behind-the-Scenes").

Category: Quality & Professionalism
quality_free_of_errors: TRUE if the title appears grammatically correct and free from noticeable typos or strange characters (like 'Ã³').
quality_uses_appropriate_caps: TRUE if the title uses standard title case or sentence case, avoiding excessive use of ALL CAPS.
quality_free_of_contact_info: TRUE if the title does NOT contain phone numbers, email addresses, or website URLs.
quality_free_of_price_info: TRUE if the title does NOT mention specific prices, costs, or discounts.
quality_concise_and_scannable: TRUE if the title is relatively short (e.g., under ~15-20 words) and easy to understand quickly.

VERY IMPORTANT: ONLY Return a JSON object containing key-value pairs ONLY for the features that are TRUE. The key MUST be the exact feature name string listed above, and the value MUST be the boolean `true`. Do NOT include features that are FALSE. If NO features are TRUE for the given title, return an empty JSON object: `{}`. Do not include any other text, explanations, or markdown formatting.

Example Output for a title like "Madrid: Skip-the-Line Guided Tour with Royal Palace Ticket":
{
  "clarity_specifies_location": true,
  "clarity_specifies_core_activity": true,
  "clarity_indicates_activity_format": true,
  "value_highlights_access_benefit": true,
  "value_specifies_guidance": true,
  "value_specifies_ticket_inclusion": true,
  "quality_free_of_errors": true,
  "quality_uses_appropriate_caps": true,
  "quality_free_of_contact_info": true,
  "quality_free_of_price_info": true,
  "quality_concise_and_scannable": true
}

Now, process this input title: '''
# --- MODIFIED PROMPT END ---


# Load the CSV file
csv_file = "dataset.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: File not found at {csv_file}")
    exit()

# Define the required columns based on the new V4 features (flattened)
# Using category_feature naming convention
required_columns = [
    "clarity_specifies_location", "clarity_specifies_core_activity", "clarity_indicates_activity_format",
    "clarity_specifies_key_transport", "clarity_indicates_day_trip_origin",
    "value_highlights_access_benefit", "value_specifies_guidance", "value_specifies_ticket_inclusion",
    "value_mentions_food_drink", "value_indicates_target_audience",
    "experience_specifies_group_size", "experience_highlights_time_benefit", "experience_specifies_duration",
    "experience_uses_hook_keywords", "experience_highlights_unique_access",
    "quality_free_of_errors", "quality_uses_appropriate_caps", "quality_free_of_contact_info",
    "quality_free_of_price_info", "quality_concise_and_scannable",
    "processed"
]

# Create a list of only the feature columns (excluding 'processed')
feature_columns = [col for col in required_columns if col != "processed"]

# Add missing columns if they don't exist, initializing with False for boolean features
for col in required_columns:
    if col not in df.columns:
        if col == "processed":
             df[col] = pd.NA # Keep NA for processed initially
        else:
            # Initialize boolean features as False, ensuring boolean dtype
            df[col] = False
            df[col] = df[col].astype('boolean') # Use pandas nullable boolean type

# Ensure 'title' column exists
if 'title' not in df.columns:
    print("Error: 'title' column not found in the CSV.")
    exit()

# Process each row where "processed" is not True (handles NA, "", False, etc.)
# Convert 'processed' column to string temporarily for consistent comparison, handle NA
df['processed'] = df['processed'].astype(str).replace('nan', '')

for index, row in df.loc[df['processed'] != 'True'].iterrows():
    # Input validation: Check if title exists and is a non-empty string
    title = row.get("title")
    if pd.isna(title) or not isinstance(title, str) or title.strip() == "":
        print(f"Skipped row {index + 1}: Title is missing, not a string, or empty")
        continue

    print(f"Processing row {index + 1}: {title[:80]}...") # Print start of title

    # LLM call: Evaluate the title features
    eval_prompt = feature_prompt + title
    response_text = ""
    try:
        response = llm.complete(eval_prompt)
        response_text = response.text

        # Use regex to robustly extract the JSON object {}, as expected now
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Clean potential markdown ```json ... ``` syntax
            json_str = re.sub(r'^```json\s*', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'\s*```$', '', json_str)
            result_dict = json.loads(json_str) # This should now be a dictionary

            # Ensure it's actually a dictionary before proceeding
            if not isinstance(result_dict, dict):
                 raise json.JSONDecodeError(f"Expected a JSON object/dict, but got type {type(result_dict)}", json_str, 0)

            true_features_list = [key for key, value in result_dict.items() if value is True]

            # Reset all feature columns for this row to False initially (important!)
            for col in feature_columns:
                df.at[index, col] = False

            # Set only the TRUE features based on the extracted keys
            for feature_name in true_features_list:
                if feature_name in feature_columns:
                    df.at[index, feature_name] = True
                else:
                    # Optional: Log if the LLM returns an unexpected feature name
                    print(f"Warning in row {index + 1}: LLM returned unexpected feature name '{feature_name}'")

        else:
            print(f"Error in row {index + 1}: No JSON object `{{...}}` found in LLM response. Response was:\n{response_text}")
            continue  # Skip to next row

        # Mark as processed
        df.at[index, "processed"] = "True"

        # Save the updated CSV after each successful row processing
        df.to_csv(csv_file, index=False)

    except json.JSONDecodeError as e:
        print(f"Error in row {index + 1}: Failed to decode JSON object. Error: {e}. Response was:\n{response_text}")
        continue # Skip to next row
    except Exception as e:
        print(f"Error in row {index + 1} during processing: {str(e)}. Response was:\n{response_text}")
        continue # Skip to next row

print("Processing complete!")
