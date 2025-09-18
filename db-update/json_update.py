from openai import OpenAI
from dotenv import load_dotenv
import argparse
import prompts
import json
import sys
import os

# Configure OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def openai_request(original_json, user_prompt, model="gpt-5-mini") -> str:
  """
  LLM call with JSON update information to OpenAI.

  params
    original_json (dict): The original JSON data to be updated.
    user-prompt (str): Natural-language instruction describing the edit.
    model (str): The OpenAI model to use for the request.

  returns
    response (openai.Response): The response object from OpenAI
  """
  prompt = f"""
    Original JSON:
    {json.dumps(original_json, indent=2)}

    Instruction:
    {user_prompt}
  """

  response = client.responses.create(
      model=model,
      input=[
          {
              "role": "system",
              "content": prompts.SYSTEM_PROMPT
          },
          {
              "role": "user",
              "content": [
                  { "type": "input_text",
                    "text": prompt,
                  },
              ]
          }
      ]
  )

  return response


def load_json_from_file(path):
  """
  Load JSON from a file.

  params
    path (str): Path to the JSON file.

  returns
    original_json (dict): The loaded JSON data.
  """
  try:
      with open(path, "r", encoding="utf-8") as f:
          original_json = json.load(f)
          print(f"Loaded JSON from: {path}")
          return original_json
  except Exception as e:
      print(f"Failed to read/parse JSON from {path}: {e}", file=sys.stderr)


def write_json_to_file(json_data, path):
  """
  Write JSON to a file.

  params
    json_data (dict): The JSON data to write.
    path (str): Path to the output file.

  returns
    None
  """
  try:
      with open(path, "w", encoding="utf-8") as f:
          json.dump(json_data, f, ensure_ascii=False, indent=2)
      print(f"\nSaved edited JSON to: {path}")
  except Exception as e:
      print(f"Failed to write output file: {e}", file=sys.stderr)


def openai_json_edit(in_path, out_path, user_prompt, model="gpt-5-mini"):
    """
    End-to-end pipeline for JSON editing (load -> LLM call -> write) with OpenAI.

    params
      in_path (str): Path to the input JSON file.
      out_path (str): Path to the output JSON file.
      user_prompt (str): Natural-language instruction describing the edit.

    returns
      None
    """
    original_json = load_json_from_file(in_path)
    request = openai_request(original_json, user_prompt, model)
    updated_json = json.loads(request.output[1].content[0].text)
    write_json_to_file(updated_json, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--in_path', type=str, default="./test-environments/Test_0_json/Level_0.json", help='input json file path to load from')
    parser.add_argument('--out_path', type=str, default="./test-environments/updated_database/Test_0_json/Level_0_out.json", help='output json file path to store to')
    parser.add_argument('--model', type=str, default="gpt-5-mini", help='OpenAI model to use')
    parser.add_argument('--instruction', type=str, help='natural language instruction to edit the input json')
    
    # user_prompt = "The route between airport 1 and airport 7 is unavailable."
    user_prompt = "Each airport with cost higher than 0.6 that airport 1 is connected to is unavailable"
    args = parser.parse_args()

    openai_json_edit(args.in_path, args.out_path, user_prompt, args.model)

