import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

from r2egym.agenthub.action import Action


def parse_response_v2(response_text: str) -> Tuple[str, Action]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)
        thought = response_text[:match.start()]
    else:
        thought = response_text
        action = ""

    thought = thought.strip()
    action = action.strip()

    action = Action.from_string(action)

    return thought, action


def extract_actions(input_json: str, output_json: str) -> None:
    """
    Extract actions from assistant messages in a JSON file and save to a new JSON file
    
    Args:
        input_json: Input JSON file path
        output_json: Output JSON file path
    """
    # Read input file
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract message list
    if isinstance(data, list) and len(data) > 0:
        messages = data[0].get('messages', []) if isinstance(data[0], dict) else []
    elif isinstance(data, dict):
        messages = data.get('messages', [])
    else:
        messages = []
    
    # Extract actions from assistant messages
    actions_list = []
    step_idx = 0
    
    for message in messages:
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            
            # Parse thought and action
            thought, action = parse_response_v2(content)
            
            # Generate executable bash command
            if action.function_name == 'execute_bash':
                executable_cmd = action.parameters.get('command', '')
            else:
                executable_cmd = action.to_bashcmd() if action.function_name else ''
            
            # Build action dictionary
            action_dict = {
                'step_idx': step_idx,
                'thought': thought,
                'action_xml': action.to_xml_string() if action.function_name else '',
                'action_function_name': action.function_name,
                'action_parameters': action.parameters if action.function_name else {},
                'executable': executable_cmd,
            }
            
            actions_list.append(action_dict)
            step_idx += 1
    
    # Save to output file
    output_data = {
        'total_actions': len(actions_list),
        'actions': actions_list,
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(actions_list)} actions from {input_json} and saved to {output_json}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract actions from assistant messages in a JSON file")
    parser.add_argument("input", nargs='?', help="Input JSON file path", default="./data/output_converted.json")
    parser.add_argument("output", nargs='?', help="Output JSON file path", default="./data/output_converted_action.json")
    
    args = parser.parse_args()
    extract_actions(args.input, args.output)
