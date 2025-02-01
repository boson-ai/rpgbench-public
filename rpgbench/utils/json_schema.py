import ast
import json
import json_repair
from importlib import resources
from jsonschema import validate, ValidationError, SchemaError


def verify_and_extract(json_string, schema):
    """
    Attempts to parse a JSON string using multiple methods and validates it against the predefined schema.
    
    Parameters:
        json_string (str): The JSON string to be parsed and validated.
    
    Returns:
        tuple: A tuple containing an error and the parsed JSON object. If the error is None, the JSON object is valid.
    
    Raises:
        ValueError: If
            - failed to parse JSON string using available methods
            - the schema is invalid
    """
    if "```json" in json_string:
        json_string = json_string.split("```json")[1].strip()
        json_string = json_string[:-3] if json_string.endswith("```") else json_string
    if isinstance(json_string, str):
        parsing_methods = [json_repair.loads, ast.literal_eval]
        parsed = None
        last_exception = None

        for method in parsing_methods:
            try:
                parsed = method(json_string)
                break
            except Exception as e:
                last_exception = e

        if parsed is None:
            raise ValueError(f"Failed to parse JSON string using available methods: {last_exception}")
    else:
        parsed = json_string

    error = None
    try:
        validate(instance=parsed, schema=schema)
    except SchemaError as se:
        raise ValueError(f"Invalid JSON schema: {se.message}")
    except ValidationError as ve:
        error = ValueError(f"JSON does not match schema: {ve.message}")

    return error, parsed