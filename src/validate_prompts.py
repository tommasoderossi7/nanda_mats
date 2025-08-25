#!/usr/bin/env python3
"""
Validate prompt files for step 1-prompts.
Checks schema and verifies 1-token differences between conflicts and siblings.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def count_tokens(text: str) -> int:
    """Simple token counting by whitespace."""
    return len(text.split())


def find_token_difference(text1: str, text2: str) -> int:
    """Find the number of token differences between two texts."""
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    if len(tokens1) != len(tokens2):
        return abs(len(tokens1) - len(tokens2))
    
    differences = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 != t2)
    return differences


def validate_schema(prompts: List[Dict], filename: str) -> bool:
    """Validate that prompts follow the required schema."""
    required_fields = {"id", "family", "text"}
    optional_fields = {"sibling_id"}
    
    print(f"Validating schema for {filename}...")
    
    for i, prompt in enumerate(prompts):
        # Check required fields
        missing_fields = required_fields - set(prompt.keys())
        if missing_fields:
            print(f"  ERROR: Prompt {i} missing required fields: {missing_fields}")
            return False
        
        # Check field types
        if not isinstance(prompt["id"], str):
            print(f"  ERROR: Prompt {i} id must be string, got {type(prompt['id'])}")
            return False
        if not isinstance(prompt["family"], str):
            print(f"  ERROR: Prompt {i} family must be string, got {type(prompt['family'])}")
            return False
        if not isinstance(prompt["text"], str):
            print(f"  ERROR: Prompt {i} text must be string, got {type(prompt['text'])}")
            return False
        
        # Check optional sibling_id
        if "sibling_id" in prompt and not isinstance(prompt["sibling_id"], str):
            print(f"  ERROR: Prompt {i} sibling_id must be string, got {type(prompt['sibling_id'])}")
            return False
    
    print(f"  ✅ Schema validation passed for {filename}")
    return True


def validate_siblings(conflicts: List[Dict], siblings: List[Dict], conflict_file: str, sibling_file: str) -> bool:
    """Validate that each conflict has exactly one sibling with 1-token difference."""
    print(f"Validating siblings between {conflict_file} and {sibling_file}...")
    
    # Create lookup dictionaries
    conflict_lookup = {p["id"]: p for p in conflicts}
    sibling_lookup = {p["sibling_id"]: p for p in siblings}
    
    # Check each conflict has a sibling
    for conflict in conflicts:
        conflict_id = conflict["id"]
        if conflict_id not in sibling_lookup:
            print(f"  ERROR: Conflict {conflict_id} has no sibling")
            return False
        
        sibling = sibling_lookup[conflict_id]
        token_diff = find_token_difference(conflict["text"], sibling["text"])
        
        if token_diff != 1:
            print(f"  ERROR: Conflict {conflict_id} has {token_diff} token differences (expected 1)")
            print(f"    Conflict: {conflict['text']}")
            print(f"    Sibling:  {sibling['text']}")
            return False
    
    # Check each sibling references a valid conflict
    for sibling in siblings:
        if sibling["sibling_id"] not in conflict_lookup:
            print(f"  ERROR: Sibling {sibling['id']} references non-existent conflict {sibling['sibling_id']}")
            return False
    
    print(f"  ✅ Sibling validation passed for {conflict_file} ↔ {sibling_file}")
    return True


def main():
    """Main validation function."""
    prompts_dir = Path("prompts")
    
    # Load all prompt files
    try:
        f1_conflicts = json.loads((prompts_dir / "f1_conflicts.json").read_text())
        f1_siblings = json.loads((prompts_dir / "f1_nonconf_minpairs.json").read_text())
        f2_conflicts = json.loads((prompts_dir / "f2_conflicts.json").read_text())
        f2_siblings = json.loads((prompts_dir / "f2_nonconf_minpairs.json").read_text())
        benign = json.loads((prompts_dir / "benign.json").read_text())
    except FileNotFoundError as e:
        print(f"ERROR: Missing prompt file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in prompt file: {e}")
        sys.exit(1)
    
    # Validate schemas
    schema_ok = (
        validate_schema(f1_conflicts, "f1_conflicts.json") and
        validate_schema(f1_siblings, "f1_nonconf_minpairs.json") and
        validate_schema(f2_conflicts, "f2_conflicts.json") and
        validate_schema(f2_siblings, "f2_nonconf_minpairs.json") and
        validate_schema(benign, "benign.json")
    )
    
    if not schema_ok:
        print("❌ Schema validation failed")
        sys.exit(1)
    
    # Validate siblings
    siblings_ok = (
        validate_siblings(f1_conflicts, f1_siblings, "f1_conflicts.json", "f1_nonconf_minpairs.json") and
        validate_siblings(f2_conflicts, f2_siblings, "f2_conflicts.json", "f2_nonconf_minpairs.json")
    )
    
    if not siblings_ok:
        print("❌ Sibling validation failed")
        sys.exit(1)
    
    # Print summary statistics
    print("\n📊 Summary:")
    print(f"  F1 conflicts: {len(f1_conflicts)}")
    print(f"  F1 siblings: {len(f1_siblings)}")
    print(f"  F2 conflicts: {len(f2_conflicts)}")
    print(f"  F2 siblings: {len(f2_siblings)}")
    print(f"  Benign prompts: {len(benign)}")
    print(f"  Total: {len(f1_conflicts) + len(f1_siblings) + len(f2_conflicts) + len(f2_siblings) + len(benign)}")
    
    print("\n✅ All validations passed!")
    print("OK")


if __name__ == "__main__":
    main() 