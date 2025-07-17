import hashlib
import json

# --- Translation Agent Side ---
def generate_proof_of_transformation(transformation_params: dict) -> str:
    """
    Generate a SHA256 hash over sorted transformation parameters.
    """
    canonical_string = json.dumps(transformation_params, sort_keys=True)
    sha256_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    return sha256_hash

# --- Local Agent Side ---
def verify_proof_of_transformation(received_params: dict, received_hash: str) -> bool:
    """
    Recompute SHA256 hash and verify it matches the received hash.
    """
    canonical_string = json.dumps(received_params, sort_keys=True)
    computed_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    return computed_hash == received_hash

# --- Example Usage ---
transformation_params = {
    "epsilon": 0.1,
    "transformation_type": "redaction",
    "redacted_fields": ["email", "SSN"],
    "timestamp": "2025-07-17T10:45:00Z"
}

# Simulate Translation Agent step
proof = generate_proof_of_transformation(transformation_params)

# Simulate Local Agent step
is_valid = verify_proof_of_transformation(transformation_params, proof)

print(f"Proof: {proof}")
print(f"Is valid: {is_valid}")
