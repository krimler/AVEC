# --- Installation Instructions ---
# To run this code, ensure you have the following libraries installed:
# pip install pandas
#
# Assumption: This code is developed and tested on a Linux (Ubuntu) environment.
# While Python is cross-platform, certain environment-specific behaviors or
# dependency compilations might differ on other operating systems.

import random
import math
import re
from datetime import datetime, timedelta
import time # For time.sleep

# --- Global Log for Simulation Results ---
simulation_results_log = []

# --- Custom Laplace Noise Function (Standard Python) ---
def generate_laplace_noise(scale: float) -> float:
    """
    Generates a random number from a Laplace distribution with mean 0 and given scale.
    Formula: -scale * sign(u - 0.5) * ln(1 - 2|u - 0.5|) where u is uniform(0,1)
    """
    # Using inverse transform sampling
    u = random.uniform(0, 1) - 0.5 # Shift u to be in [-0.5, 0.5)
    if u >= 0:
        return -scale * math.log(1 - 2 * u)
    else:
        return scale * math.log(1 + 2 * u)

# --- Scenario Mode Constants ---
SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY = "AEPB_Dynamic_High_Privacy"
SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY = "AEPB_Dynamic_Medium_Privacy"
SCENARIO_BASELINE_ALWAYS_DELEGATE = "Baseline_AlwaysDelegate"
SCENARIO_BASELINE_FIXED_EPSILON_1_0 = "Baseline_FixedEpsilon_1.0"
SCENARIO_BASELINE_FIXED_EPSILON_0_1 = "Baseline_FixedEpsilon_0.1"
SCENARIO_BASELINE_FIXED_EPSILON_5_0 = "Baseline_FixedEpsilon_5.0"
SCENARIO_BASELINE_ALWAYS_LOCAL = "Baseline_AlwaysLocal"


# --- 0. UserConfig ---
class UserConfig:
    def __init__(self, user_id: str, privacy_preference: str, epsilon_max_user: float):
        self.user_id = user_id
        self.privacy_preference = privacy_preference
        self.epsilon_max_user = epsilon_max_user
        self.query_count_today = 0
        self.historic_epsilon_spent = 0.0
        self.last_query_timestamp = None

    def reset_daily_metrics(self):
        self.query_count_today = 0
        self.last_query_timestamp = None

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "privacy_preference": self.privacy_preference,
            "epsilon_max_user": self.epsilon_max_user,
            "query_count_today": self.query_count_today,
            "historic_epsilon_spent": self.historic_epsilon_spent,
            "last_query_timestamp": str(self.last_query_timestamp) if self.last_query_timestamp else None
        }

    @classmethod
    def from_dict(cls, data: dict):
        uc = cls(data["user_id"], data["privacy_preference"], data["epsilon_max_user"])
        uc.query_count_today = data.get("query_count_today", 0)
        uc.historic_epsilon_spent = data.get("historic_epsilon_spent", 0.0)
        ts_str = data.get("last_query_timestamp")
        uc.last_query_timestamp = datetime.fromisoformat(ts_str) if ts_str else None
        return uc

# --- Placeholder Functions for LLM/External Logic (Spoofed) ---
def get_ppo_sensitivity_score(query: str) -> float:
    """
    Spoofed LLM functionality: Determines query sensitivity based on keywords.
    Higher score indicates higher sensitivity. Updated for medical and financial contexts.
    """
    query_lower = query.lower()

    # High sensitivity keywords
    high_sensitivity_keywords = [
        r'\bmy\s+diagnosis\b', r'\bmy\s+medical\s+records\b', r'\bmy\s+treatment\s+plan\b',
        r'\bi\s+have\s+cancer\b', r'\bmy\s+blood\s+test\s+results\b', r'\bmi\s+feel\s+depressed\b',
        r'\bgenetic\s+disorder\b', r'\bmi\s+am\s+pregnant\b', r'\bsymptoms\s+i\s+am\s+experiencing\b',
        r'\bmy\s+pain\s+level\b', r'\bmy\s+allergy\b', r'\bprescribed\b', # Medical
        r'\bacquisition\b', r'\bbuyout\b', r'\bmerger\b', r'\btarget\s+company\b', r'\bvaluation\b',
        r'\bdue\s+diligence\b', r'\bdeal\s+terms\b', r'\bconfidential\b', r'\bnondisclosure\s+agreement\b',
        r'\binvestment\s+strategy\b', r'\bportfolio\s+holdings\b', r'\binsider\s+information\b' # Financial/M&A
    ]
    # Medium sensitivity keywords
    medium_sensitivity_keywords = [
        r'\bdiabetes\b', r'\bhypertension\b', r'\bmedication\s+for\b', r'\bdrug\s+interaction\b',
        r'\bheart\s+disease\b', r'\bimmune\s+system\b', r'\bvaccine\b', r'\binflammation\b',
        r'\bsurgery\b', r'\btherapy\b', r'\bside\s+effects\b', # Medical
        r'\bmarket\s+trends\b', r'\bstock\s+performance\b', r'\brevenue\s+growth\b',
        r'\bprofitability\b', r'\bfundraising\b', r'\bcompetitor\s+analysis\b',
        r'\bllm\s+industry\b', r'\bai\s+startup\b' # Financial/M&A
    ]
    # Low sensitivity keywords
    low_sensitivity_keywords = [
        r'\bhealthy\s+diet\b', r'\bexercise\b', r'\bcommon\s+cold\b', r'\bflu\s+symptoms\b',
        r'\bvitamins\b', r'\bsleep\s+advice\b', r'\bfirst\s+aid\b', r'\bnutrition\b',
        r'\bgeneral\s+economic\b', r'\btech\s+news\b'
    ]

    for keyword in high_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.7, 0.95)
    for keyword in medium_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.4, 0.6)
    for keyword in low_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.1, 0.3)
    return random.uniform(0.05, 0.1)


def get_local_agent_confidence(query: str, has_local_cache_hit: bool = False, local_model_capability: str = "basic_qa") -> float:
    """
    Spoofed LLM functionality: Estimates Local_Agent's confidence in answering the query
    based on its *local capabilities* (internal generation or cache).
    Higher confidence means Local_Agent feels it can handle it well.
    """
    query_lower = query.lower()
    
    base_confidence = 0.4 # Default to moderate confidence if no strong signals
    
    if len(query) < 40 and not any(k in query_lower for k in ["medical", "finance", "acquisition", "history", "private jet"]):
        base_confidence += 0.2
        
    if has_local_cache_hit:
        base_confidence = random.uniform(0.85, 0.98) # High confidence if from a trusted cache
    
    if local_model_capability == "basic_qa" and ("what is" in query_lower or "define" in query_lower):
        base_confidence = max(base_confidence, random.uniform(0.7, 0.85))
    
    ppo_sens = get_ppo_sensitivity_score(query) 
    base_confidence -= (ppo_sens * 0.3) # More sensitive = less confident locally

    return max(0.1, min(1.0, base_confidence + random.uniform(-0.05, 0.05)))


class TranslationAgent:
    def transform_request(self, original_query: str, epsilon_budget: float) -> (str, dict, float, float):
        # Simulate cost/latency of Translation Agent's work
        simulated_latency = 0.02 + len(original_query) * 0.0001
        simulated_cost = 0.001 + len(original_query) * 0.00001
        
        transformed_query = original_query
        transformation_type = "lightly_private"
        
        financial_terms_to_redact = {
            "Acme Corp": "[TARGET_COMPANY_A]",
            "TechInnovate Inc.": "[TARGET_COMPANY_B]",
            "Q4 2025": "[FUTURE_DATE_Q4]",
            "$500 million": "[VALUE_REDACTED]",
            "exclusive negotiation": "[CONFIDENTIAL_NEGOTIATION]"
        }
        
        if epsilon_budget < 0.1: 
            for original, redacted in financial_terms_to_redact.items():
                transformed_query = transformed_query.replace(original, redacted)
            
            transformed_query = transformed_query.replace("secret", "[REDACTED_SECRET]").replace("name", "[ANONYMIZED_NAME]").replace("diagnosis", "[DIAGNOSIS_GENERALIZED]").replace("my", "user's").replace("I have", "The user has")

            transformed_query = f"Highly Privatized: {transformed_query}"
            transformation_type = "highly_private"
        elif epsilon_budget < 0.2:
            transformed_query = transformed_query.replace("name", "[ANONYMIZED_NAME]").replace("my", "user's")
            transformed_query = transformed_query.replace("Acme Corp", "[TARGET_COMPANY]") 
            transformed_query = f"Moderately Privatized: {transformed_query}"
            transformation_type = "moderately_private"
        else: 
            transformed_query = f"Lightly Privatized: {original_query}"
            transformation_type = "lightly_private"
        
        proof = {
            "transformation_type": transformation_type,
            "effective_epsilon": epsilon_budget,
            "timestamp": datetime.now().isoformat(),
            "proof_id": f"PROOF_{random.randint(1000, 9999)}"
        }
        
        return transformed_query, proof, simulated_cost, simulated_latency


def verify_proof(proof: dict, original_query: str, transformed_request: str, epsilon_expected: float) -> bool:
    """
    Spoofed Verification: Simulates the Local_Agent checking the Translation Agent's work.
    """
    if not isinstance(proof, dict) or 'transformation_type' not in proof or 'effective_epsilon' not in proof:
        return False
    
    if abs(proof['effective_epsilon'] - epsilon_expected) > 0.01:
        return False

    transformation_type = proof.get('transformation_type', 'unknown')
    
    is_transformed_correctly = False
    if "highly_private" in transformation_type and "Highly Privatized" in transformed_request:
        is_transformed_correctly = True
    elif "moderately_private" in transformation_type and "Moderately Privatized" in transformed_request:
        is_transformed_correctly = True
    elif "lightly_private" in transformation_type and "Lightly Privatized" in transformed_request:
        is_transformed_correctly = True
    else:
        if epsilon_expected < 0.1 and "Highly Privatized" in transformed_request: is_transformed_correctly = True
        elif epsilon_expected < 0.2 and "Moderately Privatized" in transformed_request: is_transformed_correctly = True
        elif epsilon_expected >= 0.2 and "Lightly Privatized" in transformed_request: is_transformed_correctly = True

    return is_transformed_correctly

# --- 1. Client ---
class Client:
    def __init__(self, user_id: str, privacy_preference: str, epsilon_max_user: float):
        self.user_config = UserConfig(user_id, privacy_preference, epsilon_max_user)
        self.local_agent = Local_Agent(self.user_config)
        

    def send_query(self, query_text: str, query_id: int, scenario_mode: str, fixed_epsilon_value: float = None) -> None:
        self.local_agent.process_query(query_text, query_id, scenario_mode, fixed_epsilon_value)

# --- 2. Local_Agent ---
class Local_Agent:
    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
        self.base_epsilon_map = {
            "High": 0.05,
            "Medium": 0.1,
            "Low": 0.3
        }
        self.kappa = 5
        
        self.delta_f_a1_sensitivity = 0.3 
        self.epsilon_eta = 0.01 
        # Using custom Laplace noise generation instead of OpenDP
        self.laplace_noise_scale = self.delta_f_a1_sensitivity / self.epsilon_eta

        self.translation_agent = TranslationAgent() 
        self.remote_agent = RemoteAgent() 

        self.local_cache = {
            "what is the capital of australia?": "The capital of Australia is Canberra.",
            "tell me a fun fact about giraffes.": "Giraffes have the same number of neck vertebrae as humans (seven), but each one is much, much longer!",
            "calculate 123 * 45.": "123 multiplied by 45 is 5535.",
            "what's the meaning of the word 'ephemeral'?": "Ephemeral means lasting for a very short time."
        }
        self.local_model_capability = "basic_qa" 


    def _update_and_get_query_count(self) -> int:
        """Handles daily reset of query count and increments for the current query."""
        if self.user_config.last_query_timestamp and \
           (datetime.now() - self.user_config.last_query_timestamp) > timedelta(days=1):
            self.user_config.reset_daily_metrics()
        self.user_config.query_count_today += 1
        return self.user_config.query_count_today

    def _get_base_epsilon(self) -> float:
        return self.base_epsilon_map.get(self.user_config.privacy_preference, 0.1)

    def _attempt_local_fulfillment(self, client_query: str) -> (str, bool):
        """
        Attempts to fulfill the query locally from cache or internal model.
        Returns (response, has_high_confidence_local_response)
        """
        query_lower = client_query.lower()
        
        if query_lower in self.local_cache:
            return self.local_cache[query_lower], True 
        
        if self.local_model_capability == "basic_qa" and ("what is" in query_lower or "define" in query_lower):
            if "capital of" in query_lower:
                return f"Locally processed: The capital of some place is [Unknown City].", False 
            if "meaning of" in query_lower:
                 return f"Locally processed: The meaning of '{query_lower.split('meaning of')[-1].strip()}' is [Unknown Meaning].", False
            
        return None, False


    def _calculate_aepb_deterministic_value(self, client_query: str, local_agent_confidence: float) -> float:
        """Computes the deterministic part of AEPB (f_a1_deterministic)."""
        delta_epsilon_base = self._get_base_epsilon()
        query_sequence_factor = math.exp(-self.user_config.query_count_today / self.kappa)

        ppo_sensitivity_score = get_ppo_sensitivity_score(client_query)
        
        f_a1_deterministic = (
            delta_epsilon_base *
            ppo_sensitivity_score *
            (1 - local_agent_confidence) * query_sequence_factor
        )
        return f_a1_deterministic

    def _apply_aepb_noise(self, f_a1_deterministic: float) -> float:
        """
        Applies Laplace noise to the deterministic AEPB value.
        Uses custom function.
        """
        delta_epsilon_proposed = f_a1_deterministic + generate_laplace_noise(self.laplace_noise_scale)

        delta_epsilon_proposed = max(0.0, delta_epsilon_proposed)
        delta_epsilon_proposed = min(delta_epsilon_proposed, self.user_config.epsilon_max_user / 2) 

        return delta_epsilon_proposed

    def _apply_local_budget_cap(self, delta_epsilon_proposed: float) -> float:
        """Manages historic_epsilon_spent and enforces epsilon_max_user."""
        epsilon_remaining = self.user_config.epsilon_max_user - self.user_config.historic_epsilon_spent
        epsilon_query = min(delta_epsilon_proposed, epsilon_remaining)
        epsilon_query = max(0.0, epsilon_query) 

        return epsilon_query

    def _interact_with_translation_agent(self, client_query: str, epsilon_query: float) -> (str, dict, float, float):
        """Handles sending query to TranslationAgent and receiving response."""
        return self.translation_agent.transform_request(client_query, epsilon_query)

    def _verify_and_send_to_remote(self, client_query: str, privacy_enhanced_request: str, proof: dict, epsilon_query: float) -> (str, bool, float, float):
        """Performs proof verification and sends to RemoteAgent."""
        verification_success = verify_proof(proof, client_query, privacy_enhanced_request, epsilon_query)
        if verification_success:
            final_response, remote_cost, remote_latency = self.remote_agent.process_request(privacy_enhanced_request, epsilon_query)
            return final_response, verification_success, remote_cost, remote_latency
        else:
            return "Error: Privacy transformation verification failed.", False, 0.0, 0.0

    def process_query(self, client_query: str, query_id: int, scenario_mode: str, fixed_epsilon_value: float = None) -> None:
        """Orchestrates the entire query processing pipeline within the Local_Agent."""
        
        log_entry = {
            "query_id": query_id,
            "user_id": self.user_config.user_id,
            "scenario_mode": scenario_mode,
            "query_text": client_query,
            "epsilon_used_by_remote": 0.0,
            "historic_epsilon_after_query": 0.0,
            "is_delegated": False,
            "response_quality_type": "N/A",
            "simulated_total_cost": 0.0,
            "simulated_total_latency": 0.0,
            "verification_status": "N/A",
            "final_response": ""
        }

        self._update_and_get_query_count() # Always update query count for sequential factor

        local_response = None
        current_epsilon_query = 0.0 # Initialize

        # --- Scenario-specific logic for delegation ---
        if scenario_mode == SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY or \
           scenario_mode == SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY:
            # AEPB's default behavior: local-first, then delegate
            local_response, has_high_confidence_local_response = self._attempt_local_fulfillment(client_query)
            local_agent_confidence_score = get_local_agent_confidence(
                client_query, 
                has_local_cache_hit=has_high_confidence_local_response, # Corrected keyword argument name
                local_model_capability=self.local_model_capability
            )

            if local_agent_confidence_score >= 0.8: 
                log_entry["is_delegated"] = False
                log_entry["response_quality_type"] = "Local (High Confidence)"
                log_entry["simulated_total_cost"] = random.uniform(0.0001, 0.001) # Very low cost
                log_entry["simulated_total_latency"] = random.uniform(0.01, 0.05) # Very low latency
                log_entry["verification_status"] = "N/A (Local)"
                log_entry["final_response"] = local_response if local_response else f"Locally handled: '{client_query}'"
                simulation_results_log.append(log_entry)
                return

            else: # Low confidence locally, proceed with AEPB and delegation
                f_a1_deterministic = self._calculate_aepb_deterministic_value(client_query, local_agent_confidence_score)
                delta_epsilon_proposed = self._apply_aepb_noise(f_a1_deterministic)
                current_epsilon_query = self._apply_local_budget_cap(delta_epsilon_proposed)

                if current_epsilon_query <= 0.001: # Effectively zero budget
                    log_entry["is_delegated"] = True # Attempted delegation
                    log_entry["response_quality_type"] = "Remote (Budget Exhausted/Highly Privatized)"
                    log_entry["final_response"] = f"Locally handled (max privacy): Cannot provide specific detail for '{client_query}' due to exhausted privacy budget."
                    log_entry["simulated_total_cost"] = random.uniform(0.001, 0.005) # Small cost for failed attempt
                    log_entry["simulated_total_latency"] = random.uniform(0.1, 0.3)
                    log_entry["verification_status"] = "N/A (Budget)"
                    simulation_results_log.append(log_entry)
                    return
        
        elif scenario_mode == SCENARIO_BASELINE_ALWAYS_DELEGATE:
            local_agent_confidence_score = get_local_agent_confidence(client_query, has_local_cache_hit=False, local_model_capability="none") 
            f_a1_deterministic = self._calculate_aepb_deterministic_value(client_query, local_agent_confidence_score) 
            delta_epsilon_proposed = self._apply_aepb_noise(f_a1_deterministic) 
            current_epsilon_query = self._apply_local_budget_cap(delta_epsilon_proposed) 
            log_entry["is_delegated"] = True

        elif scenario_mode == SCENARIO_BASELINE_FIXED_EPSILON_1_0 or \
             scenario_mode == SCENARIO_BASELINE_FIXED_EPSILON_0_1 or \
             scenario_mode == SCENARIO_BASELINE_FIXED_EPSILON_5_0:
            current_epsilon_query = fixed_epsilon_value if fixed_epsilon_value is not None else 1.0
            self.user_config.historic_epsilon_spent += current_epsilon_query
            self.user_config.last_query_timestamp = datetime.now()
            log_entry["is_delegated"] = True

        elif scenario_mode == SCENARIO_BASELINE_ALWAYS_LOCAL:
            local_response, has_high_confidence_local_response = self._attempt_local_fulfillment(client_query)
            local_agent_confidence_score = get_local_agent_confidence(
                client_query, 
                has_local_cache_hit=has_high_confidence_local_response, 
                local_model_capability=self.local_model_capability
            )
            log_entry["is_delegated"] = False
            log_entry["epsilon_used_by_remote"] = 0.0
            log_entry["response_quality_type"] = "Local (Attempted)"
            log_entry["simulated_total_cost"] = random.uniform(0.0001, 0.001) 
            log_entry["simulated_total_latency"] = random.uniform(0.01, 0.05) 
            log_entry["verification_status"] = "N/A (Local)"
            if has_high_confidence_local_response:
                log_entry["final_response"] = local_response
                log_entry["response_quality_type"] = "Local (Success)"
            else:
                log_entry["final_response"] = f"Local attempt: Could not fulfill '{client_query}' with high confidence locally."
                log_entry["response_quality_type"] = "Local (Failure)"
            simulation_results_log.append(log_entry)
            return
        
        else:
            raise ValueError(f"Unknown scenario_mode: {scenario_mode}")

        # --- Common delegation path for all non-AlwaysLocal scenarios ---
        log_entry["is_delegated"] = True
        log_entry["epsilon_used_by_remote"] = current_epsilon_query
        
        privacy_enhanced_request, proof, ta_cost, ta_latency = self._interact_with_translation_agent(client_query, current_epsilon_query)
        
        final_response, verification_success, remote_cost, remote_latency = self._verify_and_send_to_remote(
            client_query, privacy_enhanced_request, proof, current_epsilon_query
        )
        
        log_entry["simulated_total_cost"] = ta_cost + remote_cost
        log_entry["simulated_total_latency"] = ta_latency + remote_latency
        log_entry["verification_status"] = "Verified" if verification_success else "Failed Verification"
        log_entry["final_response"] = final_response
        
        if "Highly Privatized" in final_response:
            log_entry["response_quality_type"] = "Remote (Highly Privatized)"
        elif "Moderately Privatized" in final_response:
            log_entry["response_quality_type"] = "Remote (Moderately Privatized)"
        elif "Lightly Privatized" in final_response:
            log_entry["response_quality_type"] = "Remote (Lightly Privatized)"
        elif "Error" in final_response:
            log_entry["response_quality_type"] = "Remote (Error)"
        else:
            log_entry["response_quality_type"] = "Remote (Standard)"

        log_entry["historic_epsilon_after_query"] = self.user_config.historic_epsilon_spent
        simulation_results_log.append(log_entry)


# --- 4. Remote Agent (Standard LLM Inference) ---
class RemoteAgent:
    def process_request(self, privacy_enhanced_request: str, epsilon_budget: float) -> (str, float, float):
        simulated_latency = 0.5 + len(privacy_enhanced_request) * 0.00005 
        simulated_cost = 0.01 + len(privacy_enhanced_request) * 0.00001 
        
        response_content = ""
        if "Highly Privatized" in privacy_enhanced_request:
            response_content = " (Highly generalized/redacted information due to high privacy requirements.)"
            simulated_cost *= 0.5 
        elif "Moderately Privatized" in privacy_enhanced_request:
            response_content = " (General information provided with some details.)"
        elif "Lightly Privatized" in privacy_enhanced_request:
            response_content = " (Detailed and specific information, close to original intent.)"
        else:
            response_content = " (Standard response based on transformed input.)"
            
        final_response = f"LLM Response to '{privacy_enhanced_request[:50]}...': {response_content}"
        
        return final_response, simulated_cost, simulated_latency

# --- POC Orchestration ---
def run_simulation_scenario(client_instance: Client, queries: list, scenario_name: str, fixed_epsilon_value: float = None):
    """Runs a set of queries for a specific scenario and logs results."""
    print(f"\n===== Running Scenario: {scenario_name} =====")
    # Reset user config for each new scenario run for clean comparison
    client_instance.user_config.query_count_today = 0
    client_instance.user_config.historic_epsilon_spent = 0.0
    client_instance.user_config.last_query_timestamp = None
    
    for i, query in enumerate(queries):
        client_instance.send_query(query, i + 1, scenario_name, fixed_epsilon_value)
        time.sleep(0.01) 
    print(f"===== Scenario {scenario_name} Completed. =====")

# --- Main Execution ---
if __name__ == "__main__":
    # Define common query sets for consistent benchmarking
    general_queries = [
        "What is the capital of Australia?", 
        "Tell me a fun fact about giraffes.", 
        "Calculate 123 * 45.", 
        "What's the meaning of the word 'ephemeral'?", 
        "What is the largest ocean?", 
        "What is 7 times 8?", 
    ]

    medical_queries_high_sens = [
        "My recent blood test results show elevated liver enzymes. What could this indicate?",
        "Can you explain the side effects of Metformin, as I've just started taking it?",
        "I'm feeling persistently fatigued and experiencing muscle weakness. What non-private factors should I consider exploring with my doctor?",
        "I've been prescribed Citalopram for anxiety. What are the common benefits and risks?",
        "My recent diagnosis of hypertension requires lifestyle changes. What general dietary advice do you have?"
    ]

    finance_queries_high_sens = [
        "What is the market capitalization of Acme Corp and its recent stock performance?",
        "I need a deep analysis of TechInnovate Inc.'s intellectual property portfolio ahead of a potential buyout.",
        "Assess the current valuation methodologies for private LLM startups aiming for a $500 million buyout.",
        "What are the risks associated with an exclusive negotiation period with Acme Corp regarding a merger by [FUTURE_DATE_Q4]?",
    ]
    
    all_queries = general_queries + medical_queries_high_sens + finance_queries_high_sens

    # --- SCENARIOS ---
    print("--- Starting AEPB Simulation Benchmarking ---")

    # Scenario 1: AEPB Dynamic (Your system) - High Privacy User
    client_aepb_high_p = Client(user_id="user_AEPB_HighP", privacy_preference="High", epsilon_max_user=0.5)
    run_simulation_scenario(client_aepb_high_p, all_queries, SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY)

    # Scenario 2: AEPB Dynamic (Your system) - Medium Privacy User
    client_aepb_medium_p = Client(user_id="user_AEPB_MedP", privacy_preference="Medium", epsilon_max_user=1.5)
    run_simulation_scenario(client_aepb_medium_p, all_queries, SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY)

    # Scenario 3: Baseline - Always Delegate (No Local First)
    client_baseline_always_del = Client(user_id="user_Baseline_AlwaysDel", privacy_preference="Medium", epsilon_max_user=1.5)
    run_simulation_scenario(client_baseline_always_del, all_queries, SCENARIO_BASELINE_ALWAYS_DELEGATE)

    # Scenario 4: Baseline - Fixed Epsilon (e.g., epsilon=1.0)
    client_baseline_fixed_eps_1_0 = Client(user_id="user_Baseline_FixedEps_1_0", privacy_preference="Medium", epsilon_max_user=10.0) 
    run_simulation_scenario(client_baseline_fixed_eps_1_0, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_1_0, fixed_epsilon_value=1.0)
    
    # Scenario 5: Baseline - Fixed Epsilon (e.g., epsilon=0.1 - High Privacy)
    client_baseline_fixed_eps_0_1 = Client(user_id="user_Baseline_FixedEps_0_1", privacy_preference="High", epsilon_max_user=2.0) 
    run_simulation_scenario(client_baseline_fixed_eps_0_1, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_0_1, fixed_epsilon_value=0.1)

    # Scenario 6: Baseline - Fixed Epsilon (e.g., epsilon=5.0 - Low Privacy)
    client_baseline_fixed_eps_5_0 = Client(user_id="user_Baseline_FixedEps_5_0", privacy_preference="Low", epsilon_max_user=20.0) 
    run_simulation_scenario(client_baseline_fixed_eps_5_0, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_5_0, fixed_epsilon_value=5.0)

    # Scenario 7: Baseline - Always Local (Never Delegates)
    client_baseline_always_local = Client(user_id="user_Baseline_AlwaysLocal", privacy_preference="High", epsilon_max_user=0.5)
    run_simulation_scenario(client_baseline_always_local, all_queries, SCENARIO_BASELINE_ALWAYS_LOCAL)


    # --- Generate and Display Results ---
    print("\n\n===== SIMULATION COMPLETE: ALL RESULTS =====")
    
    # Manual DataFrame-like printing and analysis
    def print_results_table(results: list[dict]):
        if not results:
            print("No results to display.")
            return

        # Prepare headers and ensure consistent order
        all_keys = set()
        for row in results:
            all_keys.update(row.keys())
        # Define a desired order for readability
        headers_order = [
            "query_id", "user_id", "scenario_mode", "query_text", 
            "is_delegated", "response_quality_type", "epsilon_used_by_remote", 
            "historic_epsilon_after_query", "simulated_total_cost", 
            "simulated_total_latency", "verification_status", "final_response"
        ]
        headers = [h for h in headers_order if h in all_keys]
        # Add any remaining keys at the end if they weren't in desired order
        remaining_headers = sorted(list(all_keys - set(headers_order)))
        headers.extend(remaining_headers)


        # Calculate column widths for pretty printing
        column_widths = {header: len(header) for header in headers}
        for row in results:
            for header in headers:
                val = row.get(header, "N/A")
                if isinstance(val, float):
                    column_widths[header] = max(column_widths[header], len(f"{val:.4f}"))
                elif isinstance(val, str) and len(val) > 30: # Truncate long strings for display
                    column_widths[header] = max(column_widths[header], 30)
                else:
                    column_widths[header] = max(column_widths[header], len(str(val)))
        
        # Print headers
        header_line = " | ".join(f"{h:<{column_widths[h]}}" for h in headers)
        print(header_line)
        print("-" * len(header_line))

        for row in results:
            row_str = []
            for h in headers:
                val = row.get(h, "N/A")
                if isinstance(val, float):
                    cell_val = f"{val:<{column_widths[h]}.4f}"
                elif isinstance(val, str) and len(val) > 30:
                    cell_val = f"{val[:27]}...{ ' ' * (column_widths[h] - 30)}" # Truncate and pad
                else:
                    cell_val = f"{str(val):<{column_widths[h]}}"
                row_str.append(cell_val)
            print(" | ".join(row_str))

    print_results_table(simulation_results_log)

    # --- Basic Analysis Examples (for your paper) ---
    print("\n\n===== BASIC ANALYSIS =====")

    # Group results by scenario_mode for aggregation
    results_by_scenario = {}
    for entry in simulation_results_log:
        mode = entry["scenario_mode"]
        if mode not in results_by_scenario:
            results_by_scenario[mode] = []
        results_by_scenario[mode].append(entry)

    # Delegation Rate
    print("\nDelegation Rate per Scenario (%):")
    for mode, entries in results_by_scenario.items():
        total_queries = len(entries)
        if total_queries == 0:
            print(f"  {mode}: No queries.")
            continue
        delegated_count = sum(1 for e in entries if e["is_delegated"])
        rate = (delegated_count / total_queries) * 100
        print(f"  {mode}: {rate:.2f}%")

    # Average Epsilon by Scenario for delegated queries
    print("\nAverage Epsilon Used (Only for DELEGATED Queries) per Scenario:")
    for mode, entries in results_by_scenario.items():
        delegated_queries = [e for e in entries if e["is_delegated"] and e["epsilon_used_by_remote"] > 0]
        if not delegated_queries:
            print(f"  {mode}: No delegated queries with epsilon.")
            continue
        avg_eps = sum(e["epsilon_used_by_remote"] for e in delegated_queries) / len(delegated_queries)
        print(f"  {mode}: {avg_eps:.4f}")

    # Total Epsilon Spent per User/Scenario (Cumulative for remote calls)
    print("\nTotal Epsilon Spent per User/Scenario (Cumulative for remote calls):")
    for mode, entries in results_by_scenario.items():
        user_total_eps = {}
        for e in entries:
            user_id = e["user_id"]
            if user_id not in user_total_eps:
                user_total_eps[user_id] = 0.0
            user_total_eps[user_id] += e["epsilon_used_by_remote"]
        for user_id, total_eps in user_total_eps.items():
            print(f"  {mode} ({user_id}): {total_eps:.4f}")

    # Average Cost and Latency by Scenario (only for delegated queries)
    print("\nAverage Cost and Latency for DELEGATED Queries per Scenario:")
    for mode, entries in results_by_scenario.items():
        delegated_queries = [e for e in entries if e["is_delegated"]]
        if not delegated_queries:
            print(f"  {mode}: No delegated queries.")
            continue
        avg_cost = sum(e["simulated_total_cost"] for e in delegated_queries) / len(delegated_queries)
        avg_latency = sum(e["simulated_total_latency"] for e in delegated_queries) / len(delegated_queries)
        print(f"  {mode}: Avg Cost={avg_cost:.4f}, Avg Latency={avg_latency:.4f}")

    # Response Quality Distribution
    print("\nResponse Quality Distribution per Scenario:")
    for mode, entries in results_by_scenario.items():
        quality_counts = {}
        for e in entries:
            quality_type = e["response_quality_type"]
            quality_counts[quality_type] = quality_counts.get(quality_type, 0) + 1
        print(f"  {mode}: {quality_counts}")

    print("\n--- End of Simulation Analysis ---")
