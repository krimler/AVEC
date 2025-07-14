# --- Installation Instructions ---
# To run this code, ensure you have the following libraries installed:
# pip install pandas scipy
#
# Assumption: This code is developed and tested on a Linux (Ubuntu) environment.
# While Python is cross-platform, certain environment-specific behaviors or
# dependency compilations might differ on other operating systems.

import random
import math
import re
from datetime import datetime, timedelta
import time # For time.sleep
from collections import defaultdict 

# --- Statistical Test Imports ---
import pandas as pd # For data manipulation
from scipy import stats # For statistical tests

# --- Global Log for Simulation Results ---
simulation_results_log = []

# --- Custom Laplace Noise Function (Standard Python) ---
# NOTE: For a NeurIPS paper claiming DP, using a verified library like OpenDP
# is highly recommended over this custom implementation for actual deployments.
# This custom function serves the purpose of demonstrating the principle
# when OpenDP is explicitly excluded from dependencies.
def generate_laplace_noise(scale: float) -> float:
    """
    Generates a random number from a Laplace distribution with mean 0 and given scale.
    """
    u = random.uniform(0, 1) - 0.5 
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
                has_local_cache_hit=has_high_confidence_local_response, 
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
def run_simulation_scenario(client_instance: Client, queries: list, scenario_name: str, trial_num: int, fixed_epsilon_value: float = None):
    """Runs a set of queries for a specific scenario and logs results."""
    # Reset user config for each new scenario run for clean comparison within a trial
    client_instance.user_config.query_count_today = 0
    client_instance.user_config.historic_epsilon_spent = 0.0
    client_instance.user_config.last_query_timestamp = None
    
    for i, query in enumerate(queries):
        client_instance.send_query(query, i + 1, scenario_name, fixed_epsilon_value)
        # After send_query, the log_entry is already appended by Local_Agent.process_query
        # We need to add trial_num to the *last* appended entry
        if simulation_results_log: # Ensure log is not empty
            simulation_results_log[-1]["trial_num"] = trial_num
        time.sleep(0.001) 

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

    # --- Number of Trials for Statistical Significance ---
    NUM_TRIALS = 30 # A common number for statistical significance in simulations

    print(f"--- Starting AEPB Simulation Benchmarking with {NUM_TRIALS} trials ---")

    for trial in range(1, NUM_TRIALS + 1):
        # Re-instantiate clients for each trial to ensure fresh state
        client_aepb_high_p = Client(user_id=f"user_AEPB_HighP_T{trial}", privacy_preference="High", epsilon_max_user=0.5)
        run_simulation_scenario(client_aepb_high_p, all_queries, SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY, trial)

        client_aepb_medium_p = Client(user_id=f"user_AEPB_MedP_T{trial}", privacy_preference="Medium", epsilon_max_user=1.5)
        run_simulation_scenario(client_aepb_medium_p, all_queries, SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY, trial)

        client_baseline_always_del = Client(user_id=f"user_Baseline_AlwaysDel_T{trial}", privacy_preference="Medium", epsilon_max_user=1.5)
        run_simulation_scenario(client_baseline_always_del, all_queries, SCENARIO_BASELINE_ALWAYS_DELEGATE, trial)

        client_baseline_fixed_eps_1_0 = Client(user_id=f"user_Baseline_FixedEps_1_0_T{trial}", privacy_preference="Medium", epsilon_max_user=10.0) 
        run_simulation_scenario(client_baseline_fixed_eps_1_0, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_1_0, trial, fixed_epsilon_value=1.0)
        
        client_baseline_fixed_eps_0_1 = Client(user_id=f"user_Baseline_FixedEps_0_1_T{trial}", privacy_preference="High", epsilon_max_user=2.0) 
        run_simulation_scenario(client_baseline_fixed_eps_0_1, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_0_1, trial, fixed_epsilon_value=0.1)

        client_baseline_fixed_eps_5_0 = Client(user_id=f"user_Baseline_FixedEps_5_0_T{trial}", privacy_preference="Low", epsilon_max_user=20.0) 
        run_simulation_scenario(client_baseline_fixed_eps_5_0, all_queries, SCENARIO_BASELINE_FIXED_EPSILON_5_0, trial, fixed_epsilon_value=5.0)

        client_baseline_always_local = Client(user_id=f"user_Baseline_AlwaysLocal_T{trial}", privacy_preference="High", epsilon_max_user=0.5)
        run_simulation_scenario(client_baseline_always_local, all_queries, SCENARIO_BASELINE_ALWAYS_LOCAL, trial)


    # --- Aggregation and Statistical Testing ---
    print("\n\n===== SIMULATION COMPLETE: STATISTICAL ANALYSIS =====")
    
    # Create a Pandas DataFrame from the collected results
    results_df = pd.DataFrame(simulation_results_log)
    
    # --- Table 1: ANOVA Test Results for Average Simulated Total Cost (Delegated Queries) ---
    print("\n\nTable 1: ANOVA Test Results for Average Simulated Total Cost (Delegated Queries)")
    print("-" * 80)
    
    # Filter for only delegated queries (exclude Baseline_AlwaysLocal as it never delegates)
    delegated_results_df = results_df[results_df['is_delegated'] == True]
    
    # Collect cost data per scenario
    cost_data_for_anova = {}
    for scenario_mode in delegated_results_df['scenario_mode'].unique():
        costs = delegated_results_df[delegated_results_df['scenario_mode'] == scenario_mode]['simulated_total_cost'].tolist()
        if costs: # Only add if there's data
            cost_data_for_anova[scenario_mode] = costs
    
    if len(cost_data_for_anova) > 1: # Need at least two groups for ANOVA
        # Perform ANOVA
        F_statistic, p_value = stats.f_oneway(*cost_data_for_anova.values())
        print(f"{'Metric':<25} | {'F-statistic':<15} | {'P-value':<15} | {'Significance (p<0.05)':<25}")
        print("-" * 80)
        print(f"{'Avg Delegated Cost':<25} | {F_statistic:<15.4f} | {p_value:<15.4f} | {'Yes' if p_value < 0.05 else 'No':<25}")
    else:
        print("Not enough groups with delegated cost data to perform ANOVA.")
    print("-" * 80)

    # --- Table 2: Chi-Squared Test Results for Delegation Rate Distribution ---
    print("\n\nTable 2: Chi-Squared Test Results for Delegation Rate Distribution")
    print("-" * 80)

    # Prepare contingency table for Chi-squared test for AEPB_Dynamic_High_Privacy vs. Baseline_AlwaysDelegate
    # We compare two major competing scenarios directly for their delegation behavior.
    
    # Get counts for AEPB_Dynamic_High_Privacy
    aepb_dynamic_high_counts = results_df[results_df['scenario_mode'] == SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY]['is_delegated'].value_counts()
    aepb_dynamic_high_delegated = aepb_dynamic_high_counts.get(True, 0)
    aepb_dynamic_high_local = aepb_dynamic_high_counts.get(False, 0)
    
    # Get counts for Baseline_AlwaysDelegate
    baseline_always_del_counts = results_df[results_df['scenario_mode'] == SCENARIO_BASELINE_ALWAYS_DELEGATE]['is_delegated'].value_counts()
    baseline_always_del_delegated = baseline_always_del_counts.get(True, 0)
    baseline_always_del_local = baseline_always_del_counts.get(False, 0)

    # Create contingency table: [[Delegated_AEPB, Not_Delegated_AEPB], [Delegated_Baseline, Not_Delegated_Baseline]]
    contingency_table_delegation = [
        [aepb_dynamic_high_delegated, aepb_dynamic_high_local],
        [baseline_always_del_delegated, baseline_always_del_local]
    ]

    # Only perform test if there's valid data in the table (e.g., at least one non-zero row/column)
    # and total sum of counts is > 0
    if sum(sum(row) for row in contingency_table_delegation) > 0 and \
       all(sum(row) > 0 for row in contingency_table_delegation) and \
       all(sum(contingency_table_delegation[i][j] for i in range(len(contingency_table_delegation))) > 0 for j in range(len(contingency_table_delegation[0]))):
        
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table_delegation)

        print(f"{'Test':<30} | {'Chi2 Statistic':<15} | {'P-value':<15} | {'Significance (p<0.05)':<25}")
        print("-" * 80)
        print(f"{'Delegation Rate (AEPB HighP vs AlwaysDel)':<30} | {chi2_stat:<15.4f} | {p_value:<15.4f} | {'Yes' if p_value < 0.05 else 'No':<25}")
    else:
        print("Not enough valid data to perform Chi-squared test for Delegation Rate (AEPB HighP vs AlwaysDel).")
        print(f"Contingency Table: {contingency_table_delegation}")

    print("-" * 80)
    
    print("\n--- End of Statistical Analysis ---")
