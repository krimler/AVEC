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
import pandas as pd 
from scipy import stats 

# --- Import Prompt Data ---
from prompts_data import ALL_PROMPT_POOLS, DEFAULT_DOMAIN_MIX

# --- Global Log for Simulation Results ---
simulation_results_log = []

# --- Custom Laplace Noise Function (Standard Python) ---
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
    Higher score indicates higher sensitivity. Updated for various contexts from prompts_data.py.
    """
    query_lower = query.lower()

    # Broadened sensitivity keywords to cover all domains from prompts_data.py
    high_sensitivity_keywords = [
        r'\bmy\s+diagnosis\b', r'\bmy\s+medical\s+records\b', r'\bmy\s+treatment\s+plan\b', r'\bi\s+have\s+cancer\b',
        r'\bmy\s+blood\s+test\s+results\b', r'\bmi\s+feel\s+depressed\b', r'\bgenetic\s+disorder\b', r'\bmi\s+am\s+pregnant\b',
        r'\bsymptoms\s+i\s+am\s+experiencing\b', r'\bmy\s+pain\s+level\b', r'\bmy\s+allergy\b', r'\bprescribed\b', # Medical
        r'\bacquisition\b', r'\bbuyout\b', r'\bmerger\b', r'\btarget\s+company\b', r'\bvaluation\b',
        r'\bdue\s+diligence\b', r'\bdeal\s+terms\b', r'\bconfidential\b', r'\bnondisclosure\s+agreement\b',
        r'\binvestment\s+strategy\b', r'\bportfolio\s+holdings\b', r'\binsider\s+information\b', # Financial/M&A
        r'\blegal\s+advice\b', r'\bcourt\s+case\b', r'\bregulation\s+compliance\b', r'\bpatent\s+infringement\b',
        r'\btrade\s+secret\b', r'\bmy\s+contract\b', r'\bdual\s+citizenship\b', # Legal/Policy, Travel (PII/Sensitive Status)
        r'\bmy\s+personal\s+data\b', r'\bsecure\s+way\s+to\s+store\b', r'\bprivate\s+collection\b', # General/Cyber/Creative (Personal/Proprietary)
        r'\bmy\s+family\s+history\b', r'\bmy\s+child\b', r'\bmy\s+property\b', # History/Education/Env (Family/Personal Context)
        r'\bmy\s+company\'s\s+secret\b', r'\bmy\s+unpublished\b', r'\bmy\s+design\s+document\b', # Proprietary/Creative
        r'\bmy\s+personal\s+records\b', r'\bmy\s+account\b', r'\bmy\s+tax\s+filings\b', # General/Finance (personal records)
        r'\bmy\s+credentials\b', r'\bmy\s+password\b', r'\bmy\s+login\b' # Cyber Security (account security)
    ]
    # Medium sensitivity keywords
    medium_sensitivity_keywords = [
        r'\bdiabetes\b', r'\bhypertension\b', r'\bmedication\s+for\b', r'\bdrug\s+interaction\b',
        r'\bheart\s+disease\b', r'\bimmune\s+system\b', r'\bvaccine\b', r'\binflammation\b',
        r'\bsurgery\b', r'\btherapy\b', r'\bside\s+effects\b', # Medical
        r'\bmarket\s+trends\b', r'\bstock\s+performance\b', r'\brevenue\s+growth\b',
        r'\bprofitability\b', r'\bfundraising\b', r'\bcompetitor\s+analysis\b',
        r'\bllm\s+industry\b', r'\bai\s+startup\b', # Financial/M&A
        r'\bpolicy\s+analysis\b', r'\blegislative\s+history\b', r'\bcontract\s+drafting\b', # Legal/Policy
        r'\bclimate\s+change\s+impact\b', r'\brenewable\s+energy\b', r'\bquantum\s+computing\b', # Environmental/Science/Tech
        r'\bnuclear\s+fusion\b', r'\bgene\s+editing\b', # Science/Tech
        r'\bpersonal\s+style\b', r'\bmy\s+home\s+decor\b', r'\bmy\s+personal\s+preferences\b', # Fashion/Self-Improvement (Implied personal taste)
        r'\bmy\s+travel\s+plans\b', r'\bpersonal\s+security\b', # Travel
        r'\bmy\s+training\s+regimen\b', r'\bmy\s+injury\b', r'\bmy\s+fear\b', # Sports/Fitness (personal progress/challenge)
        r'\bmy\s+career\s+plan\b', r'\bpersonal\s+investment\b', r'\bmy\s+job\s+security\b', # Self-Improvement/Finance/AI (personal future/finances)
        r'\bpersonal\s+health\s+goals\b', r'\bmy\s+Browse\s+history\b', # General (personal data implications)
        r'\bmy\s+ancestral\b', r'\bmy\s+tribe\b', # History/Culture (group/family identity)
        r'\bmy\s+academic\s+struggle\b', r'\bmy\s+home-schooling\b', # Education (personal/family learning context)
        r'\bmy\b.+consumption\s+habits\b', r'\bmy\s+property\b', # Environmental (personal environmental impact)
        r'\bmy\s+restaurant\'s\s+sourcing\b', r'\bmy\s+commercial\s+kitchen\b', # Food/Culinary (proprietary business)
        r'\bmy\s+personal\s+debt\b', r'\bmy\s+credit\s+score\b' # Finance (sensitive personal finance)
    ]
    # Low sensitivity keywords (general knowledge, public info, abstract concepts)
    low_sensitivity_keywords = [
        r'\bcapital\s+of\b', r'\bfun\s+fact\b', r'\bcalculate\b', r'\bmeaning\s+of\b',
        r'\blargest\s+ocean\b', r'\btimes\s+\d+\b', r'\binvented\b', r'\bexplain\s+basics\b',
        r'\baverage\s+lifespan\b', r'\bdefine\b', # General
        r'\bhealthy\s+diet\b', r'\bexercise\b', r'\bcommon\s+cold\b', r'\bflu\s+symptoms\b',
        r'\bvitamins\b', r'\bsleep\s+advice\b', r'\bfirst\s+aid\b', r'\bnutrition\b', # Medical
        r'\bgeneral\s+economic\b', r'\btech\s+news\b', r'\bpublic\s+finances\b', # Financial/General
        r'\bhistorical\s+event\b', r'\bcultural\s+tradition\b', r'\bancient\s+civilization\b', # History/Culture
        r'\bwrite\s+a\s+poem\b', r'\bcreative\s+writing\b', r'\bmarketing\s+slogan\b', # Creative/Writing
        r'\bspace\s+exploration\b', r'\buniverse\b', r'\bbasic\s+physics\b', # Science/Tech/Astronomy
        r'\blearning\s+styles\b', r'\bstudy\s+strategies\b', r'\bpersonalize\s+education\b', # Education
        r'\bcircular\s+economy\b', r'\bplastic\s+waste\b', r'\bdeforestation\b', # Environmental
        r'\bsummarize\s+plot\b', r'\bwho\s+was\s+\w+\s+da\s+vinci\b', r'\bexplain\s+concept\s+of\s+surrealism\b', # Arts/Literature
        r'\bprinciples\s+of\s+french\s+cuisine\b', r'\bfermentation\b', r'\btemper\s+chocolate\b', # Food/Culinary
        r'\btourist\s+attractions\b', r'\bnorthern\s+lights\b', r'\bvisa\s+requirements\b', # Travel/Geography
        r'\brules\s+of\s+\w+\b', r'\bbenefits\s+of\s+hiit\b', r'\bimprove\s+stamina\b', # Sports/Fitness
        r'\bfashion\s+trends\b', r'\bprinciples\s+of\s+minimalist\s+design\b', # Fashion/Design
        r'\butilitarianism\b', r'\btrolley\s+problem\b', r'\bexistentialism\b', # Philosophy/Ethics
        r'\bcommon\s+symptoms\s+of\s+anxiety\b', r'\bexplain\s+cbt\b', r'\bmindfulness\s+meditation\b', # Psychology/Mental Health
        r'\bbig\s+bang\b', r'\btypes\s+of\s+galaxies\b', r'\bhow\s+stars\s+form\b', # Space/Astronomy
        r'\b5g\s+technology\b', r'\bssds\b', r'\bvirtual\s+reality\b', # Tech/Gadgets
        r'\besports\s+games\b', r'\bping\s+in\s+online\s+gaming\b', # Gaming/Esports
        r'\bglobalization\b', r'\bgovernment\s+branches\b', r'\bpublic\s+opinion\b', # Social Science/Politics
        r'\bsustainable\s+urban\s+planning\b', r'\bgothic\s+architecture\b', # Architecture/Urban Planning
        r'\bmusical\s+scales\b', r'\bsynthesizers\b', # Music Theory/Production
        r'\btime\s+management\b', r'\bpomodoro\s+technique\b', # Self-Improvement/Productivity
        r'\brobotic\s+arm\b', r'\bethical\s+considerations\s+of\s+autonomous\s+vehicles\b', # Robotics/AI Applied
        r'\bphishing\b', r'\bmultifactor\s+authentication\b', r'\bend-to-end\s+encryption\b' # Cyber Security/Privacy
    ]

    for keyword in high_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.65, 0.98) # Wider range for high sensitivity
    for keyword in medium_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.35, 0.65) # Wider range for medium sensitivity
    for keyword in low_sensitivity_keywords:
        if re.search(keyword, query_lower): return random.uniform(0.08, 0.35) # Wider range for low sensitivity
    return random.uniform(0.03, 0.12) # Wider range for default very low sensitivity


def get_local_agent_confidence(query: str, has_local_cache_hit: bool = False, local_model_capability: str = "basic_qa") -> float:
    """
    Spoofed LLM functionality: Estimates Local_Agent's confidence in answering the query
    based on its *local capabilities* (internal generation or cache).
    Higher confidence means Local_Agent feels it can handle it well.
    """
    query_lower = query.lower()
    
    # Base confidence with more variability introduced
    base_confidence = random.uniform(0.2, 0.7) # Wider initial range for more variability

    if len(query) < 40 and not any(k in query_lower for k in ["medical", "finance", "acquisition", "history", "private jet", "legal", "patent", "my", "personal", "confidential", "secret", "proprietary"]):
        base_confidence += random.uniform(0.05, 0.4) # Wider range for basic queries
        
    if has_local_cache_hit:
        base_confidence = random.uniform(0.85, 0.99) # Very high confidence for cache hits, but still some range
    
    if local_model_capability == "basic_qa" and ("what is" in query_lower or "define" in query_lower):
        base_confidence = max(base_confidence, random.uniform(0.6, 0.9)) # Good for simple Q&A, with more variability
    
    ppo_sens = get_ppo_sensitivity_score(query) 
    base_confidence -= (ppo_sens * random.uniform(0.15, 0.45)) # More sensitive = less confident locally, with wider variability

    return max(0.05, min(1.0, base_confidence + random.uniform(-0.1, 0.1))) # Overall noise/variability increased


class TranslationAgent:
    def transform_request(self, original_query: str, epsilon_budget: float) -> (str, dict, float, float):
        # Simulate cost/latency of Translation Agent's work with significantly more variability
        simulated_latency = random.uniform(0.005, 0.05) + len(original_query) * random.uniform(0.00003, 0.0002) # Wider range for base and length multiplier
        simulated_cost = random.uniform(0.0002, 0.0025) + len(original_query) * random.uniform(0.000003, 0.00002) # Wider range for base and length multiplier
        
        transformed_query = original_query
        transformation_type = "lightly_private"
        
        # Comprehensive redaction lists covering all domains for high privacy
        redaction_patterns = {
            "financial": {
                "Acme Corp": "[TARGET_COMPANY_A]", "TechInnovate Inc.": "[TARGET_COMPANY_B]",
                "Q4 2025": "[FUTURE_DATE_Q4]", "$500 million": "[VALUE_REDACTED]",
                "exclusive negotiation": "[CONFIDENTIAL_NEGOTIATION]", "buyout": "[ACQUISITION_EVENT]",
                r'\bmy\s+portfolio\b': '[USER_PORTFOLIO]', r'\bmy\s+holdings\b': '[USER_HOLDINGS]',
                r'\bmy\s+startup\b': '[USER_STARTUP]'
            },
            "medical": {
                "secret": "[REDACTED_SECRET]", r'\bmy\s+diagnosis\b': '[USER_DIAGNOSIS]',
                r'\bmy\s+medical\s+records\b': '[USER_MEDICAL_RECORDS]', r'\bmy\s+treatment\s+plan\b': '[USER_TREATMENT_PLAN]',
                r'\bmy\s+blood\s+test\s+results\b': '[USER_TEST_RESULTS]', r'\bmy\s+condition\b': '[USER_CONDITION]'
            },
            "legal": {
                r'\bmy\s+court\s+case\b': '[USER_LEGAL_CASE]', r'\bpatent\s+infringement\b': '[IP_VIOLATION_SPECIFIC]',
                r'\btrade\s+secret\b': '[CONFIDENTIAL_IP_SPECIFIC]', r'\bmy\s+contract\b': '[USER_CONTRACT]',
                r'\bmy\s+legal\s+situation\b': '[USER_LEGAL_SITUATION]'
            },
            "personal_info": { # More general PII and personal indicators
                r'\bmy\s+name\b': '[ANONYMIZED_NAME_PREFIX]', r'\bmy\s+email\b': '[ANONYMIZED_EMAIL_PREFIX]',
                r'\bmy\s+phone\b': '[ANONYMIZED_PHONE_PREFIX]', r'\bmy\s+address\b': '[ANONYMIZED_ADDRESS_PREFIX]',
                r'\bi\s+am\s+expecting\b': '[PREGNANCY_STATUS]', r'\bmy\s+child\b': '[CHILD_REFERENCE]',
                r'\bmy\s+family\b': '[FAMILY_REFERENCE]', r'\bmy\s+home\b': '[HOME_REFERENCE]',
                r'\bmy\s+routine\b': '[PERSONAL_ROUTINE]', r'\bmy\s+job\b': '[PERSONAL_JOB]',
                r'\bmy\s+career\b': '[PERSONAL_CAREER]', r'\bmy\s+life\s+choices\b': '[PERSONAL_CHOICES]',
                r'\bmy\s+feelings\b': '[PERSONAL_FEELINGS]', r'\bmy\s+beliefs\b': '[PERSONAL_BELIEFS]',
                r'\bmy\s+ancestral\b': '[FAMILY_HISTORY_REF]', r'\bmy\s+tribe\b': '[TRIBAL_ID]'
            },
            "proprietary_business": {
                r'\bmy\s+startup\b': '[PROPRIETARY_STARTUP]', r'\bmy\s+company\b': '[PROPRIETARY_COMPANY]',
                r'\bnew\s+product\s+launch\b': '[PROPRIETARY_PRODUCT_LAUNCH]', r'\bmy\s+firm\b': '[PROPRIETARY_FIRM]',
                r'\bmy\s+business\s+strategy\b': '[PROPRIETARY_STRATEGY]', r'\binternal\s+polling\s+data\b': '[PROPRIETARY_POLLING]',
                r'\bmy\s+department\b': '[PROPRIETARY_DEPARTMENT]', r'\bmy\s+factory\b': '[PROPRIETARY_FACTORY]'
            }
        }

        if epsilon_budget < 0.1: # Highest Privacy
            for category_patterns in redaction_patterns.values():
                for original, redacted in category_patterns.items():
                    transformed_query = re.sub(original, redacted, transformed_query, flags=re.IGNORECASE)
            
            # More general anonymization
            transformed_query = transformed_query.replace("my", "user's").replace("I have", "The user has")
            # Simple name/number redaction based on typical patterns (more robust in real system)
            transformed_query = re.sub(r'\b[A-Z][a-zA-Z]*\s[A-Z][a-zA-Z]*\b', '[PERSON_NAME_GEN]', transformed_query) 
            transformed_query = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_NUM_GEN]', transformed_query)
            transformed_query = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL_GEN]', transformed_query)
            transformed_query = re.sub(r'\b\d{1,5}\s[A-Za-z0-9\s]+\b(?:street|road|avenue|boulevard|lane|rd|st|ave|blvd|ln)\b', '[ADDRESS_GEN]', transformed_query, flags=re.IGNORECASE)
            
            transformed_query = f"Highly Privatized: {transformed_query}"
            transformation_type = "highly_private"
        elif epsilon_budget < 0.2: # Moderate Privacy (less aggressive redaction)
            for original, redacted in redaction_patterns["personal_info"].items(): # Only anonymize direct personal info
                 transformed_query = re.sub(original, redacted, transformed_query, flags=re.IGNORECASE)
            transformed_query = transformed_query.replace("name", "[ANONYMIZED_NAME]").replace("my", "user's")
            transformed_query = transformed_query.replace("Acme Corp", "[TARGET_COMPANY]") # Example of one specific business redaction
            transformed_query = f"Moderately Privatized: {transformed_query}"
            transformation_type = "moderately_private"
        else: # Light Privacy
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
                log_entry["simulated_total_cost"] = random.uniform(0.0001, 0.001) 
                log_entry["simulated_total_latency"] = random.uniform(0.01, 0.05) 
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
                    log_entry["simulated_total_cost"] = random.uniform(0.001, 0.005) 
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
        # Simulate latency and cost with more variability
        simulated_latency = random.uniform(0.4, 0.6) + len(privacy_enhanced_request) * random.uniform(0.00004, 0.00006) 
        simulated_cost = random.uniform(0.008, 0.012) + len(privacy_enhanced_request) * random.uniform(0.000008, 0.000012)
        
        response_content = ""
        if "Highly Privatized" in privacy_enhanced_request:
            response_content = " (Highly generalized/redacted information due to high privacy requirements.)"
            simulated_cost *= random.uniform(0.4, 0.6) # Cheaper for more privacy, with variability
        elif "Moderately Privatized" in privacy_enhanced_request:
            response_content = " (General information provided with some details.)"
            simulated_cost *= random.uniform(0.7, 0.9)
        elif "Lightly Privatized" in privacy_enhanced_request:
            response_content = " (Detailed and specific information, close to original intent.)"
            simulated_cost *= random.uniform(0.9, 1.1)
        else: # Catch-all for other scenarios, e.g., "Standard" if fixed epsilon
            response_content = " (Standard response based on transformed input.)"
            simulated_cost *= random.uniform(0.95, 1.05) # Full cost

        final_response = f"LLM Response to '{privacy_enhanced_request[:50]}...': {response_content}"
        
        return final_response, simulated_cost, simulated_latency

# --- User-specific Query Generation ---
def generate_user_specific_queries(user_id: str, num_prompts_per_user: int = 10, domain_mix_ratio: dict = None) -> list:
    """
    Generates a unique set of prompts for a user, mixing domains based on ratio.
    Ensures uniqueness within the user's prompt series.
    """
    user_seed = sum(ord(char) for char in user_id)
    current_random_state = random.getstate() 
    random.seed(user_seed) 

    if domain_mix_ratio is None:
        domain_mix_ratio = DEFAULT_DOMAIN_MIX 

    total_ratio = sum(domain_mix_ratio.values())
    if total_ratio == 0: total_ratio = 1 
    normalized_mix = {k: v / total_ratio for k, v in domain_mix_ratio.items()}

    user_prompts = []
    used_prompts_for_user = set()

    for _ in range(num_prompts_per_user):
        chosen_domain = random.choices(
            list(normalized_mix.keys()), 
            weights=list(normalized_mix.values()), 
            k=1
        )[0]
        
        available_for_domain = [p for p in ALL_PROMPT_POOLS.get(chosen_domain, []) if p not in used_prompts_for_user]
        
        if not available_for_domain:
            all_remaining_prompts = []
            for pool_name in ALL_PROMPT_POOLS.keys():
                all_remaining_prompts.extend([p for p in ALL_PROMPT_POOLS[pool_name] if p not in used_prompts_for_user])
            
            if not all_remaining_prompts:
                selected_prompt = random.choice(list(used_prompts_for_user) or ["Default fallback query."])
            else:
                selected_prompt = random.choice(all_remaining_prompts)
        else:
            selected_prompt = random.choice(available_for_domain)
        
        user_prompts.append(selected_prompt)
        used_prompts_for_user.add(selected_prompt)
    
    random.setstate(current_random_state) # Restore global random state
    return user_prompts


# --- POC Orchestration ---
def run_simulation_scenario(client_instance: Client, queries: list, scenario_name: str, trial_num: int, fixed_epsilon_value: float = None):
    """Runs a set of queries for a specific scenario and logs results."""
    client_instance.user_config.query_count_today = 0
    client_instance.user_config.historic_epsilon_spent = 0.0
    client_instance.user_config.last_query_timestamp = None
    
    shuffled_queries = list(queries) 
    random.shuffle(shuffled_queries)

    for i, query in enumerate(shuffled_queries): 
        client_instance.send_query(query, i + 1, scenario_name, fixed_epsilon_value)
        if simulation_results_log: 
            simulation_results_log[-1]["trial_num"] = trial_num
        time.sleep(0.001) 

# --- Main Execution ---
if __name__ == "__main__":
    NUM_TRIALS = 30 
    N_USERS_PER_SCENARIO = 100 
    N_PROMPTS_PER_USER = 10 

    print(f"--- Starting AEPB Simulation Benchmarking with {NUM_TRIALS} trials ---")
    print(f"Each trial involves {N_USERS_PER_SCENARIO} unique users, each submitting {N_PROMPTS_PER_USER} unique prompts.")
    print(f"Total queries per trial per scenario: {N_USERS_PER_SCENARIO * N_PROMPTS_PER_USER}")
    print(f"Total queries across all scenarios and trials: {NUM_TRIALS * N_USERS_PER_SCENARIO * N_PROMPTS_PER_USER * 7}")


    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n--- Running Trial {trial}/{NUM_TRIALS} ---")

        trial_client_profiles = [] 
        for i in range(N_USERS_PER_SCENARIO):
            user_id_base = f"user_{i:03d}_T{trial}"
            pref = random.choice(["High", "Medium", "Low"])
            max_eps = {"High": 0.5, "Medium": 1.5, "Low": 5.0}[pref] 
            trial_client_profiles.append((user_id_base, pref, max_eps))


        # --- Run each scenario for the CURRENT TRIAL'S set of user profiles ---
        
        # Scenario 1: AEPB Dynamic (Your system) - High Privacy User Mix
        print(f"\n-- Scenario: {SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY} --")
        profiles_for_scenario = [p for p in trial_client_profiles if p[1] == "High"]
        if not profiles_for_scenario:
            profiles_for_scenario.append((f"user_AEPB_HighP_Fallback_T{trial}", "High", 0.5))

        for user_profile in profiles_for_scenario:
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"medical": 0.4, "finance": 0.3, "legal": 0.2, "general": 0.1}) 
            run_simulation_scenario(client_instance, user_queries, SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY, trial)


        # Scenario 2: AEPB Dynamic (Your system) - Medium Privacy User Mix
        print(f"\n-- Scenario: {SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY} --")
        profiles_for_scenario = [p for p in trial_client_profiles if p[1] == "Medium"]
        if not profiles_for_scenario:
            profiles_for_scenario.append((f"user_AEPB_MedP_Fallback_T{trial}", "Medium", 1.5))

        for user_profile in profiles_for_scenario:
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"general": 0.4, "creative_writing": 0.3, "science_tech": 0.2, "food_culinary": 0.1}) 
            run_simulation_scenario(client_instance, user_queries, SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY, trial)


        # Scenario 3: Baseline - Always Delegate (No Local First)
        print(f"\n-- Scenario: {SCENARIO_BASELINE_ALWAYS_DELEGATE} --")
        for user_profile in trial_client_profiles: 
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"general": 0.5, "medical": 0.5}) 
            run_simulation_scenario(client_instance, user_queries, SCENARIO_BASELINE_ALWAYS_DELEGATE, trial)


        # Scenario 4: Baseline - Fixed Epsilon (e.g., epsilon=1.0)
        print(f"\n-- Scenario: {SCENARIO_BASELINE_FIXED_EPSILON_1_0} --")
        for user_profile in trial_client_profiles: 
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"general": 0.6, "finance": 0.4})
            run_simulation_scenario(client_instance, user_queries, SCENARIO_BASELINE_FIXED_EPSILON_1_0, trial, fixed_epsilon_value=1.0)
        
        # Scenario 5: Baseline - Fixed Epsilon (e.g., epsilon=0.1 - High Privacy)
        print(f"\n-- Scenario: {SCENARIO_BASELINE_FIXED_EPSILON_0_1} --")
        for user_profile in trial_client_profiles: 
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"medical": 0.7, "legal": 0.3})
            run_simulation_scenario(client_instance, user_queries, SCENARIO_BASELINE_FIXED_EPSILON_0_1, trial, fixed_epsilon_value=0.1)

        # Scenario 6: Baseline - Fixed Epsilon (e.g., epsilon=5.0 - Low Privacy)
        print(f"\n-- Scenario: {SCENARIO_BASELINE_FIXED_EPSILON_5_0} --")
        for user_profile in trial_client_profiles: 
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"creative_writing": 0.5, "general": 0.5})
            run_simulation_scenario(client_instance, user_queries, SCENARIO_BASELINE_FIXED_EPSILON_5_0, trial, fixed_epsilon_value=5.0)

        # Scenario 7: Baseline - Always Local (Never Delegates)
        print(f"\n-- Scenario: {SCENARIO_BASELINE_ALWAYS_LOCAL} --")
        for user_profile in trial_client_profiles: 
            client_instance = Client(user_profile[0], user_profile[1], user_profile[2])
            user_queries = generate_user_specific_queries(client_instance.user_config.user_id, N_PROMPTS_PER_USER, {"general": 0.7, "tech_gadgets": 0.3})
            run_simulation_scenario(client_instance, user_queries, SCENARIO_BASELINE_ALWAYS_LOCAL, trial)


    # --- Aggregation and Statistical Testing ---
    print("\n\n===== SIMULATION COMPLETE: STATISTICAL ANALYSIS =====")
    
    # Create a Pandas DataFrame from the collected results for easier aggregation
    results_df = pd.DataFrame(simulation_results_log)
    
    # --- Table 1: Kruskal-Wallis H-test Results for Average Simulated Total Cost (Delegated Queries) ---
    print("\n\nTable 1: Kruskal-Wallis H-test Results for Average Simulated Total Cost (Delegated Queries)")
    print("-" * 80)

    # Calculate average delegated cost per trial per scenario
    delegated_results_df = results_df[results_df['is_delegated'] == True]
    avg_delegated_cost_per_trial_and_scenario = delegated_results_df.groupby(['scenario_mode', 'trial_num'])['simulated_total_cost'].mean().reset_index()
    
    cost_data_for_kruskal = {}
    all_delegating_scenarios = [
        SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY, SCENARIO_AEPB_DYNAMIC_MEDIUM_PRIVACY,
        SCENARIO_BASELINE_ALWAYS_DELEGATE, SCENARIO_BASELINE_FIXED_EPSILON_1_0,
        SCENARIO_BASELINE_FIXED_EPSILON_0_1, SCENARIO_BASELINE_FIXED_EPSILON_5_0
    ]
    for mode in all_delegating_scenarios:
        costs = avg_delegated_cost_per_trial_and_scenario[avg_delegated_cost_per_trial_and_scenario['scenario_mode'] == mode]['simulated_total_cost'].tolist()
        if costs: 
            cost_data_for_kruskal[mode] = costs

    if len(cost_data_for_kruskal) > 1:
        H_statistic, p_value_kruskal = stats.kruskal(*cost_data_for_kruskal.values())
        print(f"{'Metric':<25} | {'H-statistic':<15} | {'P-value':<15} | {'Significance (p<0.05)':<25}")
        print("-" * 80)
        print(f"{'Avg Delegated Cost':<25} | {H_statistic:<15.4f} | {p_value_kruskal:<15.4f} | {'Yes' if p_value_kruskal < 0.05 else 'No':<25}")
    else:
        print("Not enough groups with delegated cost data to perform Kruskal-Wallis test.")
    print("-" * 80)

    # --- Table 2: Mann-Whitney U-test for Delegation Rate (AEPB HighP vs AlwaysDel) ---
    print("\n\nTable 2: Mann-Whitney U-test Results for Delegation Rate (AEPB HighP vs AlwaysDel)")
    print("-" * 80)

    # We need a list of delegation rates per trial for each of the two groups.
    # AEPB_Dynamic_High_Privacy
    aepb_high_p_delegation_rates = []
    for trial_num in range(1, NUM_TRIALS + 1):
        trial_df = results_df[(results_df['scenario_mode'] == SCENARIO_AEPB_DYNAMIC_HIGH_PRIVACY) & (results_df['trial_num'] == trial_num)]
        if not trial_df.empty:
            delegated_count = trial_df['is_delegated'].sum()
            total_count = len(trial_df)
            aepb_high_p_delegation_rates.append(delegated_count / total_count if total_count > 0 else 0.0)

    # Baseline_AlwaysDelegate
    baseline_always_del_delegation_rates = []
    for trial_num in range(1, NUM_TRIALS + 1):
        trial_df = results_df[(results_df['scenario_mode'] == SCENARIO_BASELINE_ALWAYS_DELEGATE) & (results_df['trial_num'] == trial_num)]
        if not trial_df.empty:
            delegated_count = trial_df['is_delegated'].sum()
            total_count = len(trial_df)
            baseline_always_del_delegation_rates.append(delegated_count / total_count if total_count > 0 else 0.0)

    if aepb_high_p_delegation_rates and baseline_always_del_delegation_rates:
        U_statistic, p_value_u = stats.mannwhitneyu(aepb_high_p_delegation_rates, baseline_always_del_delegation_rates, alternative='less') # 'less' because AEPB expects lower delegation rate
        print(f"{'Test':<30} | {'U-statistic':<15} | {'P-value':<15} | {'Significance (p<0.05)':<25}")
        print("-" * 80)
        print(f"{'Delegation Rate (AEPB HighP vs AlwaysDel)':<30} | {U_statistic:<15.4f} | {p_value_u:<15.4f} | {'Yes' if p_value_u < 0.05 else 'No':<25}")
    else:
        print("Not enough valid data to perform Mann-Whitney U-test for Delegation Rate.")
    print("-" * 80)
    
    print("\n--- End of Statistical Analysis ---")
