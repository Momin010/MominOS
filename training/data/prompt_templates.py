"""System prompts and prompt builders for synthetic data generation.

Provides structured system prompts for teacher models (OpenRouter / HuggingFace)
to generate high-quality SFT and DPO training data for kernel fault diagnosis.
"""

from typing import Optional, List, Dict


# ── System-level prompts (for teacher API calls) ─────────────────────────

SYSTEM_PROMPT_SFT = """You are a senior Linux kernel engineer with expertise in kernel debugging. Your task is to generate realistic training data for a model that diagnoses kernel faults.

Given a template describing a kernel fault scenario, you must produce:

1. **A detailed diagnostic explanation** — identify the root cause, explain the mechanism, and propose a fix.

The diagnosis must:
- Be technically precise and reference actual kernel APIs, sysctls, or debugfs interfaces
- Follow the standard diagnosis format: symptom → root cause analysis → fix/workaround
- Include specific parameters from the template
- Be 2-5 paragraphs long
- NOT include meta-commentary like "here is your diagnosis" — just the diagnosis itself

Do NOT use markdown formatting for headings. Write in plain text paragraphs."""

SYSTEM_PROMPT_DPO = """You are a senior Linux kernel engineer generating DPO (Direct Preference Optimization) training pairs. Given a kernel fault description, you must produce TWO responses:

**CHOSEN**: The correct root cause diagnosis with proper fix recommendation.
**REJECTED**: A plausible-sounding but INCORRECT or incomplete diagnosis.

The CHOSEN response should:
- Accurately identify the kernel mechanism at play
- Reference relevant code paths, sysctl values, or kernel APIs
- Provide a concrete actionable fix
- Be 2-4 paragraphs

The REJECTED response should:
- Sound credible to a junior engineer but miss the real root cause
- Suggest superficial fixes (add more RAM, upgrade hardware, reboot, reformat)
- Omit the actual kernel mechanism or reference wrong kernel subsystems
- Be 1-3 paragraphs

STOP each section with a clear delimiter '---CHOSEN_END---' or '---REJECTED_END---'."""

SYSTEM_PROMPT_DPO_COMPARE = """You are evaluating two kernel fault diagnoses: one correct (CHOSEN) and one incorrect (REJECTED). 

For each sample, explain concisely:
1. Why the CHOSEN is correct (which kernel mechanism it correctly identifies)
2. Why the REJECTED is wrong (which aspect it gets wrong or oversimplifies)

This will be used as preference data to train a model via DPO."""


# ── Builder functions ────────────────────────────────────────────────────

def build_sft_prompt(template: Dict) -> str:
    """Build the user prompt for SFT generation from a template dict."""
    template_sft = template.get("template_sft", "")
    return (
        f"Fault Type: {template.get('fault_type', 'unknown').replace('_', ' ').title()}\n"
        f"Difficulty: {template.get('difficulty', 'medium')}\n\n"
        f"Scenario:\n{template_sft}\n\n"
        f"Provide a detailed root cause diagnosis and fix recommendation."
    )


def build_dpo_prompt(template: Dict) -> str:
    """Build the user prompt for DPO pair generation from a template dict."""
    return (
        f"Generate CHOSEN and REJECTED responses for this kernel fault.\n\n"
        f"Fault: {template.get('symptom', '')}\n"
        f"Root cause hint: {template.get('root_cause', '')}\n"
        f"Template: {template.get('template_sft', '')}\n\n"
        f"CHOSEN response (correct diagnosis):\n"
    )


def build_dpo_pair_from_templates(template: Dict) -> Dict:
    """Build a structured DPO pair using pre-written templates directly.
    
    This uses the template's built-in chosen/rejected text for local
    generation without needing an API call.
    """
    return {
        "prompt": template.get("template_sft", ""),
        "chosen": template.get("template_dpo_chosen", ""),
        "rejected": template.get("template_dpo_rejected", ""),
        "expected_patterns": template.get("expected_patterns", []),
        "metadata": {
            "id": template.get("id"),
            "fault_type": template.get("fault_type"),
            "difficulty": template.get("difficulty"),
        }
    }


# ── OpenRouter API configuration ────────────────────────────────────────

OPENROUTER_MODELS = {
    "fast": "google/gemini-2.0-flash-001",
    "quality": "anthropic/claude-3.5-sonnet",
    "economy": "meta-llama/llama-3.1-8b-instruct",
}

HUGGINGFACE_MODELS = {
    "fast": "mistralai/Mistral-7B-Instruct-v0.3",
    "quality": "meta-llama/Llama-3.1-70B-Instruct",
    "economy": "HuggingFaceH4/zephyr-7b-beta",
}


def build_api_payload(
    model_id: str,
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict:
    """Build a standard API payload for OpenRouter or HF Inference API."""
    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def build_chat_messages(
    system_prompt: str,
    user_prompt: str,
) -> List[Dict]:
    """Build messages list for chat completion API."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]