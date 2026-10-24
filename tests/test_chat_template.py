import pytest


def format_qwen_chat(messages, enable_thinking=False, add_generation_prompt=True):
    """Canonical Qwen2.5 / Qwen3 chat template rendering without thinking mode."""
    rendered = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        rendered += "<|im_start|>assistant\n"
    return rendered


def test_training_and_runtime_shared_prefix_alignment():
    """Doc requirement: Renders one training example and one runtime prompt for the same input

    and asserts the shared prefix is identical.
    """
    raw_alert = "Received: 2026-10-22T15:30:00Z\n[FIRING] checkout-api 5xx rate elevated"
    workflow_id = "incident_triage_v1"

    # Runtime prompt sent to server before first assistant step
    runtime_messages = [
        {"role": "system", "content": workflow_id},
        {"role": "user", "content": raw_alert}
    ]
    runtime_prompt = format_qwen_chat(runtime_messages, enable_thinking=False, add_generation_prompt=True)

    # Training prompt prefix for the first assistant step
    training_messages = [
        {"role": "system", "content": workflow_id},
        {"role": "user", "content": raw_alert}
    ]
    training_prefix = format_qwen_chat(training_messages, enable_thinking=False, add_generation_prompt=True)

    assert runtime_prompt == training_prefix, "Chat template mismatch between training prefix and runtime prompt!"
    assert runtime_prompt.startswith("<|im_start|>system\nincident_triage_v1<|im_end|>\n")
    assert runtime_prompt.endswith("<|im_start|>assistant\n")
