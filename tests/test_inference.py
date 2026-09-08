"""Unit tests for the masters.inference module."""

import json
import tempfile
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from masters.inference import (
    InferenceAuthError,
    InferenceConfig,
    InferenceConnectionError,
    InferenceRateLimitError,
    InferenceResponseError,
    InferenceTimeoutError,
    MockInferenceClient,
    OllamaClient,
    OpenAICompatibleClient,
    get_inference_client,
)
from masters.sandbox import execute_code, extract_python_code
from masters.tasks import ImputationTask, inject_mcar


def test_inference_config_valid() -> None:
    config = InferenceConfig(
        model="qwen2.5-coder:7b",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        seed=42,
        max_tokens=1024,
        timeout_sec=30.0,
    )
    assert config.model == "qwen2.5-coder:7b"
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.top_k == 40
    assert config.seed == 42
    assert config.max_tokens == 1024
    assert config.timeout_sec == 30.0


def test_inference_config_invalid() -> None:
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig(model="m", temperature=-0.1)
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig(model="m", temperature=2.5)
    with pytest.raises(ValueError, match="top_p"):
        InferenceConfig(model="m", top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        InferenceConfig(model="m", top_p=1.2)
    with pytest.raises(ValueError, match="top_k"):
        InferenceConfig(model="m", top_k=0)
    with pytest.raises(ValueError, match="max_tokens"):
        InferenceConfig(model="m", max_tokens=0)
    with pytest.raises(ValueError, match="timeout_sec"):
        InferenceConfig(model="m", timeout_sec=0.0)


def test_mock_client_generate() -> None:
    client = MockInferenceClient()
    config = InferenceConfig(model="mock-model", seed=10)

    res = client.generate("Write code to impute_missing data", config)
    assert client.provider_name == "mock"
    assert client.is_available() is True
    assert "def impute_missing" in res.content
    assert res.model == "mock-model"
    assert res.latency_sec >= 0.0
    assert res.prompt_tokens is not None and res.prompt_tokens > 0
    assert res.completion_tokens is not None and res.completion_tokens > 0
    assert res.total_tokens == res.prompt_tokens + res.completion_tokens


def test_mock_client_canned_responses() -> None:
    canned = ["Response A", "Response B"]
    client = MockInferenceClient(canned_responses=canned)
    config = InferenceConfig(model="test")

    res1 = client.generate("prompt 1", config)
    res2 = client.generate("prompt 2", config)
    res3 = client.generate("prompt 3", config)

    assert res1.content == "Response A"
    assert res2.content == "Response B"
    assert res3.content == "Response A"


@respx.mock
def test_ollama_client_success() -> None:
    mock_payload = {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": "```python\nx = 1\n```"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "prompt_eval_count": 45,
        "eval_count": 80,
    }
    respx.post("http://localhost:11434/api/chat").respond(
        status_code=200, json=mock_payload
    )

    client = OllamaClient(base_url="http://localhost:11434")
    config = InferenceConfig(
        model="qwen2.5-coder:7b",
        temperature=0.2,
        seed=123,
    )

    resp = client.generate("Generate a script", config)
    assert resp.content == "```python\nx = 1\n```"
    assert resp.model == "qwen2.5-coder:7b"
    assert resp.prompt_tokens == 45
    assert resp.completion_tokens == 80
    assert resp.total_tokens == 125
    assert resp.latency_sec == 2.0
    assert resp.finish_reason == "stop"


@respx.mock
def test_ollama_client_connection_error() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    client = OllamaClient(base_url="http://localhost:11434")
    config = InferenceConfig(model="qwen2.5-coder:7b")

    with pytest.raises(InferenceConnectionError, match="Ensure the Ollama daemon"):
        client.generate("hello", config)


@respx.mock
def test_ollama_client_timeout_error() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.TimeoutException("Read timed out")
    )
    client = OllamaClient(base_url="http://localhost:11434")
    config = InferenceConfig(model="qwen2.5-coder:7b", timeout_sec=10.0)

    with pytest.raises(InferenceTimeoutError, match="timed out"):
        client.generate("hello", config)


@respx.mock
def test_ollama_client_model_not_found() -> None:
    respx.post("http://localhost:11434/api/chat").respond(
        status_code=404, text="model not found"
    )
    client = OllamaClient(base_url="http://localhost:11434")
    config = InferenceConfig(model="nonexistent-model")

    with pytest.raises(InferenceResponseError, match="ollama pull"):
        client.generate("hello", config)


@respx.mock
def test_ollama_is_available() -> None:
    respx.get("http://localhost:11434/api/tags").respond(status_code=200, json={})
    client = OllamaClient(base_url="http://localhost:11434")
    assert client.is_available() is True

    respx.get("http://localhost:11434/api/tags").respond(status_code=500)
    assert client.is_available() is False


@respx.mock
def test_openai_compatible_client_success() -> None:
    mock_payload = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "print('ok')"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 5,
            "total_tokens": 20,
        },
        "system_fingerprint": "fp_cloud_123",
    }
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200, json=mock_payload
    )

    client = OpenAICompatibleClient(
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
    )
    config = InferenceConfig(model="gpt-4o-mini", temperature=0.5, seed=42)

    resp = client.generate("Hi", config)
    assert resp.content == "print('ok')"
    assert resp.model == "gpt-4o-mini"
    assert resp.prompt_tokens == 15
    assert resp.completion_tokens == 5
    assert resp.total_tokens == 20
    assert resp.system_fingerprint == "fp_cloud_123"

    # Verify authorization header and payload
    sent_request = route.calls.last.request
    assert sent_request.headers["authorization"] == "Bearer sk-test-key"
    sent_json = json.loads(sent_request.content)
    assert sent_json["temperature"] == 0.5
    assert sent_json["seed"] == 42


@respx.mock
def test_openai_compatible_client_auth_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=401, text="Unauthorized: Invalid API key"
    )
    client = OpenAICompatibleClient(api_key="bad-key")
    config = InferenceConfig(model="gpt-4o-mini")

    with pytest.raises(InferenceAuthError, match="Authentication failed"):
        client.generate("hello", config)


@respx.mock
def test_openai_compatible_client_rate_limit_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=429, text="Rate limit reached"
    )
    # Configure 1 retry for fast testing
    client = OpenAICompatibleClient(api_key="test-key", max_retries=1)
    config = InferenceConfig(model="gpt-4o-mini")

    with pytest.raises(InferenceRateLimitError, match="Rate limit exceeded"):
        client.generate("hello", config)


def test_factory_dispatch() -> None:
    mock_c = get_inference_client("mock")
    assert isinstance(mock_c, MockInferenceClient)

    ollama_c = get_inference_client("ollama")
    assert isinstance(ollama_c, OllamaClient)

    openai_c = get_inference_client("openai", api_key="test")
    assert isinstance(openai_c, OpenAICompatibleClient)

    gemini_c = get_inference_client("gemini", api_key="test")
    assert isinstance(gemini_c, OpenAICompatibleClient)
    assert "generativelanguage.googleapis.com" in gemini_c.base_url

    with pytest.raises(ValueError, match="Unknown inference provider"):
        get_inference_client("unsupported_provider")


def test_e2e_pipeline_task_inference_sandbox() -> None:
    """Integration test: ImputationTask -> MockInference -> Code Extract -> Sandbox."""
    clean_df = pd.DataFrame(
        {
            "col_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "col_b": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    masked_df, mask = inject_mcar(clean_df, rate=0.40, seed=42)

    task = ImputationTask(
        task_id="test_e2e_01",
        dataset_name="demo",
        clean_df=clean_df,
        masked_df=masked_df,
        mask=mask,
    )

    # 1. Generate prompt from task specification
    prompt = task.generate_prompt()

    # 2. Invoke inference engine
    client = MockInferenceClient()
    config = InferenceConfig(model="mock-qwen-7b", temperature=0.0)
    response = client.generate(prompt, config)

    # 3. Extract Python code
    extracted_code = extract_python_code(response.content)
    assert "def impute_missing" in extracted_code

    # 4. Execute generated code in isolated sandbox
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.csv"
        output_path = Path(tmp_dir) / "output.csv"
        masked_df.to_csv(input_path, index=False)

        runner_script = (
            f"import pandas as pd\n"
            f"{extracted_code}\n\n"
            f"df = pd.read_csv(r'{input_path}')\n"
            f"res = impute_missing(df)\n"
            f"res.to_csv(r'{output_path}', index=False)\n"
        )

        exec_res = execute_code(runner_script, timeout_sec=10.0)
        assert exec_res.success is True, f"Sandbox failed: {exec_res.stderr}"
        assert output_path.exists()

        imputed_df = pd.read_csv(output_path)

    # 5. Evaluate task output against ground truth
    evaluation = task.evaluate_output(imputed_df)
    assert evaluation.syntax_success is True
    assert evaluation.data_valid is True
    assert evaluation.quality_score is not None
    assert evaluation.quality_score >= 0.0
