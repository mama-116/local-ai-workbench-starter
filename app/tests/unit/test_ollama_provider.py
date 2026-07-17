from local_llm_chat.infrastructure.llm.ollama_provider import OllamaProvider


def test_done_chunk_keeps_ollama_generation_duration_for_speed_calculation() -> None:
    chunk = OllamaProvider._parse_stream_line(
        '{"done":true,"message":{"content":""},"eval_count":20,'
        '"eval_duration":1000000000,"total_duration":2000000000}'
    )

    assert chunk.done
    assert chunk.output_tokens == 20
    assert chunk.generation_duration_ns == 1_000_000_000


def test_parses_tool_call_and_marks_broken_arguments_for_denial() -> None:
    chunk = OllamaProvider._parse_stream_line(
        '{"done":true,"message":{"content":"","tool_calls":['
        '{"function":{"name":"read_allowed_text","arguments":"broken"}}]}}'
    )

    assert chunk.tool_calls[0].name == "read_allowed_text"
    assert chunk.tool_calls[0].arguments == {"__invalid_arguments__": True}
