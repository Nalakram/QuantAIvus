def test_full_pipeline(tmp_path, monkeypatch):
    # Create a sample CSV for loader
    csv = tmp_path / "input.csv"
    csv.write_text("A,B\n1,10\n2,20\n")
    # Mock preprocessing step if necessary
    # Run the pipeline
    result_df = run_pipeline(input_path=str(csv))
    # Check final output has expected columns or transformations
    assert "preprocessed_feature" in result_df.columns

@pytest.mark.asyncio
async def test_stream_pipeline(monkeypatch):
    # Fake Twitter data stream as earlier
    fake_messages = [b'{"text": "hello"}', b'{"text": "world"}', None]
    class FakeWS:
        def __init__(self):
            self.msgs = fake_messages.copy()
        async def recv(self):
            return self.msgs.pop(0)
    async def fake_connect(*args, **kwargs):
        return FakeWS()
    monkeypatch.setattr("websockets.connect", fake_connect)
    # Run the streaming pipeline which loads, cleans, preprocesses each item
    results = []
    async for item in StreamingPipeline(loader=TwitterLoader(), cleaner=...):
        results.append(item)
    assert len(results) == 2
    # Check that missing fields or bad data are handled
    assert all("text" in msg for msg in results)

@pytest.mark.asyncio
async def test_stream_pipeline_api_timeout(monkeypatch):
    async def failing_stream(*args, **kwargs):
        raise TimeoutError("API timed out")
    monkeypatch.setattr(TwitterLoader, "stream_data", failing_stream)
    pipeline = StreamingPipeline(...)
    with pytest.raises(TimeoutError):
        async for _ in pipeline:
            pass

@pytest.mark.asyncio
async def test_stream_pipeline_corrupted_data(monkeypatch):
    class FakeWS:
        async def recv(self): return b"not a json"
    async def fake_connect(*args, **kwargs): return FakeWS()
    monkeypatch.setattr("websockets.connect", fake_connect)
    pipeline = StreamingPipeline(...)
    results = []
    with pytest.raises(ValueError):
        async for item in pipeline:
            results.append(item)
    # or if pipeline skips errors, assert results only has valid items
