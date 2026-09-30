from vexa_video.models.text_encoder import ByteTokenizer


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "Video motion: café → right"
    ids = tokenizer.encode(text, max_length=128)
    assert tokenizer.decode(ids) == text
    assert len(ids) == 128
