from indic_eot.text_model import normalize_text


def test_normalize_strips_punctuation_including_the_hindi_full_stop():
    assert normalize_text("मुझे दो टिकट चाहिए।") == "मुझे दो टिकट चाहिए"
    assert normalize_text("Hello, world!  OK?") == "hello world ok"


def test_normalize_keeps_vowel_signs_and_digits():
    # Devanagari vowel signs are "marks", not punctuation, and must survive.
    assert normalize_text("ऑर्डर नंबर 4799 है") == "ऑर्डर नंबर 4799 है"
