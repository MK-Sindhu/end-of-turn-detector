from indic_eot.config import load_params


def test_params_load_and_have_core_settings():
    params = load_params()
    assert params["audio"]["sample_rate"] == 16000
    assert params["audio"]["max_seconds"] > 0
    assert len(params["data"]["languages"]) >= 1
