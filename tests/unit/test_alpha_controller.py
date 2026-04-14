from mcp_client.alpha_controller import AlphaController, AlphaControllerConfig


def test_alpha_increases_when_confidence_high() -> None:
    ctrl = AlphaController(
        AlphaControllerConfig(
            initial=0.12,
            min_alpha=0.08,
            max_alpha=0.30,
            increase_step=0.02,
            decay_rate=0.02,
            high_threshold=0.70,
            low_threshold=0.40,
        )
    )
    for _ in range(10):
        ctrl.update(0.9)

    assert ctrl.alpha > 0.12


def test_alpha_decays_when_confidence_low() -> None:
    ctrl = AlphaController(
        AlphaControllerConfig(
            initial=0.2,
            min_alpha=0.08,
            max_alpha=0.30,
            increase_step=0.02,
            decay_rate=0.02,
            high_threshold=0.70,
            low_threshold=0.40,
        )
    )
    for _ in range(10):
        ctrl.update(0.2)

    assert ctrl.alpha < 0.2
