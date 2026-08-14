from src.trading.trading_cycle import run_trading_cycle


def main() -> None:
    result = run_trading_cycle(
        prediction=0.002,
        price=500.0,
        regime="bullish",
        active_model_id="expert_lightgbm_bullish",
    )

    print(result)


if __name__ == "__main__":
    main()
