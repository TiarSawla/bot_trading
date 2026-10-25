import pandas as pd

from src.rl_environment_agresif import GoldAggressiveGridEnv


def make_env():
    rows = []
    for i in range(40):
        price = 4000.0 + i * 0.1
        rows.append({
            "time": f"2026-01-01 00:{i:02d}:00",
            "open": price,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
            "feature": float(i),
        })
    return GoldAggressiveGridEnv(
        pd.DataFrame(rows),
        window_size=2,
        layer_distance=1.0,
        max_layers=7,
        lot_step=0.01,
        tp_from_first_entry=1.2,
        basket_sl_from_last_entry=0.0,
        basket_tp_usd=2.0,
        profit_lock_offset=0.15,
        trail_trigger_profit=3.0,
    )


def assert_close(actual, expected):
    assert abs(actual - expected) < 1e-6, (actual, expected)


def main():
    env = make_env()
    env.reset(seed=1)
    env._open_layer(1, 4000.0)
    env._open_layer(1, 3999.0)

    assert [pos["lot"] for pos in env.positions] == [0.01, 0.01]
    assert_close(env.tp_price, 4000.7)

    env._manage_exit(4000.0, 3999.5, 4000.0)
    assert not env.profit_lock_active

    env._manage_exit(4000.6, 4000.4, 4000.5)
    assert env.profit_lock_active
    assert_close(env._active_sl(), 3999.65)
    assert_close(env.tp_price, 4000.7)

    locked_sl = env._active_sl()
    env._manage_exit(4000.0, 3999.7, 3999.8)
    assert_close(env._active_sl(), locked_sl)
    env._manage_exit(4000.0, 3999.4, 3999.4)
    assert env.last_exit_reason == "PROFIT_LOCK_SL"

    sell = make_env()
    sell.reset(seed=1)
    sell._open_layer(-1, 4000.0)
    sell._open_layer(-1, 4001.0)
    sell._manage_exit(3999.5, 3999.4, 3999.4)
    assert sell.profit_lock_active
    assert_close(sell._active_sl(), 4000.35)
    assert_close(sell.tp_price, 3999.3)

    tp = make_env()
    tp.reset(seed=1)
    tp._open_layer(1, 4000.0)
    tp._open_layer(1, 3999.0)
    tp._manage_exit(4000.8, 4000.6, 4000.8)
    assert not tp.positions
    assert tp.last_exit_reason == "BASKET_TP"

    hold = make_env()
    hold.reset(seed=1)
    hold._open_layer(1, 4000.0)
    hold.step(2)
    assert hold.side == 1
    assert len(hold.positions) == 1

    lots = make_env()
    lots.reset(seed=1)
    for layer in range(7):
        lots._open_layer(1, 4000.0 - layer)
    assert [pos["lot"] for pos in lots.positions] == [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03]

    assert env.observation_space.shape == (2 * 5 + 8,)
    observation = env._get_observation()
    assert str(observation.dtype) == "float32"
    print("RL ENVIRONMENT HYBRID CHECK OK")


if __name__ == "__main__":
    main()
