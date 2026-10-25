from typing import Tuple, Dict, Any, Optional
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class GoldAggressiveGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        window_size: int = 10,
        layer_distance: float = 2.0,
        max_layers: int = 5,
        lot_step: float = 0.01,
        tp_from_first_entry: float = 1.0,
        basket_sl_from_last_entry: float = 4.0,
        trail_trigger_profit: float = 0.5,
        trail_step_profit: float = 3.0,
        contract_value_per_lot: float = 100.0,
        margin_per_001_lot: float = 20.0,
        basket_tp_usd: float = 2.0,
        profit_lock_offset: float = 0.15,
    ):
        super().__init__()
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.layer_distance = layer_distance
        self.max_layers = max_layers
        self.lot_step = lot_step
        self.tp_from_first_entry = tp_from_first_entry
        self.basket_sl_from_last_entry = basket_sl_from_last_entry
        self.trail_trigger_profit = trail_trigger_profit
        self.trail_step_profit = trail_step_profit
        self.contract_value_per_lot = contract_value_per_lot
        self.margin_per_001_lot = margin_per_001_lot
        self.basket_tp_usd = basket_tp_usd
        self.profit_lock_offset = profit_lock_offset

        self.features_df = self.df.select_dtypes(include=[np.number]).copy()
        if "time" in self.features_df.columns:
            self.features_df.drop(columns=["time"], inplace=True)
        self.features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.features_df.ffill(inplace=True)
        self.features_df.bfill(inplace=True)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.features_df.shape[1] + 8,),
            dtype=np.float32,
        )
        self.window_data = deque(maxlen=window_size)
        self.reset_state()

    def reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.side = 0
        self.realized_pnl = 0.0
        self.max_drawdown = 0.0
        self.trail_sl = None
        self.static_sl = None
        self.tp_price = None
        self.profit_lock_active = False
        self.last_exit_reason = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.reset_state()
        self.window_data.clear()
        for i in range(self.current_step - self.window_size, self.current_step):
            self.window_data.append(self.features_df.iloc[i].to_numpy(dtype=np.float32))
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_equity = self.equity
        price = float(self.df.iloc[self.current_step]["close"])
        high = float(self.df.iloc[self.current_step]["high"])
        low = float(self.df.iloc[self.current_step]["low"])

        # Live mode ignores opposite signals while a basket is active.
        if not self.positions:
            if action == 1:
                self._open_layer(1, price)
            elif action == 2:
                self._open_layer(-1, price)

        self._manage_layers(price)
        self._manage_exit(high, low, price)
        floating = self._floating_pnl(price)
        self.equity = self.balance + floating
        self.max_drawdown = min(self.max_drawdown, self.equity - self.initial_balance)

        reward = (self.equity - prev_equity) * 0.01
        reward -= 0.02 * len(self.positions)
        if floating < -10:
            reward -= abs(floating) * 0.002
        if len(self.positions) >= self.max_layers and floating < 0:
            reward -= abs(floating) * 0.004
        if self.equity <= self.initial_balance * 0.7:
            reward -= 10.0

        current_features = self.features_df.iloc[self.current_step].to_numpy(dtype=np.float32)
        self.window_data.append(current_features)
        self.current_step += 1

        terminated = self.current_step >= len(self.features_df) - 1 or self.equity <= self.initial_balance * 0.5
        truncated = False
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _open_layer(self, side: int, price: float):
        if len(self.positions) >= self.max_layers:
            return
        if not self.positions:
            self.last_exit_reason = None
        layer = len(self.positions) + 1
        multiplier = 1 if layer <= 2 else (2 if layer <= 4 else 3)
        lot = round(multiplier * self.lot_step, 2)
        self.positions.append({"side": side, "price": price, "lot": lot})
        self.side = side
        self._refresh_targets()

    def _close_all(self, price: float, reason: str = "UNSPECIFIED"):
        if not self.positions:
            self.side = 0
            return
        pnl = self._floating_pnl(price)
        self.balance += pnl
        self.realized_pnl += pnl
        self.last_exit_reason = reason
        self.positions.clear()
        self.side = 0
        self.trail_sl = None
        self.static_sl = None
        self.tp_price = None
        self.profit_lock_active = False

    def _manage_layers(self, price: float):
        if not self.positions:
            return
        side = self.side
        last = self.positions[-1]["price"]
        if side == 1 and price <= last - self.layer_distance:
            self._open_layer(side, price)
        elif side == -1 and price >= last + self.layer_distance:
            self._open_layer(side, price)

    def _active_sl(self):
        return self.trail_sl if self.trail_sl is not None else self.static_sl

    def _is_safer_sl(self, candidate: float, side: int, current: Optional[float]) -> bool:
        if current is None:
            return True
        return candidate > current if side == 1 else candidate < current

    def _raise_sl(self, candidate: Optional[float], side: int, current_price: Optional[float] = None):
        if candidate is None or candidate <= 0:
            return
        if current_price is not None:
            valid_for_market = candidate < current_price if side == 1 else candidate > current_price
            if not valid_for_market:
                return
        current = self._active_sl()
        if not self._is_safer_sl(candidate, side, current):
            return
        if self.trail_sl is None and self.static_sl is None:
            self.static_sl = candidate
        else:
            self.trail_sl = candidate

    def _manage_exit(self, high: float, low: float, close_price: float):
        if not self.positions:
            return
        side = self.side
        floating = self._floating_pnl(close_price)
        avg = self._avg_price()

        if len(self.positions) >= 2 and floating >= self.basket_tp_usd:
            self.profit_lock_active = True

        if len(self.positions) >= self.max_layers and self.basket_sl_from_last_entry > 0:
            last_price = self.positions[-1]["price"]
            static_sl = last_price - self.basket_sl_from_last_entry if side == 1 else last_price + self.basket_sl_from_last_entry
            self._raise_sl(static_sl, side)

        if self.tp_price is not None:
            if side == 1 and high >= self.tp_price:
                self._close_all(self.tp_price, reason="BASKET_TP")
                return
            if side == -1 and low <= self.tp_price:
                self._close_all(self.tp_price, reason="BASKET_TP")
                return

        if self.profit_lock_active:
            lock_sl = avg + self.profit_lock_offset if side == 1 else avg - self.profit_lock_offset
            self._raise_sl(lock_sl, side, close_price)

        if floating >= self.trail_trigger_profit and self.trail_step_profit > 0:
            step = int(floating / self.trail_step_profit)
            if step > 0:
                trail_offset = step * self.trail_step_profit * 0.5
                trail_sl = avg + trail_offset if side == 1 else avg - trail_offset
                self._raise_sl(trail_sl, side, close_price)

        sl = self._active_sl()
        if sl is not None:
            if side == 1 and low <= sl:
                self._close_all(sl, reason="PROFIT_LOCK_SL" if self.profit_lock_active else "BASKET_SL")
            elif side == -1 and high >= sl:
                self._close_all(sl, reason="PROFIT_LOCK_SL" if self.profit_lock_active else "BASKET_SL")

    def _refresh_targets(self):
        if not self.positions:
            return
        avg = self._avg_price()
        if self.side == 1:
            self.tp_price = avg + self.tp_from_first_entry
        elif self.side == -1:
            self.tp_price = avg - self.tp_from_first_entry

    def _floating_pnl(self, price: float) -> float:
        total = 0.0
        for pos in self.positions:
            direction = 1 if pos["side"] == 1 else -1
            total += direction * (price - pos["price"]) * pos["lot"] * self.contract_value_per_lot
        return total

    def _avg_price(self) -> float:
        if not self.positions:
            return 0.0
        vol = sum(p["lot"] for p in self.positions)
        return sum(p["price"] * p["lot"] for p in self.positions) / vol if vol else 0.0

    def _margin_level(self, price: float) -> float:
        used_margin = sum((p["lot"] / 0.01) * self.margin_per_001_lot for p in self.positions)
        if used_margin <= 0:
            return 9999.0
        return (self.balance + self._floating_pnl(price)) / used_margin * 100.0

    def _get_observation(self) -> np.ndarray:
        market = np.array(self.window_data).flatten().astype(np.float32)
        price = float(self.df.iloc[min(self.current_step, len(self.df) - 1)]["close"])
        floating = self._floating_pnl(price)
        state = np.array([
            self.side,
            len(self.positions),
            self._avg_price(),
            floating,
            self.equity,
            self._margin_level(price),
            self.tp_price or 0.0,
            self.trail_sl or self.static_sl or 0.0,
        ], dtype=np.float32)
        return np.concatenate([market, state])

    def _get_info(self) -> Dict[str, Any]:
        price = float(self.df.iloc[min(self.current_step, len(self.df) - 1)]["close"])
        return {
            "step": self.current_step,
            "side": self.side,
            "layers": len(self.positions),
            "balance": self.balance,
            "equity": self.equity,
            "floating": self._floating_pnl(price),
            "margin_level": self._margin_level(price),
            "realized_pnl": self.realized_pnl,
            "profit_lock_active": self.profit_lock_active,
            "tp_price": self.tp_price,
            "stop_price": self._active_sl(),
            "last_exit_reason": self.last_exit_reason,
        }
