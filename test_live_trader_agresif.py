import logging
import sys
from types import SimpleNamespace

logging.FileHandler = lambda *args, **kwargs: logging.NullHandler()
sys.path.insert(0, 'src')
import live_trader_agresif as bot


class FakeMT5:
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TYPE_SELL = 0
    ORDER_TYPE_BUY = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    TRADE_ACTION_SLTP = 11
    ORDER_TIME_GTC = 20
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_INVALID_FILL = 10030

    def __init__(self):
        self.requests = []
        self.position = SimpleNamespace(
            magic=202606,
            ticket=456,
            type=self.POSITION_TYPE_BUY,
            price_open=3998.0,
            sl=0.0,
            tp=0.0,
            time=1,
            volume=0.01,
            profit=100000.0,
            swap=0.0,
        )
        self.positions = [self.position]
        self.bid = 4000.0
        self.ask = 4000.1

    def symbol_info(self, symbol):
        return SimpleNamespace(
            filling_mode=2,
            trade_stops_level=0,
            trade_freeze_level=0,
            point=0.01,
            digits=2,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self.bid, ask=self.ask)

    def positions_get(self, symbol=None):
        return self.positions

    def order_send(self, request):
        self.requests.append(request.copy())
        if len(self.requests) == 1 and request['action'] == self.TRADE_ACTION_DEAL:
            return SimpleNamespace(retcode=self.TRADE_RETCODE_INVALID_FILL, comment='invalid fill')
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment='done', price=4000.0)


def main():
    assert bot.model_action_to_bias(0, 3) == 'HOLD'
    assert bot.model_action_to_bias(2, 3) == 'SELL'
    assert bot.model_action_to_bias(0, 2) == 'SELL'

    fake = FakeMT5()
    original_mt5 = bot.mt5
    bot.mt5 = fake
    bot.MAGIC_NUMBER = 202606
    bot.DEVIATION = 20
    bot.DRY_RUN = False
    original_basket_tp = bot.BASKET_TP_USD
    original_basket_sl = bot.BASKET_SL_PIPS
    try:
        assert bot.close_position(123, 0, 'XAUUSD', 0.01)
        assert len(fake.requests) == 2
        assert fake.requests[0]['type_filling'] == fake.ORDER_FILLING_IOC
        assert fake.requests[1]['type_filling'] == fake.ORDER_FILLING_RETURN
        assert fake.requests[0]['magic'] == 202606
        bot.micro_trailing_stop('XAUUSD')
        assert fake.requests[2]['action'] == fake.TRADE_ACTION_SLTP
        assert fake.requests[2]['sl'] == 3998.3
        fake.position.price_open = 3999.1
        bot.manage_basket_tp_sl('XAUUSD', 202606, [fake.position], [])
        assert fake.requests[3]['action'] == fake.TRADE_ACTION_SLTP
        assert fake.requests[3]['sl'] == 3999.55
        fake.position.price_open = 3998.0
        bot.manage_basket_tp_sl('XAUUSD', 202606, [fake.position], [])
        assert fake.requests[4]['action'] == fake.TRADE_ACTION_DEAL
        assert fake.requests[4]['position'] == 456

        bot.BASKET_TP_USD = 2.0
        bot.BASKET_SL_PIPS = 0.0
        hybrid = FakeMT5()
        hybrid.position.price_open = 3998.0
        hybrid.position.profit = 1.0
        second_buy = SimpleNamespace(
            magic=202606,
            ticket=457,
            type=hybrid.POSITION_TYPE_BUY,
            price_open=3998.2,
            sl=0.0,
            tp=0.0,
            time=1,
            volume=0.01,
            profit=1.0,
            swap=0.0,
        )
        hybrid.positions = [hybrid.position, second_buy]
        avg_buy = 3998.1
        hybrid.bid = avg_buy + 0.7
        hybrid.ask = hybrid.bid + 0.1
        bot.mt5 = hybrid
        bot._sltp_last_update.clear()
        bot.manage_basket_tp_sl('XAUUSD', 202606, hybrid.positions, [])
        assert not any(req['action'] == hybrid.TRADE_ACTION_DEAL for req in hybrid.requests)
        # P1 (on_profit_side guard) + P3: trail BUY = avg+0.4 = 3998.5, BUY clamp min(sl, bid-0.4) = 3998.4
        assert any(req['action'] == hybrid.TRADE_ACTION_SLTP and req['sl'] == 3998.4 for req in hybrid.requests)

        hybrid.position.sl = 3998.8
        second_buy.sl = 3998.7
        before_safe_update = len(hybrid.requests)
        bot._sltp_last_update.clear()
        bot.manage_basket_tp_sl('XAUUSD', 202606, hybrid.positions, [])
        for req in hybrid.requests[before_safe_update:]:
            if req['action'] == hybrid.TRADE_ACTION_SLTP:
                expected_sl = 3998.8 if req['position'] == 456 else 3998.7
                assert req['sl'] == expected_sl

        tp_distance = bot.TP_FROM_FIRST_ENTRY_PIPS * bot.XAU_PIP_VALUE
        hybrid.bid = avg_buy + tp_distance + 0.1
        hybrid.ask = hybrid.bid + 0.1
        bot.manage_basket_tp_sl('XAUUSD', 202606, hybrid.positions, [])
        assert any(req['action'] == hybrid.TRADE_ACTION_DEAL for req in hybrid.requests)

        sell = FakeMT5()
        sell.position.type = sell.POSITION_TYPE_SELL
        sell.position.price_open = 4001.0
        sell.position.profit = 1.0
        second_sell = SimpleNamespace(
            magic=202606,
            ticket=458,
            type=sell.POSITION_TYPE_SELL,
            price_open=4000.8,
            sl=0.0,
            tp=0.0,
            time=1,
            volume=0.01,
            profit=1.0,
            swap=0.0,
        )
        sell.positions = [sell.position, second_sell]
        avg_sell = 4000.9
        sell.ask = avg_sell - 0.7
        sell.bid = sell.ask - 0.1
        bot.mt5 = sell
        bot._sltp_last_update.clear()
        bot.manage_basket_tp_sl('XAUUSD', 202606, [], sell.positions)
        assert not any(req['action'] == sell.TRADE_ACTION_DEAL for req in sell.requests)
        # P1/P2/P3: trail SELL = avg-0.4 = 4000.5, SELL clamp max(sl, ask+0.4) = 4000.6
        assert any(req['action'] == sell.TRADE_ACTION_SLTP and req['sl'] == 4000.6 for req in sell.requests)

        sell.position.sl = 4000.2
        second_sell.sl = 4000.3
        before_sell_safe_update = len(sell.requests)
        bot.manage_basket_tp_sl('XAUUSD', 202606, [], sell.positions)
        for req in sell.requests[before_sell_safe_update:]:
            if req['action'] == sell.TRADE_ACTION_SLTP:
                expected_sl = 4000.2 if req['position'] == 456 else 4000.3
                assert req['sl'] == expected_sl

        sell.ask = avg_sell - tp_distance - 0.1
        sell.bid = sell.ask - 0.1
        bot._sltp_last_update.clear()
        bot.manage_basket_tp_sl('XAUUSD', 202606, [], sell.positions)
        assert any(req['action'] == sell.TRADE_ACTION_DEAL for req in sell.requests)
    finally:
        bot.mt5 = original_mt5
        bot.BASKET_TP_USD = original_basket_tp
        bot.BASKET_SL_PIPS = original_basket_sl

    # --- REGRESSION (P1/P2): SELL 2-layer, kandidat lock jatuh di sisi rugi -> TIDAK ADA spam SLTP ---
    #     Ask masih di atas avg + min_distance (sisi rugi). Lock/trail harus mati, hanya TP (jika valid).
    reg = FakeMT5()
    reg.position.type = reg.POSITION_TYPE_SELL
    reg.position.price_open = 4001.0
    reg.position.profit = 1.0
    reg.position.sl = 0.0
    reg.position.tp = 0.0
    reg_second = SimpleNamespace(
        magic=202606, ticket=459, type=reg.POSITION_TYPE_SELL,
        price_open=4000.8, sl=0.0, tp=0.0, time=1, volume=0.01, profit=1.0, swap=0.0,
    )
    reg.positions = [reg.position, reg_second]
    reg_avg = 4000.9
    # ask masih 4000.1 (default) -> avg - ask = 0.8 > 0.4 (sisi profit? TIDAK, SELL profit = ask < avg)
    # avg - ask = 4000.9 - 4000.1 = 0.8 > min_distance -> actually SELL IS on profit side here.
    # To force loss side: ask > avg - min_distance, i.e. ask >= avg - 0.4 = 4000.5
    reg.ask = 4000.7
    reg.bid = reg.ask - 0.1
    bot.mt5 = reg
    bot.BASKET_TP_USD = 2.0
    bot._sltp_last_update.clear()
    bot.manage_basket_tp_sl('XAUUSD', 202606, [], reg.positions)
    # Tidak ada SLTP dengan SL valid (semua SL=0 -> di-cancel sisi rugi). Tidak ada DEAL.
    sltp_with_sl = [r for r in reg.requests if r.get('action') == reg.TRADE_ACTION_SLTP and r.get('sl', 0) > 0]
    assert not sltp_with_sl, f"REG: lock SELL di sisi rugi harus di-skip, got {sltp_with_sl}"
    assert not any(r.get('action') == reg.TRADE_ACTION_DEAL for r in reg.requests)

    # --- REGRESSION (P3): SL tidak boleh melewati TP -> ditolak, tidak dipaksakan ---
    #     SELL: SL trail di bawah TP -> harus di-cancel.
    reg.position.sl = 4000.6
    reg_second.sl = 4000.6
    bot._sltp_last_update.clear()
    # ask jauh di bawah avg -> profit, trail aktif. trail = avg - 0.4 = 4000.5
    reg.ask = reg_avg - 2.0
    reg.bid = reg.ask - 0.1
    bot.manage_basket_tp_sl('XAUUSD', 202606, [], reg.positions)
    # SL trail (4000.5) < TP (4000.9 - 1.2 = 3999.7)? Tidak, 4000.5 > 3999.7.
    # Untuk memaksa SL > TP di SELL, TP harus di bawah SL trail.
    # TP SELL = avg - 1.2 = 3999.7. SL trail = 4000.5 > TP -> SL di atas TP -> di-cancel.
    reg_sl_sent = [r for r in reg.requests if r.get('action') == reg.TRADE_ACTION_SLTP and r.get('sl', 0) > 0]
    # Jika ada SL, tidak boleh > TP (3999.7) -> jadi harus = 0 (cancel)
    for r in reg_sl_sent:
        assert r['sl'] < (reg_avg - 1.2) or r['sl'] == 0, f"REG: SL SELL ({r['sl']}) boleh melebihi TP"
    # Tegaskan: tidak ada SL yang melewati TP ke atas
    assert not any(r.get('sl', 0) > (reg_avg - 1.2) for r in reg_sl_sent)

    # Reset ke state normal
    bot.BASKET_TP_USD = original_basket_tp
    bot.mt5 = original_mt5

    print('LIVE TRADER CHECK OK')


if __name__ == '__main__':
    main()
