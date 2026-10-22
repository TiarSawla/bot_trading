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
        return SimpleNamespace(bid=4000.0, ask=4000.1)

    def positions_get(self, symbol=None):
        return [self.position]

    def order_send(self, request):
        self.requests.append(request.copy())
        if len(self.requests) == 1:
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
    try:
        assert bot.close_position(123, 0, 'XAUUSD', 0.01)
        assert len(fake.requests) == 2
        assert fake.requests[0]['type_filling'] == fake.ORDER_FILLING_IOC
        assert fake.requests[1]['type_filling'] == fake.ORDER_FILLING_RETURN
        assert fake.requests[0]['magic'] == 202606
        bot.micro_trailing_stop('XAUUSD')
        assert fake.requests[2]['action'] == fake.TRADE_ACTION_SLTP
        assert fake.requests[2]['sl'] == 3998.2
        fake.position.price_open = 3999.1
        bot.manage_basket_tp_sl('XAUUSD', 202606, [fake.position], [])
        assert fake.requests[3]['action'] == fake.TRADE_ACTION_SLTP
        assert fake.requests[3]['sl'] == 3999.6
        fake.position.price_open = 3998.0
        bot.manage_basket_tp_sl('XAUUSD', 202606, [fake.position], [])
        assert fake.requests[4]['action'] == fake.TRADE_ACTION_DEAL
        assert fake.requests[4]['position'] == 456
    finally:
        bot.mt5 = original_mt5

    print('LIVE TRADER CHECK OK')


if __name__ == '__main__':
    main()
