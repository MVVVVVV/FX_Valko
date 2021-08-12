from fifo_doubles_list import FifoDoublesList
from quote import Quote
from trade_situation import TradeSituation
from limit_order_book import LimitOrderBook
import pickle

class MomentumStrategy:
    # This variable will be incremented after each call of TradeSituation.generate_next_id().
    # It is used to populate __trade_situation_id.
    __common_momentum_strategy_id: int = 0
    # This is a global reference to the order book
    __common_order_book: LimitOrderBook
    # Unique ID of the momentum strategy
    __strategy_id: int

    # Position is open in this way currently
    # False: sold
    # True: bought
    __current_trading_way: bool

    # Currently opened position
    __open_position: TradeSituation

    # List of trade situation
    __positions_history: list

    # This variable is set once to True when the required (minimal) data points are populated into fifo_list(s)
    __is_filled_start_data: bool
    __filled_data_points: int
    # This variable describes that we are using the best BID and best OFFER to calculate PnL
    __is_best_price_calculation: bool
    # This is the strategy's traded amount
    __traded_amount: float


    def __init__(self, curr_pair: str, target_profit_arg: float, traded_amount: float, is_best_px_calc: bool):
        """
        Initializes the trading strategy calculator. Please feed it with arguments for your moving average trading
        strategy. The MA_SLOW > MA_FAST. By construction the FAST average is low-period.
        :param ma_slow: slow moving moving average
        :param ma_fast: fast moving moving average
        :param target_profit_arg: target profit for this strategy
        """
        self.__curr_pair = curr_pair
        self.__strategy_id = MomentumStrategy.generate_next_id()
        self.__is_best_price_calculation = is_best_px_calc
        self.__traded_amount = traded_amount
        self.FX_list = ['EUR/JPY', 'AUD/USD', 'USD/CHF', 'NOK/SEK', 'USD/JPY', 'EUR/USD', 'USD/CAD', 'GBP/USD']

        self.__target_profit = target_profit_arg

        # Get the neural networks
        self.get_nn()

        # Init locals
        self.__current_trading_way = False
        self.__open_position = None
        self.__positions_history = []
        self.__is_filled_start_data = False
        self.__filled_data_points = 0

    def get_nn(self):
        self.nn = {}
        for curr_pair in self.FX_list:
            self.filename = 'finalized_model_' + curr_pair.replace('/', '_') + '.sav'
            self.nn[curr_pair] = pickle.load(open(self.filename, 'rb'))

    def step(self, quote: Quote):
        """
        Calculates the indicator and performs update/open/close action on the TradeSituation class
        (representing investment position)
        :param quote: float; the price of the invested stock
        :return: no return
        """
        # Update values (prices) in the fifo_lists (with put method)
        price_mid: float = (MomentumStrategy.__common_order_book.get_best_bid_price() +\
                            MomentumStrategy.__common_order_book.get_best_offer_price()) / 2.0

        # Update position with arrived quote
        if self.__open_position is not None:
            # We closed the position (returns true if the position is closed)
            if self.__open_position.update_on_order(quote):
                self.__open_position = None


        # The fifo_list(s) are filled?
        if self.__is_filled_start_data:
            # You must not reopen the position if the trading direction (__current_trading_way) has not changed.
            if self.nn[self.__curr_pair].predict()  and not self.__current_trading_way:
                # Buy: open position if there is none; close the position if it's hanging in the other way; append the
                # positions history (to save how much it gained); save the new __current_trading_way (repeat for SELL)
                if self.__open_position is not None:
                    self.__open_position.close_position(quote)
                self.__open_position = TradeSituation(quote, True, self.__target_profit, self.__traded_amount,
                                                      self.__is_best_price_calculation)
                self.__open_position.open_position(quote)
                self.__current_trading_way = True
                self.__positions_history.append(self.__open_position)
            elif not self.nn[self.__curr_pair].predict() and self.__current_trading_way:
                # Sell
                if self.__open_position is not None:
                    self.__open_position.close_position(quote)
                self.__open_position = TradeSituation(quote, False, self.__target_profit, self.__traded_amount,
                                                      self.__is_best_price_calculation)
                self.__current_trading_way = False
                self.__positions_history.append(self.__open_position)
        else:
            # The fifo_list(s) are not yet filled. Do the necessary updates and checks
            self.__filled_data_points += 1
            if self.__filled_data_points > self.__ma_slow_var:
                self.__is_filled_start_data = True

    def close_pending_position(self, quote: Quote):
        """
        Called at the end of the program execution. Checks if the position is still opened and closes that position.
        :param quote: last quote available in the data set
        :return:
        """
        # If there is still a position --> close it with the quote provided to you in arguments.
        if self.__open_position is not None and not self.__open_position.is_closed():
            self.__open_position.close_position(quote)

    def all_positions(self) -> list:
        """
        Returns the positions_history object
        :return:
        """
        # Returns __positions_history
        return self.__positions_history


    def get_target_profit(self) -> float:
        """
        Returns the target profit of this strategy
        :return:
        """
        return self.__target_profit

    def get_strategy_id(self):
        """
        Returns the (local) unique strategy ID.
        :return:
        """
        return self.__strategy_id

    @staticmethod
    def generate_next_id():
        MomentumStrategy.__common_momentum_strategy_id += 1
        return MomentumStrategy.__common_momentum_strategy_id

    @staticmethod
    def set_limit_order_book(limit_order_book: LimitOrderBook):
        MomentumStrategy.__common_order_book = limit_order_book
