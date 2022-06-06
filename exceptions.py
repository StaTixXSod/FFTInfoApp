class NotFittedData(Exception):
    """The Object wasn't fitted. Use 'fit' method to predict the data."""


class TestDataAlreadyExists(Exception):
    """Test data already exists. Don't use 'backtest' feature if you have your own data."""


class MissingTestData(Exception):
    """There is no testing data. Nothing to test."""
