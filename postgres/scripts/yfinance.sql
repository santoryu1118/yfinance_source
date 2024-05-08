CREATE TABLE IF NOT EXISTS yfinance (
    ticker VARCHAR(20) NOT NULL,
    datetime timestamp NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    adjClose double precision NOT NULL,
    volume integer NOT NULL,
    PRIMARY KEY (ticker, datetime)
);

INSERT INTO yfinance (ticker, datetime, open, high, low, close, adjClose, volume) VALUES
        ('AAPL', '2022-07-01 09:30:00-04:00', 136.0399932861328, 138.32000732421875, 136.0, 136.08999633789062, 136.08999633789062, 17598079);
