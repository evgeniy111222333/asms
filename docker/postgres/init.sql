-- ACMS Database Schema Initialization
-- PostgreSQL 16+ compatible

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Users & Authentication ────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    username        VARCHAR(100) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'trader', 'viewer', 'user')),
    is_active       BOOLEAN DEFAULT TRUE,
    last_login      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- API Keys for exchange access
CREATE TABLE IF NOT EXISTS api_keys (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exchange        VARCHAR(50) NOT NULL,
    key_label       VARCHAR(100) NOT NULL,
    api_key         VARCHAR(255) NOT NULL,
    api_secret      VARCHAR(500) NOT NULL,  -- encrypted
    passphrase      VARCHAR(255),            -- for OKX
    is_testnet      BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    last_used       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_exchange ON api_keys(exchange);

-- Refresh tokens
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      VARCHAR(255) NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    is_revoked      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash);

-- ─── Strategies ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS strategies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    type            VARCHAR(50) NOT NULL CHECK (type IN ('momentum', 'mean_reversion', 'breakout', 'grid', 'dca', 'custom')),
    config          JSONB NOT NULL DEFAULT '{}',
    symbols         JSONB NOT NULL DEFAULT '[]',
    exchanges       JSONB NOT NULL DEFAULT '[]',
    is_active       BOOLEAN DEFAULT FALSE,
    started_at      TIMESTAMPTZ,
    stopped_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_strategies_user ON strategies(user_id);
CREATE INDEX idx_strategies_active ON strategies(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_strategies_type ON strategies(type);

-- ─── Orders & Trades ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS orders (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id     UUID REFERENCES strategies(id) ON DELETE SET NULL,
    exchange        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(50) NOT NULL,
    side            VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    type            VARCHAR(20) NOT NULL CHECK (type IN ('market', 'limit', 'stop', 'stop_limit', 'trailing_stop')),
    quantity        DECIMAL(20, 8) NOT NULL,
    price           DECIMAL(20, 8),
    stop_price      DECIMAL(20, 8),
    status          VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected', 'error')),
    filled_qty      DECIMAL(20, 8) DEFAULT 0,
    avg_fill_price  DECIMAL(20, 8) DEFAULT 0,
    exchange_order_id VARCHAR(255),
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_strategy ON orders(strategy_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_exchange ON orders(exchange);
CREATE INDEX idx_orders_created ON orders(created_at);

-- Individual trade fills
CREATE TABLE IF NOT EXISTS trades (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id        UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exchange        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(50) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        DECIMAL(20, 8) NOT NULL,
    price           DECIMAL(20, 8) NOT NULL,
    fee             DECIMAL(20, 8) DEFAULT 0,
    fee_currency    VARCHAR(20),
    exchange_trade_id VARCHAR(255),
    executed_at     TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_order ON trades(order_id);
CREATE INDEX idx_trades_user ON trades(user_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed ON trades(executed_at);

-- ─── Positions & Portfolio ─────────────────────────────────

CREATE TABLE IF NOT EXISTS positions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id     UUID REFERENCES strategies(id) ON DELETE SET NULL,
    exchange        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(50) NOT NULL,
    side            VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    quantity        DECIMAL(20, 8) NOT NULL,
    entry_price     DECIMAL(20, 8) NOT NULL,
    current_price   DECIMAL(20, 8),
    unrealized_pnl  DECIMAL(20, 8) DEFAULT 0,
    realized_pnl    DECIMAL(20, 8) DEFAULT 0,
    leverage        DECIMAL(10, 2) DEFAULT 1,
    liquidation_price DECIMAL(20, 8),
    is_open         BOOLEAN DEFAULT TRUE,
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_positions_user ON positions(user_id);
CREATE INDEX idx_positions_open ON positions(is_open) WHERE is_open = TRUE;
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Portfolio snapshots for historical tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_value     DECIMAL(20, 8) NOT NULL,
    unrealized_pnl  DECIMAL(20, 8) DEFAULT 0,
    realized_pnl    DECIMAL(20, 8) DEFAULT 0,
    drawdown        DECIMAL(10, 6) DEFAULT 0,
    positions_count INT DEFAULT 0,
    snapshot_time   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_portfolio_snapshots_user ON portfolio_snapshots(user_id);
CREATE INDEX idx_portfolio_snapshots_time ON portfolio_snapshots(snapshot_time);

-- ─── Risk Management ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS risk_limits (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    max_position_size DECIMAL(20, 8) DEFAULT 100000,
    max_daily_loss  DECIMAL(20, 8) DEFAULT 5000,
    max_drawdown    DECIMAL(5, 2) DEFAULT 20.00,
    max_leverage    DECIMAL(10, 2) DEFAULT 3.0,
    max_open_positions INT DEFAULT 10,
    max_orders_per_minute INT DEFAULT 60,
    kill_switch_active BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Backtesting ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtests (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id     UUID REFERENCES strategies(id) ON DELETE SET NULL,
    name            VARCHAR(255),
    config          JSONB NOT NULL DEFAULT '{}',
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    final_capital   DECIMAL(20, 8),
    total_return    DECIMAL(10, 6),
    sharpe_ratio    DECIMAL(10, 4),
    max_drawdown    DECIMAL(10, 6),
    win_rate        DECIMAL(5, 2),
    total_trades    INT,
    status          VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    results         JSONB,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_backtests_user ON backtests(user_id);
CREATE INDEX idx_backtests_status ON backtests(status);

-- ─── Market Data Cache ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS market_data (
    id              BIGSERIAL PRIMARY KEY,
    exchange        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(50) NOT NULL,
    interval        VARCHAR(10) NOT NULL DEFAULT '1m',
    open_time       TIMESTAMPTZ NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(20, 8) NOT NULL,
    close_time      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(exchange, symbol, interval, open_time)
);

CREATE INDEX idx_market_data_lookup ON market_data(exchange, symbol, interval, open_time DESC);

-- ─── Signals ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS signals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol          VARCHAR(50) NOT NULL,
    exchange        VARCHAR(50) NOT NULL,
    signal_type     VARCHAR(20) NOT NULL CHECK (signal_type IN ('buy', 'sell', 'close_long', 'close_short')),
    strength        DECIMAL(5, 2) DEFAULT 0.5,
    price           DECIMAL(20, 8),
    metadata        JSONB DEFAULT '{}',
    is_executed     BOOLEAN DEFAULT FALSE,
    executed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_strategy ON signals(strategy_id);
CREATE INDEX idx_signals_user ON signals(user_id);
CREATE INDEX idx_signals_created ON signals(created_at);

-- ─── Audit Log ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    action          VARCHAR(100) NOT NULL,
    resource_type   VARCHAR(50) NOT NULL,
    resource_id     VARCHAR(255),
    details         JSONB DEFAULT '{}',
    ip_address      INET,
    user_agent      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);

-- ─── System Configuration ──────────────────────────────────

CREATE TABLE IF NOT EXISTS system_config (
    key             VARCHAR(255) PRIMARY KEY,
    value           JSONB NOT NULL,
    description     TEXT,
    updated_by      UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default system configuration
INSERT INTO system_config (key, value, description) VALUES
    ('kill_switch', 'false', 'Global kill switch - stops all trading'),
    ('max_global_leverage', '5.0', 'Maximum leverage allowed across all users'),
    ('maintenance_mode', 'false', 'Maintenance mode - read-only access'),
    ('supported_exchanges', '["binance", "bybit", "okx"]', 'Supported exchanges'),
    ('default_symbols', '["BTCUSDT", "ETHUSDT", "SOLUSDT"]', 'Default trading pairs')
ON CONFLICT (key) DO NOTHING;

-- ─── Helper Functions ──────────────────────────────────────

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_risk_limits_updated_at BEFORE UPDATE ON risk_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─── Default Admin User ────────────────────────────────────
-- Password: admin (bcrypt hash)
INSERT INTO users (email, username, password_hash, role) VALUES
    ('admin@acms.local', 'admin', '$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Default risk limits for admin
INSERT INTO risk_limits (user_id, max_position_size, max_daily_loss, max_drawdown, max_leverage)
    SELECT id, 100000, 5000, 20.00, 3.0 FROM users WHERE username = 'admin'
ON CONFLICT (user_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO acms;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO acms;
