"""CLI Module - Command-line interface for ACMS.

Implements a Click-based CLI with subcommands:
- run: start the trading system with configuration
- backtest: run backtests from command line
- optimize: run hyperparameter optimization
- migrate: run Alembic migrations
- seed: seed database with initial data
- status: show system status
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", default=None, help="Path to configuration file")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Logging level")
@click.option("--db-url", default=None, help="Database URL override")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], log_level: str, db_url: Optional[str]):
    """ACMS - Algorithmic Crypto Management System CLI.

    Manage and control the ACMS trading platform from the command line.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["log_level"] = log_level
    ctx.obj["db_url"] = db_url

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    if config:
        try:
            with open(config) as f:
                ctx.obj["config"] = json.load(f)
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            ctx.obj["config"] = {}
    else:
        ctx.obj["config"] = {}


@cli.command()
@click.option("--symbol", "-s", default="BTC/USDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="1m", help="Candle timeframe")
@click.option("--strategy", default="momentum_trend", help="Strategy type")
@click.option("--exchange", default="paper", help="Exchange adapter")
@click.option("--dry-run", is_flag=True, help="Run without placing real orders")
@click.pass_context
def run(ctx: click.Context, symbol: str, timeframe: str, strategy: str,
        exchange: str, dry_run: bool):
    """Start the ACMS trading system.

    Launches the orchestrator with the specified configuration,
    connecting to the exchange and running the trading loop.

    Examples:

        acms run --symbol BTC/USDT --strategy momentum_trend --exchange paper

        acms run --config production.json --dry-run
    """
    click.echo(f"Starting ACMS trading system...")
    click.echo(f"  Symbol: {symbol}")
    click.echo(f"  Timeframe: {timeframe}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Exchange: {exchange}")
    click.echo(f"  Dry run: {dry_run}")

    if dry_run:
        exchange = "paper"
        click.echo("  (Dry run mode - using paper trading)")

    try:
        from acms.orchestrator import Orchestrator, OrchestratorConfig
        from acms.core import ACMSConfig

        config = OrchestratorConfig(
            symbol=symbol, timeframe=timeframe,
            strategy_type=strategy, exchange=exchange,
        )
        acms_config = ACMSConfig()
        if ctx.obj.get("db_url"):
            acms_config.db_url = ctx.obj["db_url"]

        orchestrator = Orchestrator(config=config, acms_config=acms_config)

        async def _run():
            await orchestrator.start()
            click.echo("Trading system running. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
                    status = orchestrator.get_status()
                    if status["state"] not in ("running", "paused", "degraded"):
                        click.echo(f"System state: {status['state']}")
                        break
            except KeyboardInterrupt:
                click.echo("\nShutting down...")
                await orchestrator.stop()

        asyncio.run(_run())

    except Exception as e:
        click.echo(f"Error starting trading system: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--symbol", "-s", default="BTC/USDT", help="Trading symbol")
@click.option("--strategy", default="momentum_trend", help="Strategy type")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=100000.0, help="Initial capital")
@click.option("--output", "-o", default=None, help="Output file for results")
@click.pass_context
def backtest(ctx: click.Context, symbol: str, strategy: str,
             start_date: str, end_date: str, capital: float, output: Optional[str]):
    """Run a backtest from the command line.

    Executes a historical simulation of the specified strategy
    over the given date range with the provided initial capital.

    Examples:

        acms backtest -s BTC/USDT --strategy momentum_trend --start-date 2024-01-01 --end-date 2024-06-01

        acms backtest --config backtest.json --output results.json
    """
    click.echo(f"Running backtest...")
    click.echo(f"  Symbol: {symbol}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Period: {start_date} to {end_date}")
    click.echo(f"  Capital: ${capital:,.2f}")

    try:
        from acms.backtest import BacktestEngine

        engine = BacktestEngine()
        results = engine.run(
            strategy_type=strategy, symbol=symbol,
            start_date=start_date, end_date=end_date,
            initial_capital=capital,
        )

        click.echo("\n--- Backtest Results ---")
        click.echo(f"  Total Return: {results.get('total_return', 0) * 100:.2f}%")
        click.echo(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        click.echo(f"  Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
        click.echo(f"  Win Rate: {results.get('win_rate', 0) * 100:.1f}%")
        click.echo(f"  Total Trades: {results.get('total_trades', 0)}")

        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\nResults saved to {output}")

    except Exception as e:
        click.echo(f"Backtest error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model-type", default="lightgbm", help="Model type to optimize")
@click.option("--trials", default=50, help="Number of optimization trials")
@click.option("--timeout", default=600, help="Optimization timeout in seconds")
@click.option("--output", "-o", default=None, help="Output file for best params")
@click.pass_context
def optimize(ctx: click.Context, model_type: str, trials: int,
             timeout: int, output: Optional[str]):
    """Run hyperparameter optimization.

    Uses Optuna to find optimal hyperparameters for the specified
    ML model type.

    Examples:

        acms optimize --model-type lightgbm --trials 100

        acms optimize --model-type lightgbm --output best_params.json
    """
    click.echo(f"Running hyperparameter optimization...")
    click.echo(f"  Model type: {model_type}")
    click.echo(f"  Trials: {trials}")
    click.echo(f"  Timeout: {timeout}s")

    try:
        from acms.ml import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(model_type=model_type)
        click.echo("  Generating sample data...")

        # Generate synthetic data for optimization demo
        import numpy as np
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)

        results = optimizer.optimize(X, y, n_trials=trials, timeout=timeout)

        click.echo("\n--- Optimization Results ---")
        click.echo(f"  Best value: {results.get('best_value', 0):.4f}")
        click.echo(f"  Trials completed: {results.get('n_trials', 0)}")
        if results.get("best_params"):
            click.echo("  Best parameters:")
            for k, v in results["best_params"].items():
                click.echo(f"    {k}: {v}")

        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"\nResults saved to {output}")

    except Exception as e:
        click.echo(f"Optimization error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--direction", default="upgrade",
              type=click.Choice(["upgrade", "downgrade", "stamp"]),
              help="Migration direction")
@click.option("--revision", default="head", help="Target revision")
@click.pass_context
def migrate(ctx: click.Context, direction: str, revision: str):
    """Run Alembic database migrations.

    Applies pending database schema migrations using Alembic.

    Examples:

        acms migrate --direction upgrade --revision head

        acms migrate --direction downgrade --revision -1
    """
    click.echo(f"Running database migration...")
    click.echo(f"  Direction: {direction}")
    click.echo(f"  Target: {revision}")

    try:
        from acms.db import init_db, DatabaseManager

        db_url = ctx.obj.get("db_url", "postgresql://acms:acms@localhost:5432/acms")
        click.echo(f"  Database: {db_url}")

        # Check migration status
        db_manager = DatabaseManager(db_url=db_url)
        status = DatabaseManager.check_migration_status(db_manager._get_engine())
        current = status.get("current_version", "none")
        click.echo(f"  Current version: {current}")

        if direction == "upgrade":
            init_db(db_url)
            click.echo("  Migration completed successfully")
        elif direction == "stamp":
            init_db(db_url)
            click.echo(f"  Database stamped at {revision}")

    except Exception as e:
        click.echo(f"Migration error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--sample-data", is_flag=True, help="Include sample market data")
@click.pass_context
def seed(ctx: click.Context, sample_data: bool):
    """Seed database with initial data.

    Creates initial database records including default user,
    sample strategies, and optionally sample market data.

    Examples:

        acms seed

        acms seed --sample-data
    """
    click.echo("Seeding database...")

    try:
        from acms.db import DatabaseManager

        db_url = ctx.obj.get("db_url", "postgresql://acms:acms@localhost:5432/acms")
        db = DatabaseManager(db_url=db_url)

        # Create default admin user
        user_id = asyncio.run(db.create_user(
            email="admin@acms.local", password="admin123", username="admin", is_admin=True,
        ))
        click.echo(f"  Created admin user: {user_id}")

        # Create sample strategies
        for strat_type in ["momentum_trend", "mean_reversion", "carry"]:
            strat_id = asyncio.run(db.create_strategy(
                user_id=user_id, name=f"Sample {strat_type}",
                type=strat_type, symbol="BTC/USDT",
            ))
            click.echo(f"  Created strategy: {strat_id}")

        if sample_data:
            click.echo("  Generating sample market data...")
            import numpy as np
            np.random.seed(42)
            candles = []
            base_price = 50000.0
            for i in range(500):
                change = np.random.randn() * 100
                base_price += change
                candles.append({
                    "symbol": "BTC/USDT", "timeframe": "1h",
                    "open_time": f"2024-01-01T{i:02d}:00:00",
                    "open": base_price - change / 2,
                    "high": base_price + abs(change),
                    "low": base_price - abs(change),
                    "close": base_price,
                    "volume": np.random.uniform(100, 1000),
                    "exchange": "sample",
                })
            count = asyncio.run(db.bulk_insert_candles(candles))
            click.echo(f"  Inserted {count} sample candles")

        click.echo("Seeding complete!")

    except Exception as e:
        click.echo(f"Seed error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Show ACMS system status.

    Displays the current state of all ACMS components including
    database connectivity, exchange connections, and active strategies.

    Examples:

        acms status
    """
    click.echo("=== ACMS System Status ===\n")

    # Check database
    click.echo("Database:")
    db_url = ctx.obj.get("db_url", "postgresql://acms:acms@localhost:5432/acms")
    try:
        from acms.db import DatabaseManager
        db = DatabaseManager(db_url=db_url)
        engine = db._get_engine()
        migration_status = DatabaseManager.check_migration_status(engine)
        click.echo(f"  Connection: OK")
        click.echo(f"  Migration version: {migration_status.get('current_version', 'N/A')}")
    except Exception as e:
        click.echo(f"  Connection: FAILED ({e})")

    # Check Redis
    click.echo("\nRedis:")
    try:
        from acms.redis_client import get_redis
        redis = get_redis()
        click.echo(f"  Connection: OK (or in-memory fallback)")
    except Exception as e:
        click.echo(f"  Connection: FAILED ({e})")

    # Check exchanges
    click.echo("\nExchanges:")
    for exchange in ["binance", "bybit", "okx", "paper"]:
        try:
            from acms.exchanges import create_exchange_adapter
            adapter = create_exchange_adapter(exchange)
            click.echo(f"  {exchange}: available")
        except Exception as e:
            click.echo(f"  {exchange}: error ({e})")

    # Check configuration
    click.echo("\nConfiguration:")
    config = ctx.obj.get("config", {})
    if config:
        click.echo(f"  Config file: {ctx.obj.get('config_path', 'none')}")
        click.echo(f"  Settings: {len(config)} keys loaded")
    else:
        click.echo("  No configuration file loaded")

    click.echo("\n=== End Status ===")


def main():
    """Entry point for the ACMS CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
