"""Stress Testing for ACMS."""

import numpy as np
from typing import Optional, Dict, List
from acms.core import Position, Side


class StressTesting:
    """Stress testing with historical scenario replay and custom scenarios.

    Includes both synthetic stress scenarios and historically-calibrated
    scenario replays based on actual market events.
    """

    SCENARIOS = {
        "flash_crash": {"equity_shock": -0.20, "vol_mult": 5.0, "corr_to_1": True},
        "slow_bleed": {"equity_shock": -0.10, "vol_mult": 2.0, "corr_to_1": False},
        "vol_spike": {"equity_shock": -0.05, "vol_mult": 4.0, "corr_to_1": False},
        "liquidity_crisis": {"equity_shock": -0.15, "vol_mult": 3.0, "corr_to_1": True, "spread_mult": 10.0},
        "correlation_breakdown": {"equity_shock": -0.08, "vol_mult": 2.5, "corr_to_1": True},
        "black_swan": {"equity_shock": -0.40, "vol_mult": 8.0, "corr_to_1": True, "spread_mult": 20.0},
        "covid_crash_march2020": {"equity_shock": -0.35, "vol_mult": 6.0, "corr_to_1": True, "spread_mult": 15.0},
        "ftx_collapse_nov2022": {"equity_shock": -0.25, "vol_mult": 5.0, "corr_to_1": True, "spread_mult": 12.0},
        "luna_crash_may2022": {"equity_shock": -0.60, "vol_mult": 10.0, "corr_to_1": True, "spread_mult": 25.0},
    }

    # Historical scenario dates and detailed parameters
    HISTORICAL_SCENARIOS = {
        "covid_crash_feb_mar_2020": {
            "description": "COVID-19 market crash Feb-Mar 2020",
            "start_date": "2020-02-19",
            "end_date": "2020-03-23",
            "equity_shock": -0.35,
            "vol_mult": 6.0,
            "corr_to_1": True,
            "spread_mult": 15.0,
            "btc_shock": -0.50,
            "alt_shock": -0.65,
            "recovery_days": 150,
        },
        "ftx_collapse_nov_2022": {
            "description": "FTX exchange collapse Nov 2022",
            "start_date": "2022-11-06",
            "end_date": "2022-11-14",
            "equity_shock": -0.25,
            "vol_mult": 5.0,
            "corr_to_1": True,
            "spread_mult": 12.0,
            "btc_shock": -0.25,
            "alt_shock": -0.50,
            "recovery_days": 60,
        },
        "luna_crash_may_2022": {
            "description": "Terra/Luna ecosystem collapse May 2022",
            "start_date": "2022-05-07",
            "end_date": "2022-05-18",
            "equity_shock": -0.60,
            "vol_mult": 10.0,
            "corr_to_1": True,
            "spread_mult": 25.0,
            "btc_shock": -0.30,
            "alt_shock": -0.70,
            "recovery_days": 90,
        },
        "china_ban_may_2021": {
            "description": "China cryptocurrency ban May 2021",
            "start_date": "2021-05-12",
            "end_date": "2021-05-23",
            "equity_shock": -0.35,
            "vol_mult": 4.0,
            "corr_to_1": True,
            "spread_mult": 8.0,
            "btc_shock": -0.35,
            "alt_shock": -0.55,
            "recovery_days": 45,
        },
    }

    def run_scenario(self, positions: List[Position], scenario_name: str,
                     correlations: Optional[np.ndarray] = None) -> dict:
        """Run a stress scenario on current positions.

        Args:
            positions: List of current positions.
            scenario_name: Name of the scenario to run.
            correlations: Optional correlation matrix for positions.

        Returns:
            Dict with scenario results per position and total PnL.
        """
        scenario = self.SCENARIOS.get(scenario_name)
        if scenario is None:
            return {"error": f"Unknown scenario: {scenario_name}"}

        equity_shock = scenario["equity_shock"]
        total_pnl = 0.0
        position_results = []

        for pos in positions:
            pos_shock = equity_shock * (1.5 if pos.leverage > 1 else 1.0)
            if pos.side == Side.BUY:
                pnl = pos.notional_value * pos_shock
            else:
                pnl = pos.notional_value * (-pos_shock)

            position_results.append({
                "symbol": pos.symbol, "pnl": pnl, "shock_pct": pos_shock * 100,
            })
            total_pnl += pnl

        if scenario.get("corr_to_1") and correlations is not None:
            correlation_penalty = abs(total_pnl) * 0.2
            total_pnl += -correlation_penalty * np.sign(total_pnl)

        return {
            "scenario": scenario_name, "total_pnl": total_pnl,
            "position_results": position_results, "parameters": scenario,
        }

    def run_all_scenarios(self, positions: List[Position],
                          correlations: Optional[np.ndarray] = None) -> Dict[str, dict]:
        """Run all stress scenarios.

        Args:
            positions: List of current positions.
            correlations: Optional correlation matrix.

        Returns:
            Dict mapping scenario name to results.
        """
        return {name: self.run_scenario(positions, name, correlations)
                for name in self.SCENARIOS}

    def run_historical_scenario(self, positions: List[Position],
                                scenario_name: str,
                                is_alt: Optional[Dict[str, bool]] = None) -> Dict:
        """Run a historically-calibrated scenario replay.

        Applies historically-accurate shock parameters including
        differentiated shocks for BTC vs altcoins.

        Args:
            positions: List of current positions.
            scenario_name: Name from HISTORICAL_SCENARIOS.
            is_alt: Dict mapping symbol to True if altcoin, False if BTC.

        Returns:
            Dict with detailed scenario replay results.
        """
        scenario = self.HISTORICAL_SCENARIOS.get(scenario_name)
        if scenario is None:
            return {"error": f"Unknown historical scenario: {scenario_name}"}

        if is_alt is None:
            is_alt = {}

        total_pnl = 0.0
        position_results = []

        for pos in positions:
            alt = is_alt.get(pos.symbol, True)
            shock = scenario["alt_shock"] if alt else scenario["btc_shock"]
            shock *= (1.5 if pos.leverage > 1 else 1.0)

            if pos.side == Side.BUY:
                pnl = pos.notional_value * shock
            else:
                pnl = pos.notional_value * (-shock)

            position_results.append({
                "symbol": pos.symbol,
                "pnl": pnl,
                "shock_pct": shock * 100,
                "is_alt": alt,
                "recovery_estimate_days": scenario["recovery_days"],
            })
            total_pnl += pnl

        return {
            "scenario": scenario_name,
            "description": scenario["description"],
            "start_date": scenario["start_date"],
            "end_date": scenario["end_date"],
            "total_pnl": total_pnl,
            "position_results": position_results,
            "parameters": scenario,
        }

    def run_all_historical_scenarios(self, positions: List[Position],
                                     is_alt: Optional[Dict[str, bool]] = None) -> Dict[str, dict]:
        """Run all historical scenario replays.

        Args:
            positions: List of current positions.
            is_alt: Dict mapping symbol to whether it's an altcoin.

        Returns:
            Dict mapping scenario name to results.
        """
        return {name: self.run_historical_scenario(positions, name, is_alt)
                for name in self.HISTORICAL_SCENARIOS}

__all__ = ['StressTesting']
