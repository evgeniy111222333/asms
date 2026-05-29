//! Real-time exposure tracker
//!
//! Tracks net, gross, delta-adjusted, and correlation-adjusted
//! portfolio exposure across all positions. Supports beta adjustment,
//! concentration HHI, and FX exposure.

use crate::core::types::*;
use dashmap::DashMap;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exposure metrics for the portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureMetrics {
    pub net_exposure: Decimal,
    pub gross_exposure: Decimal,
    pub long_exposure: Decimal,
    pub short_exposure: Decimal,
    pub delta_adjusted_exposure: Decimal,
    pub beta_adjusted_exposure: Decimal,
    pub correlation_adjusted_exposure: Decimal,
    pub fx_exposure: HashMap<String, Decimal>,
    pub concentration: HashMap<Symbol, Decimal>,
    pub concentration_hhi: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for ExposureMetrics {
    fn default() -> Self {
        ExposureMetrics {
            net_exposure: Decimal::ZERO,
            gross_exposure: Decimal::ZERO,
            long_exposure: Decimal::ZERO,
            short_exposure: Decimal::ZERO,
            delta_adjusted_exposure: Decimal::ZERO,
            beta_adjusted_exposure: Decimal::ZERO,
            correlation_adjusted_exposure: Decimal::ZERO,
            fx_exposure: HashMap::new(),
            concentration: HashMap::new(),
            concentration_hhi: Decimal::ZERO,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Position beta relative to a benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionBeta {
    pub symbol: Symbol,
    pub beta: Decimal,
    pub correlation: Decimal,
}

/// Exposure tracker
pub struct ExposureTracker {
    positions: DashMap<Symbol, Position>,
    betas: DashMap<Symbol, Decimal>,
    correlations: DashMap<(Symbol, Symbol), Decimal>,
    fx_rates: DashMap<String, Decimal>,
    portfolio_value: Decimal,
}

impl ExposureTracker {
    pub fn new(portfolio_value: Decimal) -> Self {
        ExposureTracker {
            positions: DashMap::new(),
            betas: DashMap::new(),
            correlations: DashMap::new(),
            fx_rates: DashMap::new(),
            portfolio_value,
        }
    }

    /// Update a position
    pub fn update_position(&self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position
    pub fn remove_position(&self, symbol: &Symbol) {
        self.positions.remove(symbol);
    }

    /// Update beta for a symbol
    pub fn update_beta(&self, symbol: Symbol, beta: Decimal) {
        self.betas.insert(symbol, beta);
    }

    /// Update correlation between two symbols
    pub fn update_correlation(&self, a: Symbol, b: Symbol, corr: Decimal) {
        self.correlations.insert((a, b), corr);
    }

    /// Update FX rate for a currency
    pub fn update_fx_rate(&self, currency: String, rate: Decimal) {
        self.fx_rates.insert(currency, rate);
    }

    /// Compute net exposure (long - short)
    pub fn net_exposure(&self) -> Decimal {
        self.positions.iter().map(|r| {
            let pos = r.value();
            match pos.side {
                Side::Buy => pos.notional_value(),
                Side::Sell => -pos.notional_value(),
            }
        }).sum()
    }

    /// Compute gross exposure (long + short)
    pub fn gross_exposure(&self) -> Decimal {
        self.long_exposure() + self.short_exposure()
    }

    /// Compute long exposure only
    pub fn long_exposure(&self) -> Decimal {
        self.positions.iter()
            .filter(|r| r.value().side == Side::Buy)
            .map(|r| r.value().notional_value())
            .sum()
    }

    /// Compute short exposure only
    pub fn short_exposure(&self) -> Decimal {
        self.positions.iter()
            .filter(|r| r.value().side == Side::Sell)
            .map(|r| r.value().notional_value())
            .sum()
    }

    /// Compute delta-adjusted exposure (accounts for leverage)
    pub fn delta_adjusted(&self) -> Decimal {
        self.positions.iter().map(|r| {
            let pos = r.value();
            let signed = match pos.side {
                Side::Buy => pos.notional_value(),
                Side::Sell => -pos.notional_value(),
            };
            signed * pos.leverage
        }).sum()
    }

    /// Compute beta-adjusted exposure
    pub fn beta_adjusted(&self) -> Decimal {
        self.positions.iter().map(|r| {
            let pos = r.value();
            let signed = match pos.side {
                Side::Buy => pos.notional_value(),
                Side::Sell => -pos.notional_value(),
            };
            let beta = self.betas.get(&pos.symbol).map(|b| *b.value()).unwrap_or(Decimal::ONE);
            signed * beta
        }).sum()
    }

    /// Compute concentration HHI (Herfindahl-Hirschman Index)
    ///
    /// HHI = sum of (weight_i)^2 where weight_i = notional_i / total_notional
    /// Range: 1/N (perfectly diversified) to 1.0 (single position)
    pub fn concentration_hhi(&self) -> Decimal {
        let total: Decimal = self.positions.iter()
            .map(|r| r.value().notional_value())
            .sum();

        if total <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        self.positions.iter()
            .map(|r| {
                let weight = r.value().notional_value() / total;
                weight * weight
            })
            .sum()
    }

    /// Compute all exposure metrics
    pub fn compute_metrics(&self) -> ExposureMetrics {
        let mut long_exp = Decimal::ZERO;
        let mut short_exp = Decimal::ZERO;
        let mut net_exp = Decimal::ZERO;
        let mut delta_adj = Decimal::ZERO;
        let mut beta_adj = Decimal::ZERO;
        let mut corr_adj = Decimal::ZERO;
        let mut fx_exp: HashMap<String, Decimal> = HashMap::new();
        let mut concentration: HashMap<Symbol, Decimal> = HashMap::new();

        let positions: Vec<Position> = self.positions.iter().map(|r| r.value().clone()).collect();
        let symbols: Vec<Symbol> = positions.iter().map(|p| p.symbol.clone()).collect();

        for pos in &positions {
            let notional = pos.notional_value();
            let signed = match pos.side {
                Side::Buy => notional,
                Side::Sell => -notional,
            };

            if signed > Decimal::ZERO {
                long_exp += notional;
            } else {
                short_exp += notional;
            }

            net_exp += signed;
            delta_adj += signed * pos.leverage;

            let beta = self.betas.get(&pos.symbol).map(|b| *b.value()).unwrap_or(Decimal::ONE);
            beta_adj += signed * beta;

            if let Some(quote) = pos.symbol.quote() {
                *fx_exp.entry(quote.to_string()).or_insert(Decimal::ZERO) += notional;
            }

            if self.portfolio_value > Decimal::ZERO {
                concentration.insert(pos.symbol.clone(), notional / self.portfolio_value);
            }
        }

        // Correlation-adjusted exposure
        if !symbols.is_empty() {
            let mut variance_sum = Decimal::ZERO;
            for sym in &symbols {
                let pos = self.positions.get(sym).map(|r| r.value().notional_value()).unwrap_or(Decimal::ZERO);
                let side = self.positions.get(sym).map(|r| r.value().side).unwrap_or(Side::Buy);
                let signed = match side { Side::Buy => pos, Side::Sell => -pos };
                variance_sum += signed * signed;
            }

            for (i, sym_a) in symbols.iter().enumerate() {
                let pos_a = self.positions.get(sym_a).map(|r| r.value().notional_value()).unwrap_or(Decimal::ZERO);
                let side_a = self.positions.get(sym_a).map(|r| r.value().side).unwrap_or(Side::Buy);
                let signed_a = match side_a { Side::Buy => pos_a, Side::Sell => -pos_a };

                for sym_b in symbols.iter().skip(i + 1) {
                    let corr = self.correlations.get(&(sym_a.clone(), sym_b.clone()))
                        .map(|r| *r.value())
                        .unwrap_or_else(|| {
                            self.correlations.get(&(sym_b.clone(), sym_a.clone()))
                                .map(|r| *r.value())
                                .unwrap_or(Decimal::ZERO)
                        });
                    let pos_b = self.positions.get(sym_b).map(|r| r.value().notional_value()).unwrap_or(Decimal::ZERO);
                    let side_b = self.positions.get(sym_b).map(|r| r.value().side).unwrap_or(Side::Buy);
                    let signed_b = match side_b { Side::Buy => pos_b, Side::Sell => -pos_b };
                    variance_sum += rust_decimal_macros::dec!(2) * signed_a * signed_b * corr;
                }
            }

            if variance_sum > Decimal::ZERO {
                let var_f: f64 = variance_sum.to_string().parse().unwrap_or(0.0);
                corr_adj = Decimal::from_f64_retain(var_f.sqrt()).unwrap_or(Decimal::ZERO);
            } else {
                corr_adj = Decimal::ZERO;
            }
        } else {
            corr_adj = Decimal::ZERO;
        }

        let hhi = self.concentration_hhi();

        ExposureMetrics {
            net_exposure: net_exp,
            gross_exposure: long_exp + short_exp,
            long_exposure: long_exp,
            short_exposure: short_exp,
            delta_adjusted_exposure: delta_adj,
            beta_adjusted_exposure: beta_adj,
            correlation_adjusted_exposure: corr_adj,
            fx_exposure: fx_exp,
            concentration,
            concentration_hhi: hhi,
            timestamp: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_position(symbol: &str, side: Side, qty: Decimal, price: Decimal, leverage: Decimal) -> Position {
        Position {
            symbol: Symbol::new(symbol),
            side,
            quantity: qty,
            entry_price: price,
            mark_price: price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            liquidation_price: None,
            leverage,
            margin: Decimal::ZERO,
            exchange: ExchangeId::Paper,
            opened_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_exposure_metrics() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(51000), Decimal::ONE));

        let metrics = tracker.compute_metrics();
        assert_eq!(metrics.long_exposure, dec!(51000));
        assert_eq!(metrics.net_exposure, dec!(51000));
        assert_eq!(metrics.gross_exposure, dec!(51000));
    }

    #[test]
    fn test_net_exposure_with_short() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/USDT", Side::Sell, dec!(10), dec!(3000), Decimal::ONE));

        let net = tracker.net_exposure();
        assert_eq!(net, dec!(20000)); // 50000 - 30000
        let gross = tracker.gross_exposure();
        assert_eq!(gross, dec!(80000)); // 50000 + 30000
    }

    #[test]
    fn test_delta_adjusted() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), dec!(5)));

        let delta = tracker.delta_adjusted();
        assert_eq!(delta, dec!(250000)); // 50000 * 5x leverage
    }

    #[test]
    fn test_remove_position() {
        let tracker = ExposureTracker::new(dec!(100000));
        let symbol = Symbol::new("BTC/USDT");
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        assert_eq!(tracker.net_exposure(), dec!(50000));
        tracker.remove_position(&symbol);
        assert_eq!(tracker.net_exposure(), Decimal::ZERO);
    }

    #[test]
    fn test_fx_exposure() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));

        let metrics = tracker.compute_metrics();
        assert_eq!(*metrics.fx_exposure.get("USDT").unwrap_or(&Decimal::ZERO), dec!(50000));
    }

    #[test]
    fn test_beta_adjusted() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_beta(Symbol::new("BTC/USDT"), dec!(1.5));

        let beta_adj = tracker.beta_adjusted();
        assert_eq!(beta_adj, dec!(75000)); // 50000 * 1.5
    }

    #[test]
    fn test_concentration_hhi_single() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));

        let hhi = tracker.concentration_hhi();
        assert_eq!(hhi, Decimal::ONE); // Single position = 1.0
    }

    #[test]
    fn test_concentration_hhi_equal() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));

        let hhi = tracker.concentration_hhi();
        // Each position is 50%, HHI = 0.25 + 0.25 = 0.5
        assert_eq!(hhi, dec!(0.5));
    }

    #[test]
    fn test_concentration_hhi_diversified() {
        let tracker = ExposureTracker::new(dec!(100000));
        for i in 0..4 {
            let sym = format!("SYM{}/USDT", i);
            tracker.update_position(make_position(&sym, Side::Buy, dec!(1), dec!(25000), Decimal::ONE));
        }
        let hhi = tracker.concentration_hhi();
        // Each is 25%, HHI = 4 * 0.0625 = 0.25
        assert_eq!(hhi, dec!(0.25));
    }

    #[test]
    fn test_long_exposure_only() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/USDT", Side::Sell, dec!(10), dec!(3000), Decimal::ONE));

        assert_eq!(tracker.long_exposure(), dec!(50000));
    }

    #[test]
    fn test_short_exposure_only() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/USDT", Side::Sell, dec!(10), dec!(3000), Decimal::ONE));

        assert_eq!(tracker.short_exposure(), dec!(30000));
    }

    #[test]
    fn test_correlation_adjusted() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/USDT", Side::Buy, dec!(10), dec!(3000), Decimal::ONE));
        tracker.update_correlation(Symbol::new("BTC/USDT"), Symbol::new("ETH/USDT"), dec!(0.8));

        let metrics = tracker.compute_metrics();
        // under correct portfolio risk math:
        // var = 50000^2 + 30000^2 + 2*50000*30000*0.8 = 5.8e9
        // corr_adj = sqrt(5.8e9) = 76157.73
        assert!(metrics.correlation_adjusted_exposure < metrics.net_exposure);
        assert!(metrics.correlation_adjusted_exposure > dec!(76150));
        assert!(metrics.correlation_adjusted_exposure < dec!(76165));

        // Hedged case: BTC Buy, ETH Sell
        let tracker_hedged = ExposureTracker::new(dec!(100000));
        tracker_hedged.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker_hedged.update_position(make_position("ETH/USDT", Side::Sell, dec!(10), dec!(3000), Decimal::ONE));
        tracker_hedged.update_correlation(Symbol::new("BTC/USDT"), Symbol::new("ETH/USDT"), dec!(0.8));

        let metrics_hedged = tracker_hedged.compute_metrics();
        // net_exposure = 20000
        // var = 50000^2 + 30000^2 - 2*50000*30000*0.8 = 1.0e9
        // corr_adj = sqrt(1e9) = 31622.78
        assert!(metrics_hedged.correlation_adjusted_exposure > metrics_hedged.net_exposure);
        assert!(metrics_hedged.correlation_adjusted_exposure > dec!(31620));
        assert!(metrics_hedged.correlation_adjusted_exposure < dec!(31630));
    }

    #[test]
    fn test_fx_exposure_multiple_currencies() {
        let tracker = ExposureTracker::new(dec!(100000));
        tracker.update_position(make_position("BTC/USDT", Side::Buy, dec!(1), dec!(50000), Decimal::ONE));
        tracker.update_position(make_position("ETH/BTC", Side::Buy, dec!(5), dec!(0.5), Decimal::ONE));

        let metrics = tracker.compute_metrics();
        assert!(metrics.fx_exposure.contains_key("USDT"));
        assert!(metrics.fx_exposure.contains_key("BTC"));
    }
}
