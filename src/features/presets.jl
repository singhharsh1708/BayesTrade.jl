"""
Assembled feature sets.

The default set is deliberately small. Every feature added is another parameter for a model to
fit and another way to find a pattern that was never there, and a Bayesian model with a proper
prior pays for width in wider posteriors rather than in worse point estimates. Features here
earn their place by measuring something the others do not.
"""

"""
    minimal_feature_set()

The smallest set the first Bayesian return model needs.

One recent return, one trend measure and one volatility measure. Enough to fit something
meaningful, short enough to warm up in a quarter.
"""
minimal_feature_set() = FeatureSet(
    Feature[LogReturn(1), Momentum(20, 1), RealisedVolatility(20)],
)

"""
    default_feature_set()

The standard set: trend, mean reversion, volatility and liquidity.

Warms up in roughly a trading year, which is the binding constraint on how early a backtest can
start.
"""
default_feature_set() = FeatureSet(
    Feature[
        LogReturn(1),
        LogReturn(5),
        Momentum(60, 5),
        Momentum(120, 20),
        PriceToMovingAverage(50),
        MovingAverageSpread(12, 26),
        TrendSlope(60),
        TrendQuality(60),
        PriceZScore(20),
        RelativeStrengthIndex(14),
        DrawdownFromHigh(120),
        RealisedVolatility(20),
        EwmaVolatility(20),
        ParkinsonVolatility(20),
        DownsideVolatility(60),
        VolatilityRatio(10, 120),
        AverageTrueRange(14),
        RelativeVolume(20),
        VolumeZScore(60),
        MedianTurnover(20),
        AmihudIlliquidity(20),
    ],
)
