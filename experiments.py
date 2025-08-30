from tradier_impl import getAvgVol as getAvgVolTradier, _get_underlying_price as getUnderlyingPriceTradier, _get_filtered_expirations as getFilteredExpirationsTradier, _get_atm_iv as getAtmIvTradier

from polygon_io_impl import getAvgVol as getAvgVolPolygon, _get_underlying_price as getUnderlyingPricePolygon
import asyncio

# Volume of stock trades
async def testAvgVol():
    ticker = "AMZN"
    tradier_avg_vol = await getAvgVolTradier(ticker)
    tradier_stock_price = await getUnderlyingPriceTradier(ticker)
    print(f"Tradier avg vol = {round(tradier_avg_vol)}")
    print(f"Tradier Underlying Price = {tradier_stock_price}")
    tradier_expirations = await getFilteredExpirationsTradier(ticker)
    atm_iv, straddle = await getAtmIvTradier(ticker, tradier_stock_price, tradier_expirations);
    print(f"tradier atm_iv", atm_iv)
    print(f"tradier straddle", straddle)
    
    print("")
    polygon_avg_vol = getAvgVolPolygon(ticker)
    print(f"Polygon IO avg vol = {round(polygon_avg_vol)}")
    print(f"Polygon IO Underlying Price = {getUnderlyingPricePolygon(ticker)}")


if __name__ == "__main__":
    asyncio.run(testAvgVol())
