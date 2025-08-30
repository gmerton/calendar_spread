from tradier_impl import getAvgVol as getAvgVolTradier
from polygon_io_impl import getAvgVol as getAvgVolPolygon
import asyncio

# Volume of stock trades
async def testAvgVol():
    ticker = "AMZN"
    tradier_avg_vol = await getAvgVolTradier(ticker)
    print(f"Tradier avg vol = {round(tradier_avg_vol)}")
    polygon_avg_vol = getAvgVolPolygon(ticker)
    print(f"Polygon IO avg vol = {round(polygon_avg_vol)}")

if __name__ == "__main__":
    asyncio.run(testAvgVol())
