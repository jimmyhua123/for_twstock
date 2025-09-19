"""資料集設定與欄位對映集中管理。"""

from __future__ import annotations

from typing import Dict, List

DATASET_CATALOG: Dict[str, Dict[str, object]] = {
    "TaiwanStockPrice": {
        "description": "台股日價量",
        "normalizer": "normalize_taiwan_stock_price",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockPriceAdj": {
        "description": "台股日價量(還原權息)",
        "normalizer": "normalize_taiwan_stock_price_adj",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "description": "三大法人買賣超",
        "normalizer": "normalize_institutional_investors",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "description": "融資融券餘額",
        "normalizer": "normalize_margin_short",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockMonthRevenue": {
        "description": "月營收",
        "normalizer": "normalize_month_revenue",
        "requires_stock": True,
        "frequency": "M",
    },
}

DATASET_FILENAME_TAG: Dict[str, str] = {
    "TaiwanStockPrice": "stockprice",
    "TaiwanStockPriceAdj": "stockprice_adj",
    "TaiwanStockInstitutionalInvestorsBuySell": "buysell",
    "TaiwanStockMarginPurchaseShortSale": "margin_short",
    "TaiwanStockMonthRevenue": "monthrevenue",
}

COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "TaiwanStockPrice": {
        "max": "high",
        "min": "low",
        "Trading_Volume": "volume",
        "Trading_money": "turnover",
        "Trading_turnover": "transactions",
    },
    "TaiwanStockPriceAdj": {
        "max": "high",
        "min": "low",
        "Trading_Volume": "volume",
        "Trading_money": "turnover",
        "Trading_turnover": "transactions",
        "Adj_Close": "adj_close",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "Foreign_Investor_Net_Buy_Sell": "foreign",
        "Investment_Trust_Net_Buy_Sell": "investment_trust",
        "Dealer_Net_Buy_Sell": "dealer",
        "Dealer_Self_Net_Buy_Sell": "dealer_self",
        "Dealer_Hedging_Net_Buy_Sell": "dealer_hedging",
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "MarginPurchaseTodayBalance": "margin_balance",
        "ShortSaleTodayBalance": "short_balance",
        "MarginPurchaseChange": "margin_change",
        "ShortSaleChange": "short_change",
    },
    "TaiwanStockMonthRevenue": {
        "revenue": "revenue",
        "revenue_month": "revenue_month",
        "revenue_year": "revenue_year",
        "revenue_last_month": "revenue_last_month",
        "revenue_last_year": "revenue_last_year",
        "revenue_month_growth": "revenue_month_growth",
        "revenue_year_growth": "revenue_year_growth",
        "accumulated_revenue": "accumulated_revenue",
        "accumulated_revenue_last_year": "accumulated_revenue_last_year",
        "accumulated_revenue_growth": "accumulated_revenue_growth",
    },
}

NUMERIC_COLUMN_HINTS: Dict[str, List[str]] = {
    "TaiwanStockPrice": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "spread",
        "transactions",
    ],
    "TaiwanStockPriceAdj": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "adj_close",
        "spread",
        "transactions",
    ],
    "TaiwanStockInstitutionalInvestorsBuySell": [
        "foreign",
        "investment_trust",
        "dealer",
        "dealer_self",
        "dealer_hedging",
        "total",
    ],
    "TaiwanStockMarginPurchaseShortSale": [
        "margin_balance",
        "short_balance",
        "margin_change",
        "short_change",
    ],
    "TaiwanStockMonthRevenue": [
        "revenue",
        "revenue_month",
        "revenue_year",
        "revenue_last_month",
        "revenue_last_year",
        "revenue_month_growth",
        "revenue_year_growth",
        "accumulated_revenue",
        "accumulated_revenue_last_year",
        "accumulated_revenue_growth",
    ],
}

__all__ = [
    "DATASET_CATALOG",
    "DATASET_FILENAME_TAG",
    "COLUMN_MAP",
    "NUMERIC_COLUMN_HINTS",
]
