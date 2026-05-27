"""
tests/test_transformations.py
------------------------------
Unit tests for the transform.py logic.

Strategy
--------
We spin up a *local* SparkSession (no Hive, no HDFS) and apply the same
transformations that transform.py runs on the raw data.  This lets tests
run in CI without any Docker infrastructure.

Run:
    pytest tests/test_transformations.py -v
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, round as spark_round, to_timestamp, year
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """
    Local SparkSession shared across all tests in this module.
    Hive support is NOT enabled — we use in-memory DataFrames only.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ecommerce-transform-tests")
        .config("spark.sql.shuffle.partitions", "2")  # keep tests fast
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()


# Raw data schema mirrors the CSV / Hive raw table
RAW_SCHEMA = StructType(
    [
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", DoubleType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Country", StringType(), True),
    ]
)

SAMPLE_ROWS = [
    # (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
    ("536365", "85123A", "CREAM HANGING HEART",    6,  "12/1/2010 8:26",  2.55,  "17850", "United Kingdom"),
    ("536366", "22633",  "HAND WARMER UNION JACK",  6,  "12/1/2010 8:28",  1.85,  "17850", "United Kingdom"),
    # Negative Quantity — should be filtered out
    ("536367", "22632",  "HAND WARMER RED POLKA",  -1,  "12/1/2010 8:34",  1.85,  "17850", "United Kingdom"),
    # Zero UnitPrice — should be filtered out
    ("536368", "21326",  "MATHS BLACKBOARD CHALK",  1,  "12/1/2010 8:41",  0.00,  "17850", "United Kingdom"),
    # Different country for aggregation tests
    ("536369", "84406B", "CREAM CUPID HEARTS",     8,  "12/1/2010 8:45",  1.69,  "13047", "France"),
    # Different month/year for monthly aggregation tests
    ("536380", "35004C", "SET OF 3 BUTTERFLY COOKIE",  6, "1/5/2011 9:00", 4.25, "13047", "France"),
]


@pytest.fixture(scope="module")
def raw_df(spark):
    return spark.createDataFrame(SAMPLE_ROWS, schema=RAW_SCHEMA)


@pytest.fixture(scope="module")
def transformed_df(raw_df):
    """Apply the same transformations as transform.py."""
    return (
        raw_df.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"))
        .withColumn("Month", month(col("InvoiceDate")))
        .withColumn("Year", year(col("InvoiceDate")))
        .withColumn("TotalAmount", spark_round(col("Quantity") * col("UnitPrice"), 2))
        .filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
    )


# ---------------------------------------------------------------------------
# Tests — basic filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_negative_quantity_rows_removed(self, transformed_df):
        """Rows with Quantity <= 0 must be excluded."""
        bad = transformed_df.filter(col("Quantity") <= 0)
        assert bad.count() == 0, "Negative/zero Quantity rows were not filtered"

    def test_zero_unit_price_rows_removed(self, transformed_df):
        """Rows with UnitPrice <= 0 must be excluded."""
        bad = transformed_df.filter(col("UnitPrice") <= 0)
        assert bad.count() == 0, "Zero UnitPrice rows were not filtered"

    def test_valid_row_count(self, transformed_df):
        """4 out of 6 sample rows should survive filtering."""
        assert transformed_df.count() == 4


# ---------------------------------------------------------------------------
# Tests — derived columns
# ---------------------------------------------------------------------------

class TestDerivedColumns:
    def test_total_amount_calculated(self, transformed_df):
        """TotalAmount = round(Quantity * UnitPrice, 2)."""
        row = (
            transformed_df.filter(col("InvoiceNo") == "536365")
            .select("TotalAmount")
            .first()
        )
        assert row is not None
        assert abs(row["TotalAmount"] - 15.30) < 0.01, (
            f"Expected 15.30, got {row['TotalAmount']}"
        )

    def test_month_extracted(self, transformed_df):
        """Month column must be extracted correctly from InvoiceDate."""
        row = (
            transformed_df.filter(col("InvoiceNo") == "536365")
            .select("Month")
            .first()
        )
        assert row["Month"] == 12

    def test_year_extracted(self, transformed_df):
        """Year column must be extracted correctly from InvoiceDate."""
        row = (
            transformed_df.filter(col("InvoiceNo") == "536365")
            .select("Year")
            .first()
        )
        assert row["Year"] == 2010

    def test_invoice_date_is_timestamp(self, transformed_df):
        """InvoiceDate must be cast to TimestampType."""
        field = next(
            f for f in transformed_df.schema.fields if f.name == "InvoiceDate"
        )
        assert isinstance(field.dataType, TimestampType), (
            f"Expected TimestampType, got {field.dataType}"
        )


# ---------------------------------------------------------------------------
# Tests — sales_per_country aggregation
# ---------------------------------------------------------------------------

class TestSalesPerCountry:
    @pytest.fixture(scope="class")
    def sales_per_country(self, transformed_df):
        from pyspark.sql.functions import sum as spark_sum
        return (
            transformed_df.groupBy("Country")
            .agg(spark_sum("TotalAmount").alias("TotalSales"))
        )

    def test_country_count(self, sales_per_country):
        """Sample data has two countries."""
        assert sales_per_country.count() == 2

    def test_uk_sales(self, sales_per_country):
        """UK total = 6*2.55 + 6*1.85 = 15.30 + 11.10 = 26.40."""
        row = sales_per_country.filter(col("Country") == "United Kingdom").first()
        assert row is not None
        assert abs(row["TotalSales"] - 26.40) < 0.01


# ---------------------------------------------------------------------------
# Tests — monthly_sales aggregation
# ---------------------------------------------------------------------------

class TestMonthlySales:
    @pytest.fixture(scope="class")
    def monthly_sales(self, transformed_df):
        from pyspark.sql.functions import sum as spark_sum
        return (
            transformed_df.groupBy("Year", "Month")
            .agg(spark_sum("TotalAmount").alias("MonthlySales"))
        )

    def test_monthly_row_count(self, monthly_sales):
        """Sample data spans 2 year-month combos (Dec 2010, Jan 2011)."""
        assert monthly_sales.count() == 2

    def test_column_named_monthly_sales(self, monthly_sales):
        """Output column must be MonthlySales — not TotalSales (transform_simple.py bug)."""
        assert "MonthlySales" in monthly_sales.columns


# ---------------------------------------------------------------------------
# Tests — customer_metrics aggregation
# ---------------------------------------------------------------------------

class TestCustomerMetrics:
    @pytest.fixture(scope="class")
    def customer_metrics(self, transformed_df):
        from pyspark.sql.functions import count, sum as spark_sum
        return (
            transformed_df.groupBy("CustomerID")
            .agg(
                spark_sum("TotalAmount").alias("TotalPurchases"),
                count("InvoiceNo").alias("NumberOfTransactions"),
            )
        )

    def test_column_names(self, customer_metrics):
        """
        Critical: ml_customer_segmentation.py depends on exactly these names.
        If they change here they must change in the ML script too.
        """
        cols = customer_metrics.columns
        assert "TotalPurchases" in cols, "'TotalPurchases' column missing"
        assert "NumberOfTransactions" in cols, "'NumberOfTransactions' column missing"

    def test_customer_count(self, customer_metrics):
        """Two distinct customers in sample data."""
        assert customer_metrics.count() == 2

    def test_customer_17850_total(self, customer_metrics):
        """CustomerID 17850: 2 valid transactions, total = 15.30 + 11.10 = 26.40."""
        row = customer_metrics.filter(col("CustomerID") == "17850").first()
        assert row is not None
        assert abs(row["TotalPurchases"] - 26.40) < 0.01
        assert row["NumberOfTransactions"] == 2


# ---------------------------------------------------------------------------
# Tests — product_performance aggregation
# ---------------------------------------------------------------------------

class TestProductPerformance:
    @pytest.fixture(scope="class")
    def product_performance(self, transformed_df):
        from pyspark.sql.functions import sum as spark_sum
        return (
            transformed_df.groupBy("StockCode", "Description")
            .agg(
                spark_sum("Quantity").alias("TotalQuantitySold"),
                spark_sum("TotalAmount").alias("TotalRevenue"),
            )
        )

    def test_product_count(self, product_performance):
        """4 distinct valid products in sample data."""
        assert product_performance.count() == 4

    def test_column_names(self, product_performance):
        cols = product_performance.columns
        assert "TotalQuantitySold" in cols
        assert "TotalRevenue" in cols
