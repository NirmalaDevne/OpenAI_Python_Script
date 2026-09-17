import os
import sys
import json
import time
import logging
from datetime import date, timedelta

import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "creds.env"))

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://api.ads.openai.com/v1"

OPENAI_ADS_API_KEY = os.getenv("OPENAI_ADS_API_KEY")

DAYS_TO_FETCH = int(os.getenv("DAYS_TO_FETCH", "30"))
REPORTING_TIMEZONE = os.getenv("REPORTING_TIMEZONE", "UTC")
INSIGHTS_PAGE_SIZE = int(os.getenv("INSIGHTS_PAGE_SIZE", "2000"))
CONVERSION_BATCH_SIZE = int(os.getenv("CONVERSION_BATCH_SIZE", "100"))

# Redshift
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = int(os.getenv("REDSHIFT_PORT", "5439"))
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")
REDSHIFT_SCHEMA = os.getenv("REDSHIFT_SCHEMA", "public")
REDSHIFT_TABLE = os.getenv("REDSHIFT_TABLE", "openai_ads")

# Conversion events
CONVERSION_EVENT_IDS = [x.strip() for x in os.getenv("CONVERSION_EVENT_IDS", "").split(",") if x.strip()]
CONVERSION_EVENT_NAMES = [x.strip() for x in os.getenv("CONVERSION_EVENT_NAMES", "").split(",") if x.strip()]

if CONVERSION_EVENT_NAMES and len(CONVERSION_EVENT_NAMES) != len(CONVERSION_EVENT_IDS):
    raise ValueError("CONVERSION_EVENT_NAMES must have the same number of values as CONVERSION_EVENT_IDS.")


def get_event_column_name(event_id, index):
    """Turn a conversion event id/name into a Redshift-safe column name."""
    name = CONVERSION_EVENT_NAMES[index] if CONVERSION_EVENT_NAMES else f"conversion_{event_id}"
    return name.lower().strip().replace(" ", "_").replace("-", "_").replace(".", "_")


EVENT_COLUMN_MAP = {
    event_id: get_event_column_name(event_id, index)
    for index, event_id in enumerate(CONVERSION_EVENT_IDS)
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("openai_ads_pipeline")


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

def validate_environment():
    required = {
        "OPENAI_ADS_API_KEY": OPENAI_ADS_API_KEY,
        "REDSHIFT_HOST": REDSHIFT_HOST,
        "REDSHIFT_DATABASE": REDSHIFT_DATABASE,
        "REDSHIFT_USER": REDSHIFT_USER,
        "REDSHIFT_PASSWORD": REDSHIFT_PASSWORD,
    }

    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    if not CONVERSION_EVENT_IDS:
        logger.warning("No conversion event IDs configured.")


# ============================================================
# DATE RANGE
# ============================================================

def get_date_range():
    end_date = date.today()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH - 1)
    logger.info(f"Date range: {start_date} -> {end_date}")
    return start_date, end_date


# ============================================================
# API CLIENT
# ============================================================

class OpenAIAdsClient:

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def request(self, method, endpoint, params=None, json_body=None, max_retries=5):
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self.session.request(
                    method=method, url=url, params=params, json=json_body, timeout=90
                )

                if response.ok:
                    return response.json()

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = min(2 ** attempt, 60)
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                logger.error(f"API error {response.status_code}")
                logger.error(response.text)
                response.raise_for_status()

            except requests.RequestException as exc:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt, 60)
                logger.warning(f"Request failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"API request failed after {max_retries} attempts: {endpoint}")

    def get_account(self):
        return self.request("GET", "/ad_account")

    def paginated_get(self, endpoint, params=None, page_size=500):
        params = dict(params or {})
        params["limit"] = page_size

        all_rows = []
        after = None
        page_number = 0

        while True:
            page_number += 1
            request_params = dict(params)
            if after:
                request_params["after"] = after

            response = self.request("GET", endpoint, params=request_params)
            rows = response.get("data", [])
            all_rows.extend(rows)

            logger.info(f"{endpoint} | page={page_number} | rows={len(rows)} | total={len(all_rows)}")

            if not response.get("has_more", False):
                break

            after = response.get("last_id")
            if not after:
                logger.warning("has_more=True but last_id is missing.")
                break

        return all_rows

    def get_campaigns(self):
        return self.paginated_get("/campaigns", page_size=500)

    def get_ad_groups(self, campaign_id=None):
        params = {"campaign_id": campaign_id} if campaign_id else {}
        return self.paginated_get("/ad_groups", params=params, page_size=500)

    def get_ads(self, ad_group_id=None):
        params = {"ad_group_id": ad_group_id} if ad_group_id else {}
        return self.paginated_get("/ads", params=params, page_size=500)

    def get_ad_insights(self, start_date, end_date):
        time_range = json.dumps({
            "type": "date_range",
            "since": str(start_date),
            "until": str(end_date),
            "timezone": REPORTING_TIMEZONE,
        })

        fields = [
            "metadata.readable_time",
            "ad.id",
            "ad.name",
            "ad.impressions",
            "ad.clicks",
            "ad.spend",
        ]

        params = (
            [("time_granularity", "daily"), ("aggregation_level", "ad")]
            + [("fields[]", field) for field in fields]
            + [("time_ranges[]", time_range), ("limit", str(INSIGHTS_PAGE_SIZE))]
        )

        rows = []
        after = None
        page = 0

        while True:
            page += 1
            request_params = list(params)
            if after:
                request_params.append(("after", after))

            response = self.request("GET", "/ad_account/insights", params=request_params)
            page_rows = response.get("data", [])
            rows.extend(page_rows)

            logger.info(f"Ad insights | page={page} | rows={len(page_rows)} | total={len(rows)}")

            if not response.get("has_more", False):
                break

            after = response.get("last_id")
            if not after:
                raise RuntimeError("OpenAI returned has_more=true without last_id.")

        return rows

    def get_conversion_insights(self, ad_ids, start_date, end_date):
        if not ad_ids:
            return []

        time_range = json.dumps({
            "type": "date_range",
            "since": str(start_date),
            "until": str(end_date),
            "timezone": REPORTING_TIMEZONE,
        })

        all_rows = []

        # NOTE: OpenAI's conversion-insights API does not currently expose
        # event_setting_id as a request parameter, so the caller must supply
        # only ads belonging to campaigns configured for that event setting.
        for batch_start in range(0, len(ad_ids), CONVERSION_BATCH_SIZE):
            batch = ad_ids[batch_start: batch_start + CONVERSION_BATCH_SIZE]

            payload = {
                "aggregation_level": "ad",
                "time_granularity": "daily",
                "time_ranges": [time_range],
                "entity_ids": batch,
                "group_by_entity": True,
            }

            response = self.request("POST", "/conversions/insights", json_body=payload)
            rows = response.get("data", [])
            all_rows.extend(rows)

            logger.info(
                f"Conversion insights | batch={batch_start // CONVERSION_BATCH_SIZE + 1} | "
                f"ads={len(batch)} | rows={len(rows)}"
            )

        return all_rows


# ============================================================
# BUILD HIERARCHY
# ============================================================

def build_hierarchy(client):
    logger.info("Fetching campaign hierarchy...")

    campaigns = client.get_campaigns()
    logger.info(f"Campaigns found: {len(campaigns):,}")

    campaign_map, adgroup_map, ad_map = {}, {}, {}

    for campaign in campaigns:
        campaign_id = campaign.get("id")
        if not campaign_id:
            continue

        campaign_map[campaign_id] = {
            "campaign_id": campaign_id,
            "campaign_name": campaign.get("name", ""),
            "conversion_event_setting_ids": campaign.get("conversion_event_setting_ids") or [],
        }

    for campaign_id in campaign_map:
        adgroups = client.get_ad_groups(campaign_id)
        logger.info(f"Campaign {campaign_id}: {len(adgroups):,} ad groups")

        for adgroup in adgroups:
            adgroup_id = adgroup.get("id")
            if not adgroup_id:
                continue

            adgroup_map[adgroup_id] = {
                "adgroup_id": adgroup_id,
                "adgroup_name": adgroup.get("name", ""),
                "campaign_id": campaign_id,
            }

    for adgroup_id in adgroup_map:
        ads = client.get_ads(adgroup_id)
        logger.info(f"Ad group {adgroup_id}: {len(ads):,} ads")

        for ad in ads:
            ad_id = ad.get("id")
            if not ad_id:
                continue

            ad_map[ad_id] = {
                "ad_id": ad_id,
                "ad_name": ad.get("name", ""),
                "adgroup_id": adgroup_id,
            }

    logger.info(
        f"Hierarchy complete | campaigns={len(campaign_map):,} | "
        f"adgroups={len(adgroup_map):,} | ads={len(ad_map):,}"
    )

    return campaign_map, adgroup_map, ad_map


# ============================================================
# GET ADS BELONGING TO EVENT
# ============================================================

def get_event_ads(event_id, campaign_map, adgroup_map, ad_map):
    matching_campaigns = {
        campaign_id
        for campaign_id, campaign in campaign_map.items()
        if event_id in campaign.get("conversion_event_setting_ids", [])
    }

    matching_adgroups = {
        adgroup_id
        for adgroup_id, adgroup in adgroup_map.items()
        if adgroup["campaign_id"] in matching_campaigns
    }

    matching_ads = [
        ad_id for ad_id, ad in ad_map.items() if ad["adgroup_id"] in matching_adgroups
    ]

    logger.info(
        f"Event {event_id}: {len(matching_campaigns):,} campaigns | "
        f"{len(matching_adgroups):,} ad groups | {len(matching_ads):,} ads"
    )

    return matching_ads


# ============================================================
# BUILD DELIVERY DATAFRAME
# ============================================================

def build_delivery_dataframe(rows, account_id, account_name, ad_map, adgroup_map, campaign_map):
    output = []

    for row in rows:
        ad_id = row.get("ad_id")
        if not ad_id:
            continue

        ad = ad_map.get(ad_id, {})
        adgroup_id = ad.get("adgroup_id", "")
        adgroup = adgroup_map.get(adgroup_id, {})
        campaign_id = adgroup.get("campaign_id", "")
        campaign = campaign_map.get(campaign_id, {})

        output.append({
            "date": row.get("readable_time"),
            "account_id": account_id,
            "account_name": account_name,
            "campaign_id": campaign_id,
            "campaign_name": campaign.get("campaign_name", ""),
            "adgroup_id": adgroup_id,
            "adgroup_name": adgroup.get("adgroup_name", ""),
            "ad_id": ad_id,
            "ad_name": ad.get("ad_name", row.get("ad_name", "")),
            "impressions": row.get("impressions", 0),
            "clicks": row.get("clicks", 0),
            "spend": row.get("spend", 0),
        })

    return pd.DataFrame(output)


# ============================================================
# ADD CONVERSION DATA
# ============================================================

def add_conversion_columns(delivery_df, client, start_date, end_date, campaign_map, adgroup_map, ad_map):
    if delivery_df.empty:
        return delivery_df

    for event_id in CONVERSION_EVENT_IDS:
        delivery_df[EVENT_COLUMN_MAP[event_id]] = 0

    for event_id in CONVERSION_EVENT_IDS:
        logger.info(f"Processing conversion event: {event_id}")

        event_ads = get_event_ads(event_id, campaign_map, adgroup_map, ad_map)
        if not event_ads:
            logger.warning(f"No ads found for event setting {event_id}")
            continue

        conversion_rows = client.get_conversion_insights(event_ads, start_date, end_date)
        if not conversion_rows:
            logger.info(f"No conversion data returned for {event_id}")
            continue

        conversion_df = pd.DataFrame([
            {
                "date": row.get("date"),
                "ad_id": row.get("entity_id"),
                "conversions": row.get("conversions", 0),
            }
            for row in conversion_rows
        ])

        if conversion_df.empty:
            continue

        conversion_df["date"] = pd.to_datetime(conversion_df["date"], errors="coerce").dt.date
        conversion_df["conversions"] = pd.to_numeric(conversion_df["conversions"], errors="coerce").fillna(0)

        # Aggregate in case the API returns duplicate rows.
        conversion_df = conversion_df.groupby(["date", "ad_id"], as_index=False)["conversions"].sum()

        column_name = EVENT_COLUMN_MAP[event_id]
        delivery_df = delivery_df.merge(
            conversion_df.rename(columns={"conversions": column_name}),
            on=["date", "ad_id"],
            how="left",
        )
        delivery_df[column_name] = delivery_df[column_name].fillna(0)

    return delivery_df


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_dataframe(df):
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    numeric_columns = ["impressions", "clicks", "spend"] + list(EVENT_COLUMN_MAP.values())
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    string_columns = [
        "account_id", "account_name",
        "campaign_id", "campaign_name",
        "adgroup_id", "adgroup_name",
        "ad_id", "ad_name",
    ]
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    return df


# ============================================================
# QA
# ============================================================

def validate_dataframe(df):
    if df.empty:
        logger.warning("DataFrame is empty.")
        return False

    required = [
        "date",
        "account_id", "account_name",
        "campaign_id", "campaign_name",
        "adgroup_id", "adgroup_name",
        "ad_id", "ad_name",
        "impressions", "clicks", "spend",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    null_dates = df["date"].isna().sum()
    if null_dates:
        raise ValueError(f"{null_dates} rows have NULL dates.")

    grain = ["date", "account_id", "campaign_id", "adgroup_id", "ad_id"]
    duplicate_count = df.duplicated(subset=grain, keep=False).sum()
    if duplicate_count:
        logger.warning(f"Found {duplicate_count:,} duplicate rows at expected grain.")

    negative_spend = (df["spend"] < 0).sum()
    if negative_spend:
        logger.warning(f"{negative_spend:,} rows have negative spend.")

    logger.info(f"QA complete: {len(df):,} rows.")
    return True


# ============================================================
# REDSHIFT
# ============================================================

def get_redshift_connection():
    return psycopg2.connect(
        host=REDSHIFT_HOST,
        port=REDSHIFT_PORT,
        database=REDSHIFT_DATABASE,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASSWORD,
    )


def validate_identifier(value):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

    if not value:
        raise ValueError("Empty SQL identifier.")

    if not all(character in allowed for character in value):
        raise ValueError(f"Unsafe SQL identifier: {value}")

    return value


def create_redshift_table(conn):
    schema = validate_identifier(REDSHIFT_SCHEMA)
    table = validate_identifier(REDSHIFT_TABLE)

    conversion_columns = [
        f'"{validate_identifier(column)}" DOUBLE PRECISION'
        for column in EVENT_COLUMN_MAP.values()
    ]
    conversion_sql = (",\n        " + ",\n        ".join(conversion_columns)) if conversion_columns else ""

    sql = f"""
    CREATE TABLE IF NOT EXISTS {schema}.{table} (
        date DATE NOT NULL,
        account_id VARCHAR(255),
        account_name VARCHAR(1000),
        campaign_id VARCHAR(255),
        campaign_name VARCHAR(2000),
        adgroup_id VARCHAR(255),
        adgroup_name VARCHAR(2000),
        ad_id VARCHAR(255),
        ad_name VARCHAR(2000),
        impressions BIGINT,
        clicks BIGINT,
        spend DOUBLE PRECISION
        {conversion_sql}
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(sql)
    conn.commit()

    logger.info(f"Verified table {schema}.{table}")


def delete_existing_dates(conn, start_date, end_date):
    schema = validate_identifier(REDSHIFT_SCHEMA)
    table = validate_identifier(REDSHIFT_TABLE)

    sql = f"DELETE FROM {schema}.{table} WHERE date >= %s AND date <= %s"

    with conn.cursor() as cursor:
        cursor.execute(sql, (start_date, end_date))
        deleted = cursor.rowcount
    conn.commit()

    logger.info(f"Deleted {deleted:,} existing rows.")


def insert_dataframe(conn, df):
    if df.empty:
        logger.warning("Nothing to insert.")
        return

    schema = validate_identifier(REDSHIFT_SCHEMA)
    table = validate_identifier(REDSHIFT_TABLE)

    columns = [
        "date",
        "account_id", "account_name",
        "campaign_id", "campaign_name",
        "adgroup_id", "adgroup_name",
        "ad_id", "ad_name",
        "impressions", "clicks", "spend",
    ] + list(EVENT_COLUMN_MAP.values())

    columns = [validate_identifier(column) for column in columns if column in df.columns]

    values = [
        tuple(row_dict.get(column) for column in columns)
        for row_dict in (dict(zip(df.columns, row)) for row in df.itertuples(index=False, name=None))
    ]

    column_sql = ", ".join(f'"{column}"' for column in columns)
    sql = f"INSERT INTO {schema}.{table} ({column_sql}) VALUES %s"

    with conn.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=5000)
    conn.commit()

    logger.info(f"Inserted {len(values):,} rows.")


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):
    if df.empty:
        return

    print()
    print("=" * 75)
    print("OPENAI ADS → REDSHIFT")
    print("=" * 75)
    print(f"Rows:        {len(df):,}")
    print(f"Dates:       {df['date'].min()} → {df['date'].max()}")
    print(f"Campaigns:   {df['campaign_id'].nunique():,}")
    print(f"Ad Groups:   {df['adgroup_id'].nunique():,}")
    print(f"Ads:         {df['ad_id'].nunique():,}")
    print(f"Impressions: {df['impressions'].sum():,.0f}")
    print(f"Clicks:      {df['clicks'].sum():,.0f}")
    print(f"Spend:       {df['spend'].sum():,.2f}")

    for column in EVENT_COLUMN_MAP.values():
        if column in df.columns:
            print(f"{column}: {df[column].sum():,.0f}")

    print("=" * 75)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("Starting OpenAI Ads pipeline...")

    validate_environment()
    start_date, end_date = get_date_range()

    client = OpenAIAdsClient(OPENAI_ADS_API_KEY)

    logger.info("Getting ad account...")
    account = client.get_account()
    account_id = account.get("id", "")
    account_name = account.get("name", "")
    logger.info(f"Account: {account_name} ({account_id})")

    campaign_map, adgroup_map, ad_map = build_hierarchy(client)

    logger.info("Fetching daily ad-level insights...")
    insight_rows = client.get_ad_insights(start_date, end_date)
    logger.info(f"Delivery rows: {len(insight_rows):,}")

    df = build_delivery_dataframe(insight_rows, account_id, account_name, ad_map, adgroup_map, campaign_map)

    if df.empty:
        logger.warning("No delivery data found.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        if CONVERSION_EVENT_IDS:
            logger.info("Fetching conversion insights...")
            df = add_conversion_columns(df, client, start_date, end_date, campaign_map, adgroup_map, ad_map)

        df = clean_dataframe(df)
        validate_dataframe(df)

    print_summary(df)

    logger.info("Connecting to Redshift...")
    conn = get_redshift_connection()
    try:
        create_redshift_table(conn)
        delete_existing_dates(conn, start_date, end_date)
        insert_dataframe(conn, df)
    finally:
        conn.close()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"PIPELINE FAILED: {exc}")
        sys.exit(1)