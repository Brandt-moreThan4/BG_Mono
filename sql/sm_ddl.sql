-- Create schema
CREATE SCHEMA IF NOT EXISTS security_master;

-- Cleanup (drop tables in dependency order)
DROP TABLE IF EXISTS security_master.trading_item_metric;
DROP TABLE IF EXISTS security_master.security_attribute;
DROP TABLE IF EXISTS security_master.identifier_map;
DROP TABLE IF EXISTS security_master.trading_item;
DROP TABLE IF EXISTS security_master.security;
DROP TABLE IF EXISTS security_master.company_attribute;
DROP TABLE IF EXISTS security_master.company;

-- Table creation

CREATE TABLE security_master.company (
    company_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE security_master.company_attribute (
    company_id INTEGER NOT NULL,
    attribute_from_date DATE NOT NULL,
    attribute_to_date DATE,
    attribute_name TEXT NOT NULL,
    attribute_value TEXT,
    PRIMARY KEY (company_id, attribute_from_date, attribute_name),
    FOREIGN KEY (company_id) REFERENCES security_master.company(company_id)
);

CREATE TABLE security_master.security (
    security_id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    security_type TEXT,
    cusip TEXT,
    isin TEXT,
    share_class_figi TEXT,
    underlying_security_id INTEGER,
    FOREIGN KEY (company_id) REFERENCES security_master.company(company_id),
    FOREIGN KEY (underlying_security_id) REFERENCES security_master.security(security_id)
);

CREATE TABLE security_master.security_attribute (
    security_id INTEGER NOT NULL,
    attribute_from_date DATE NOT NULL,
    attribute_to_date DATE,
    attribute_name TEXT NOT NULL,
    attribute_value TEXT,
    PRIMARY KEY (security_id, attribute_from_date, attribute_name),
    FOREIGN KEY (security_id) REFERENCES security_master.security(security_id)
);

CREATE TABLE security_master.trading_item (
    trading_item_id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (security_id) REFERENCES security_master.security(security_id)
);

CREATE TABLE security_master.trading_item_metric (
    trading_item_id INTEGER NOT NULL,
    metric_date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value TEXT,
    PRIMARY KEY (trading_item_id, metric_date, metric_name),
    FOREIGN KEY (trading_item_id) REFERENCES security_master.trading_item(trading_item_id)
);

CREATE TABLE security_master.identifier_map (
    entity_type TEXT NOT NULL,  -- 'company', 'security', 'trading_item'
    entity_id INTEGER NOT NULL,
    identifier_type TEXT NOT NULL,  -- 'ticker', 'ISIN', 'CUSIP', etc.
    identifier_value TEXT NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE,
    PRIMARY KEY (entity_type, entity_id, identifier_type, valid_from)
);

-- Verification (list tables in security_master schema)
-- You can run this separately to check
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'security_master' ORDER BY table_name;
