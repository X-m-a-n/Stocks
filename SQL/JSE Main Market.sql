-- Part 1: Database Schema and Main Market Companies
CREATE DATABASE IF NOT EXISTS jse_database;
USE jse_database;

-- Create sectors table
CREATE TABLE sectors (
    sector_id INT AUTO_INCREMENT PRIMARY KEY,
    sector_name VARCHAR(50) NOT NULL UNIQUE
);

-- Create market types table
CREATE TABLE market_types (
    market_type_id INT AUTO_INCREMENT PRIMARY KEY,
    market_name VARCHAR(20) NOT NULL UNIQUE
);

-- Create companies table
CREATE TABLE companies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    company_name VARCHAR(100) NOT NULL,
    isin VARCHAR(12) NOT NULL UNIQUE,
    currency VARCHAR(3) NOT NULL,
    sector_id INT,
    market_type_id INT,
    security_type ENUM('ORDINARY', 'PREFERENCE') NOT NULL,
    has_website BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id),
    FOREIGN KEY (market_type_id) REFERENCES market_types(market_type_id)
);

-- Insert market types
INSERT INTO market_types (market_name) VALUES
('Main Market'),
('Junior Market');

-- Insert sectors
INSERT INTO sectors (sector_name) VALUES
('FINANCE'),
('MANUFACTURING'),
('RETAIL'),
('INSURANCE'),
('COMMUNICATIONS'),
('CONGLOMERATES'),
('TOURISM'),
('TECHNOLOGY'),
('OTHER');

-- Insert Main Market companies
INSERT INTO companies (symbol, company_name, isin, currency, sector_id, market_type_id, security_type, has_website) VALUES
-- Investment & Finance
('138SL', '138 Student Living Jamaica Limited', 'JME201400061', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', FALSE),
('BIL', 'Barita Investments Limited', 'JMP161681094', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),
('CPFV', 'Eppley Caribbean Property Fund Limited SCC', 'JME201900136', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),
('EPLY', 'Eppley Limited', 'JM0170271079', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),
('JSE', 'Jamaica Stock Exchange', 'JME201300022', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),
('JMMBGL', 'JMMB Group Limited', 'JME201400020', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', FALSE),
('NCBFG', 'NCB Financial Group Limited', 'JME201700064', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),
('PROVEN', 'Proven Group Limited', 'JMP792231061', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', FALSE),
('SGJ', 'Scotia Group Jamaica Limited', 'JMP8537F1042', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 1, 'ORDINARY', TRUE),

-- Manufacturing
('ASBH', 'A.S. Bryden & Sons Holdings Limited', 'JME202300088', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', FALSE),
('BRG', 'Berger Paints Jamaica Ltd.', 'JMP165551079', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),
('CCC', 'Caribbean Cement Company Ltd.', 'JMP210961059', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),
('LASM', 'Lasco Manufacturing Limited', 'JMP620991092', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),
('SALF', 'Salada Foods Jamaica Ltd.', 'JMP831721098', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),
('SEP', 'Seprod Limited', 'JMP8579T1037', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),
('WISYNCO', 'Wisynco Group Ltd', 'JME201700221', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 1, 'ORDINARY', TRUE),

-- Conglomerates
('GK', 'GraceKennedy Limited', 'JMP4897P1050', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'CONGLOMERATES'), 1, 'ORDINARY', TRUE),
('JP', 'Jamaica Producers Group Ltd.', 'JMP589351007', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'CONGLOMERATES'), 1, 'ORDINARY', TRUE),
('PJAM', 'Pan Jamaica Group Limited', 'JMP749981081', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'CONGLOMERATES'), 1, 'ORDINARY', TRUE),
('SJ', 'Sagicor Group Jamaica Limited', 'JME201300139', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'CONGLOMERATES'), 1, 'ORDINARY', TRUE);

-- Create indexes for better query performance
CREATE INDEX idx_company_symbol ON companies(symbol);
CREATE INDEX idx_company_sector ON companies(sector_id);
CREATE INDEX idx_company_market ON companies(market_type_id);