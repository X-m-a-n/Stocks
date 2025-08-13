-- Part 2: Junior Market Companies
USE jse_database;

-- Insert Junior Market companies
INSERT INTO companies (symbol, company_name, isin, currency, sector_id, market_type_id, security_type, has_website) VALUES
-- Financial Services
('AFS', 'Access Financial Services Limited', 'JMP004681012', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 2, 'ORDINARY', TRUE),
('ISP', 'ISP Finance Services Limited', 'JME201600074', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 2, 'ORDINARY', TRUE),
('LASF', 'Lasco Financial Services Limited', 'JMP620981010', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 2, 'ORDINARY', TRUE),
('DOLLA', 'Dolla Financial Services Limited', 'JME202200080', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'FINANCE'), 2, 'ORDINARY', TRUE),

-- Manufacturing
('AMG', 'AMG Packaging & Paper Company Limited', 'JMP4171A1054', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('BPOW', 'Blue Power Group Limited', 'JMP170271069', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', FALSE),
('KREMI', 'Caribbean Cream Limited', 'JME201300014', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('CFF', 'Caribbean Flavours & Fragrances Limited', 'JME201300063', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('PURITY', 'Consolidated Bakeries (Jamaica) Limited', 'JME201200024', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('HONBUN', 'Honey Bun (1982) Limited', 'JMP5178T1046', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('JAMT', 'Jamaican Teas Limited', 'JMP5894D1021', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('OMNI', 'Omni Industries Limited', 'JME202400037', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),
('PTL', 'Paramount Trading (Jamaica) Limited', 'JME201200032', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'MANUFACTURING'), 2, 'ORDINARY', TRUE),

-- Retail & Distribution
('CAC', 'CAC 2000 Limited', 'JME201500084', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', TRUE),
('DTL', 'Derrimon Trading Company Limited', 'JME201300147', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', TRUE),
('FTNA', 'Fontana Limited', 'JME201900011', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', TRUE),
('INDIES', 'Indies Pharma Jamaica Limited', 'JME201800161', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', FALSE),
('JETCON', 'Jetcon Corporation Limited', 'JME201600082', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', TRUE),
('MDS', 'Medical Disposables & Supplies Limited', 'JME201300154', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', FALSE),
('RPL', 'Regency Petroleum Co. Limited', 'JME202200288', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'RETAIL'), 2, 'ORDINARY', FALSE),

-- Technology & Services
('LEARN', 'Edufocal Limited', 'JME202200031', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'TECHNOLOGY'), 2, 'ORDINARY', FALSE),
('TTECH', 'tTech Limited', 'JME201600017', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'TECHNOLOGY'), 2, 'ORDINARY', TRUE),
('LAB', 'The Limners and Bards Limited', 'JME201900102', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'TECHNOLOGY'), 2, 'ORDINARY', TRUE),

-- Other Junior Market Companies
('CABROKERS', 'Caribbean Assurance Brokers Limited', 'JME202000035', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('CHL', 'Cargo Handlers Limited', 'JMP210381076', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('EFRESH', 'Everything Fresh Limited', 'JME201800146', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('ECL', 'Express Catering Limited', 'JME201700130', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('FOSRICH', 'Fosrich Company Limited', 'JME201700197', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', TRUE),
('GWEST', 'GWest Corporation Limited', 'JME201700213', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('MAILPAC', 'Mailpac Group Limited', 'JME201900235', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('MEEG', 'Main Event Entertainment Group', 'JME201700023', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', TRUE),
('SOS', 'Stationery and Office Supplies Limited', 'JME201700148', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE),
('SPURTREE', 'Spur Tree Spices Jamaica Limited', 'JME202200015', 'JMD', (SELECT sector_id FROM sectors WHERE sector_name = 'OTHER'), 2, 'ORDINARY', FALSE);