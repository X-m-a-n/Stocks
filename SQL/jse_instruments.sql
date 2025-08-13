-- JSE Instruments Data Import Script
-- Creates and populates the instruments table in jse_database

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS jse_database;
USE jse_database;

-- Drop table if exists to ensure clean import
DROP TABLE IF EXISTS instruments;

-- Create the instruments table
CREATE TABLE instruments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    security_name VARCHAR(255) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    isin VARCHAR(50),
    currency CHAR(3) NOT NULL,
    sector VARCHAR(100),
    type VARCHAR(50) NOT NULL,
    market VARCHAR(50) NOT NULL,
    jse_link TEXT,
    website_url TEXT,
    has_website CHAR(1) DEFAULT 'N',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Add indexes for performance
    INDEX idx_symbol (symbol),
    INDEX idx_currency (currency),
    INDEX idx_sector (sector),
    INDEX idx_type (type),
    INDEX idx_market (market)
);

-- Insert all instrument data
INSERT INTO instruments (security_name, symbol, isin, currency, sector, type, market, jse_link, website_url, has_website) VALUES
('138 STUDENT LIVING JAMAICA LIMITED', '138SL', 'JME201400061', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=311', '', 'N'),
('138 STUDENT LIVING JAMAICA LIMITED VARIABLE PREFERENCE', '138SLVR', 'JME201400079', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=313', '', 'N'),
('A.S. BRYDEN & SONS HOLDINGS LIMITED', 'ASBH', 'JME202300088', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600148', '', 'N'),
('BARITA INVESTMENTS LIMITED', 'BIL', 'JMP161681094', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=21', 'http://www.barita.com/', 'Y'),
('BERGER PAINTS JAMAICA LTD.', 'BRG', 'JMP165551079', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=23', 'http://www.bergeronline.com/caribbean', 'Y'),
('CARIBBEAN CEMENT COMPANY LTD.', 'CCC', 'JMP210961059', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=37', 'http://www.caribcement.com/', 'Y'),
('CARIBBEAN PRODUCERS JAMAICA LIMITED', 'CPJ', 'JMP2115K1063', 'JMD', 'RETAIL', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=256', 'http://www.caribbeanproducers.com/', 'Y'),
('CARRERAS LIMITED', 'CAR', 'JMP213891048', 'JMD', 'RETAIL', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=39', 'http://www.carrerasltd.com/', 'Y'),
('EPPLEY CARIBBEAN PROPERTY FUND LIMITED SCC', 'CPFV', 'JME201900136', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600091', 'https://eppleylimited.com/eppley-caribbean-property-fund/', 'Y'),
('EPPLEY LIMITED', 'EPLY', 'JM0170271079', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=279', 'http://www.eppleylimited.com/', 'Y'),
('EPPLEY LIMITED 7.25% PREFERENCE SHARES', 'EPLY7.25', 'JMC202100490', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600116', 'https://www.eppleylimited.com/', 'Y'),
('EPPLEY LIMITED 7.75% PREFERENCE SHARES', 'EPLY7.75', 'JMC202100508', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600117', 'https://www.eppleylimited.com/', 'Y'),
('FIRST ROCK REAL ESTATE INVESTMENTS LIMITED', 'FIRSTROCKJMD', 'JME202000019', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600097', 'https://firstrock.com/', 'Y'),
('GENERAL ACCIDENT INSURANCE COMPANY (JA) LIMITED', 'GENAC', 'JMP4775P1024', 'JMD', 'INSURANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=258', 'http://www.genac.com/', 'Y'),
('GRACEKENNEDY LIMITED', 'GK', 'JMP4897P1050', 'JMD', 'CONGLOMERATES', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=55', 'http://www.gracekennedy.com/', 'Y'),
('GUARDIAN HOLDINGS LIMITED', 'GHL', 'TTP501551035', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=57', '', 'N'),
('INNOVATIVE ENERGY GROUP LIMITED', 'ENERGY', 'JMP283451095', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=41', 'https://innovativeenergy.net/', 'Y'),
('JAMAICA BROILERS GROUP', 'JBG', 'JMP5892N1021', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=65', 'http://www.jamaicabroilersgroup.com/', 'Y'),
('JAMAICA PRODUCERS GROUP LTD.', 'JP', 'JMP589351007', 'JMD', 'CONGLOMERATES', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=75', 'http://www.jpjamaica.com/', 'Y'),
('JAMAICA PUBLIC SERVICE 5% C', 'JPS5C', 'JMP589361246', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=77', 'http://www.jpsco.com/', 'Y'),
('JAMAICA PUBLIC SERVICE 5% D', 'JPS5D', 'JMP589361402', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=79', 'http://www.jpsco.com/', 'Y'),
('JAMAICA PUBLIC SERVICE CO. 5%', 'JPS5', 'JA5893612461', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=700550', '', 'N'),
('JAMAICA PUBLIC SERVICE CO. 6%', 'JPS6', 'JMP589361329', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=81', 'http://www.jpsco.com/', 'Y'),
('JAMAICA PUBLIC SERVICE CO. 9.5%', 'JPS9.5', 'JME201300097', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=288', 'http://www.jpsco.com/', 'Y'),
('JAMAICA PUBLIC SERVICE CO. LTD. 7%', 'JPS7', 'JMP589361162', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=83', 'http://www.jpsco.com/', 'Y'),
('JAMAICA STOCK EXCHANGE', 'JSE', 'JME201300022', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=276', 'http://www.jamstockex.com/', 'Y'),
('JMMB GROUP LIMITED', 'JMMBGL', 'JME201400020', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=319', '', 'N'),
('JMMB GROUP LIMITED 7.15% CUMULATIVE REDEEMABLE PREFERENCE SHARE', 'JMMBGL7.15', 'JME202100041', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600108', 'https://jm.jmmb.com/', 'Y'),
('JMMB GROUP LIMITED 7.35% CUMULATIVE REDEEMABLE PREFERENCE SHARE', 'JMMBGL7.35', 'JME202100033', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600109', 'https://jm.jmmb.com/', 'Y'),
('JMMB GROUP LIMITED 9.50%', 'JMMBGL9.50', 'JME202400029', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=329', 'https://jm.jmmb.com/', 'Y'),
('JMMB GROUP LTD 10.00% VR JMD PREFERENCE SHARES', 'JMMBGL10.00', 'JME202500042', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600059', 'https://jm.jmmb.com/', 'Y'),
('KEY INSURANCE COMPANY LIMITED', 'KEY', 'JME201600090', 'JMD', 'INSURANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=336', 'https://keyinsurancejm.com/dashboards', 'Y'),
('KINGSTON PROPERTIES LIMITED', 'KPREIT', 'JMP607311090', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=89', 'http://www.kingstonpropertiesreit.com/', 'Y'),
('KINGSTON WHARVES LIMITED', 'KW', 'JMP607321073', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=91', 'http://www.kingstonwharves.com.jm/', 'Y'),
('LASCO DISTRIBUTORS LIMITED', 'LASD', 'JMP6209P1033', 'JMD', 'RETAIL', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=99', 'http://www.lascodistributors.com/', 'Y'),
('LASCO MANUFACTURING LIMITED', 'LASM', 'JMP620991092', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=103', 'http://www.lascojamaica.com/foodManufacturing.asp', 'Y'),
('MARGARITAVILLE (TURKS) LIMITED', 'MTL', 'JME201400012', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600081', '', 'N'),
('MASSY HOLDINGS LTD', 'MASSY', 'TTP710771044', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600126', 'https://massygroup.com/', 'Y'),
('MAYBERRY GROUP LTD.', 'MGL', 'JME202300187', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600150', 'https://www.mayberryinv.com/', 'Y'),
('MAYBERRY JAMAICAN EQUITIES LIMITED', 'MJE', 'JME201800153', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600072', 'https://www.mayberryinv.com/', 'Y'),
('MPC CARIBBEAN CLEAN ENERGY LIMITED', 'MPCCEL', 'TTE000000116', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600078', 'https://www.mpc-cleanenergy.com/', 'Y'),
('NCB FINANCIAL GROUP LIMITED', 'NCBFG', 'JME201700064', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600012', 'https://www.myncb.com/', 'Y'),
('PALACE AMUSEMENT CO. LTD.', 'PAL', 'JMP744941080', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=113', 'http://www.palaceamusement.com/', 'Y'),
('PAN JAMAICA GROUP LIMITED', 'PJAM', 'JMP749981081', 'JMD', 'CONGLOMERATES', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=121', 'http://www.panjam.com/', 'Y'),
('PORTLAND JSX LIMITED', 'PJX', 'JME201500068', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=337', 'https://www.portlandjsx.com/', 'Y'),
('PRODUCTIVE BUSINESS SOLUTIONS LTD 10.50% PERPETUAL CUMULATIVE REDEEMABLE J$ PREFERENCE SHARE', 'PBS10.50', 'JME202200239', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600132', '', 'N'),
('PROVEN GROUP LIMITED', 'PROVEN', 'JMP792231061', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=321', '', 'N'),
('PULSE INVESTMENTS LIMITED', 'PULS', 'JMP792921018', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=125', 'http://www.caribbeanfashionweek.com/pulse_profile.html', 'Y'),
('QWI INVESTMENTS LIMITED', 'QWI', 'JME201900151', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600092', 'https://qwiinvestments.com/', 'Y'),
('RADIO JAMAICA LIMITED', 'RJR', 'JMP798411055', 'JMD', 'COMMUNICATIONS', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=127', 'http://www.radiojamaica.com/', 'Y'),
('SAGICOR GROUP JAMAICA LIMITED', 'SJ', 'JME201300139', 'JMD', 'CONGLOMERATES', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=297', 'http://www.sagicorjamaica.com/', 'Y'),
('SAGICOR REAL ESTATE X FUND LTD.', 'XFUND', 'JME201300105', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=290', 'http://www.sagicorjamaica.com/', 'Y'),
('SAGICOR SELECT FUNDS LIMITED - FINANCIAL', 'SELECTF', 'JME201900094', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600090', 'https://www.sagicor.com/', 'Y'),
('SAGICOR SELECT FUNDS LIMITED MANUFACTURING & DISTRIBUTION', 'SELECTMD', 'JME201900219', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600096', 'https://www.sagicor.com/', 'Y'),
('SALADA FOODS JAMAICA LTD.', 'SALF', 'JMP831721098', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=131', 'http://www.saladafoodsja.com/', 'Y'),
('SCOTIA GROUP JAMAICA LIMITED', 'SGJ', 'JMP8537F1042', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=133', 'http://www.scotiabank.com.jm/', 'Y'),
('SEPROD LIMITED', 'SEP', 'JMP8579T1037', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=137', 'http://www.seprod.com/', 'Y'),
('STANLEY MOTTA LIMITED ORDINARY SHARES', 'SML', 'JME201800179', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600074', 'http://www.stanleymotta.com/', 'Y'),
('STERLING INVESTMENTS LIMITED', 'SIL', 'JME201400046', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=309', 'http://www.sterlingasset.net/nmcms.php?snippet=services&p=service_details&id=27', 'Y'),
('SUPREME VENTURES LIMITED', 'SVL', 'JMP8805B1082', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=139', 'http://www.supremeventures.com/', 'Y'),
('SYGNUS CREDIT INVESTMENTS 10% PREFERENCE SHARES - CLASS H', 'SCIJMD10.00H', 'JME202500018', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600162', 'https://sygnusgroup.com/alternative-investment-platform/sygnus-credit-investments/', 'Y'),
('SYGNUS CREDIT INVESTMENTS 10.50% PREFERENCE SHARES DUE 2025 - CLASS C', 'SCIJMD10.50C', 'JME202300120', 'JMD', 'FINANCE', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600151', '', 'N'),
('SYGNUS CREDIT INVESTMENTS LIMITED JMD (SCIJMD) ORDINARY SHARES', 'SCIJMD', 'JME201700106', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600069', 'https://sygnusgroup.com/', 'Y'),
('SYGNUS REAL ESTATE FINANCIAL LIMITED (JMD)', 'SRFJMD', 'JME201900110', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600118', 'https://sygnusgroup.com/', 'Y'),
('TRANSJAMAICAN HIGHWAY LIMITED', 'TJH', 'JME202000068', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600100', 'https://www.h2k-jio.com/', 'Y'),
('TRANSJAMAICAN HIGHWAY LTD 8%', 'TJH8.0', 'JME202000084', 'JMD', 'OTHER', 'PREFERENCE', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600105', 'https://www.h2k-jio.com/', 'Y'),
('VM INVESTMENTS LIMITED', 'VMIL', 'JME201700205', 'JMD', 'FINANCE', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600034', 'https://vmil.myvmgroup.com/', 'Y'),
('WIGTON ENERGY LIMITED', 'WIG', 'JME201900086', 'JMD', 'OTHER', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600087', 'https://www.wwfja.com/', 'Y'),
('WISYNCO GROUP LTD ORDINARY SHARES', 'WISYNCO', 'JME201700221', 'JMD', 'MANUFACTURING', 'ORDINARY', 'MAIN', 'https://www.jamstockex.com/trading/instruments/?instrument=1600033', 'https://wisynco.com/', 'Y'),
('ACCESS FINANCIAL SERVICES LIMITED', 'AFS', 'JMP004681012', 'JMD', 'FINANCE', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=13', 'http://www.accessfinanceonline.com/', 'Y'),
('AMG PACKAGING & PAPER COMPANY LIMITED', 'AMG', 'JMP4171A1054', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=254', 'http://www.amgpackaging.com/', 'Y'),
('ATLANTIC HARDWARE AND PLUMBING COMPANY LIMITED', 'AHPC', 'JME202500075', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600164', '', 'N'),
('BLUE POWER GROUP LIMITED', 'BPOW', 'JMP170271069', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=25', '', 'N'),
('CAC 2000 LIMITED', 'CAC', 'JME201500084', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=323', 'http://cacjamaica.com/', 'Y'),
('CARGO HANDLERS LIMITED', 'CHL', 'JMP210381076', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=35', '', 'N'),
('CARIBBBEAN ASSURANCE BROKERS LIMITED', 'CABROKERS', 'JME202000035', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600099', '', 'N'),
('CARIBBEAN CREAM LIMITED', 'KREMI', 'JME201300014', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=274', 'http://www.caribcream.com/', 'Y'),
('CARIBBEAN FLAVOURS & FRAGRANCES LIMITED', 'CFF', 'JME201300063', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=285', 'http://caribbeanflavoursjm.com/', 'Y'),
('CONSOLIDATED BAKERIES (JAMAICA) LIMITED', 'PURITY', 'JME201200024', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=270', 'http://www.purityjamaica.com/', 'Y'),
('DERRIMON TRADING COMPANY LIMITED', 'DTL', 'JME201300147', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=295', 'http://www.derrimon.com/', 'Y'),
('DOLLA FINANCIAL SERVICES LIMITED', 'DOLLA', 'JME202200080', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600130', 'https://dollafinancial.com/', 'Y'),
('DOLPHIN COVE LIMITED', 'DCOVE', 'JMP360321039', 'JMD', 'TOURISM', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=45', 'http://www.dolphincovejamaica.com/', 'Y'),
('EDUFOCAL LIMITED (SUSPENDED)', 'LEARN', 'JME202200031', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600128', '', 'N'),
('ELITE DIAGNOSTIC LIMITED', 'ELITE', 'JME201800047', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600035', 'https://elite-diagnostic.com/', 'Y'),
('EVERYTHING FRESH LIMITED', 'EFRESH', 'JME201800146', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600071', '', 'N'),
('EXPRESS CATERING LIMITED', 'ECL', 'JME201700130', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600016', '', 'N'),
('FONTANA LIMITED', 'FTNA', 'JME201900011', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600077', 'https://fontanapharmacy.com/', 'Y'),
('FOSRICH COMPANY LIMITED', 'FOSRICH', 'JME201700197', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600029', 'http://fosrich.com/shop/', 'Y'),
('FUTURE ENERGY SOURCE COMPANY LTD ORDINARY SHARES', 'FESCO', 'JME202100074', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600110', '', 'N'),
('GWEST CORPORATION LIMITED ORDINARY SHARES', 'GWEST', 'JME201700213', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600030', '', 'N'),
('HONEY BUN (1982) LIMITED', 'HONBUN', 'JMP5178T1046', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=61', 'http://www.honeybunja.com/', 'Y'),
('IMAGE PLUS CONSULTANTS LIMITED', 'IPCL', 'JME202200296', 'JMD', '', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600138', 'https://apex-radiology.com/', 'Y'),
('INDIES PHARMA JAMAICA LIMITED ORDINARY SHARES', 'INDIES', 'JME201800161', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600073', '', 'N'),
('IRONROCK INSURANCE COMPANY LIMITED', 'ROC', 'JME201600066', 'JMD', 'INSURANCE', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=333', 'https://www.ironrockjamaica.com/', 'Y'),
('ISP FINANCE SERVICES LIMITED', 'ISP', 'JME201600074', 'JMD', 'FINANCE', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=335', 'https://ispfinanceservices.com/', 'Y'),
('JAMAICAN TEAS LIMITED', 'JAMT', 'JMP5894D1021', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=87', 'http://www.jamaicanteas.com/', 'Y'),
('JETCON CORPORATION LIMITED', 'JETCON', 'JME201600082', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=334', 'https://jetconcars.com/', 'Y'),
('JFP LIMITED', 'JFP', 'JME202200023', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600127', '', 'N'),
('K.L.E. GROUP LIMITED', 'KLE', 'JME201200016', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=268', 'http://www.klegroupltd.com/', 'Y'),
('KINTYRE HOLDINGS (JA) LIMITED', 'KNTYR', 'JME201900052', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600082', 'http://icreateedu.com/', 'Y'),
('KNUTSFORD EXPRESS SERVICES LIMITED', 'KEX', 'JME201300170', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=301', 'http://www.knutsfordexpress.com/', 'Y'),
('LASCO FINANCIAL SERVICES LIMITED', 'LASF', 'JMP620981010', 'JMD', 'FINANCE', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=101', 'http://www.lascojamaica.com/financial.asp', 'Y'),
('LUMBER DEPOT LIMITED', 'LUMBER', 'JME201900243', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600095', '', 'N'),
('MAILPAC GROUP LIMITED', 'MAILPAC', 'JME201900235', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600094', '', 'N'),
('MAIN EVENT ENTERTAINMENT GROUP', 'MEEG', 'JME201700023', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600010', 'https://www.maineventjamaica.com/', 'Y'),
('MEDICAL DISPOSABLES & SUPPLIES LIMITED', 'MDS', 'JME201300154', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=299', '', 'N'),
('MFS CAPITAL PARTNERS LIMITED', 'MFS', 'JME201900029', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600075', 'https://mfscapltd.com/', 'Y'),
('OMNI INDUSTRIES LIMITED', 'OMNI', 'JME202400037', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600156', 'https://omniindustriesltd.org/', 'Y'),
('ONE GREAT STUDIO COMPANY LIMITED', '1GS', 'JME202300096', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600147', 'https://www.jamstockex.com/listings/listed-companies/www.onegreatstudio.com', 'Y'),
('ONE ON ONE EDUCATIONAL SERVICES LIMITED', 'ONE', 'JME202200106', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600131', 'https://www.oneononelms.com/', 'Y'),
('PARAMOUNT TRADING (JAMAICA) LIMITED', 'PTL', 'JME201200032', 'JMD', 'MANUFACTURING', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=272', 'http://www.paramountjm.com/', 'Y'),
('R.A. WILLIAMS DISTRIBUTORS LIMITED', 'RAWILL', 'JME202400078', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600160', 'https://rawilliamsltd.com/', 'Y'),
('REGENCY PETROLEUM CO. LIMITED', 'RPL', 'JME202200288', 'JMD', 'RETAIL', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600136', '', 'N'),
('SPUR TREE SPICES JAMAICA LIMITED', 'SPURTREE', 'JME202200015', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600125', '', 'N'),
('STATIONERY AND OFFICE SUPPLIES LIMITED', 'SOS', 'JME201700148', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600017', '', 'N'),
('THE LIMNERS AND BARDS LIMITED', 'LAB', 'JME201900102', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600089', 'https://www.thelabjamaica.com/', 'Y'),
('TROPICAL BATTERY COMPANY LIMITED', 'TROPICAL', 'JME202000092', 'JMD', 'OTHER', 'ORDINARY', 'JUNIOR', 'https://www.jamstockex.com/trading/instruments/?instrument=1600104', '', 'N'),
('A.S. BRYDEN & SONS HOLDINGS 6.00% CLASS A PREFERENCE SHARE', 'ASBH6.00', 'JME202300112', 'USD', 'MANUFACTURING', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600149', '', 'N'),
('A.S. BRYDEN & SONS HOLDINGS LIMITED USD', 'ASBH', 'JME202300088', 'USD', 'MANUFACTURING', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600154', '', 'N'),
('FIRST ROCK REAL ESTATE INVESTMENTS LIMITED (USD)', 'FIRSTROCKUSD', 'JME202000027', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600098', 'https://firstrock.com/', 'Y'),
('JMMB 6.0% USD PREFERENCE SHARES', 'JMMBUS6.00', 'JME201800211', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600084', '', 'N'),
('JMMB GROUP 5.75% FR USD CR PREFERENCE SHARES', 'JMMB5.75C', 'JME201800070', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600062', '', 'N'),
('JMMB GROUP LIMITED 8.50% USD', 'JMMBGLUSD8.50', 'JME202400011', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=327', 'https://jm.jmmb.com/', 'Y'),
('JMMB GROUP LTD 7.50% FR USD PREFERENCE SHARES', 'JMMBGLUS7.50', 'JME202500034', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600057', 'https://jm.jmmb.com/', 'Y'),
('MARGARITAVILLE (TURKS) LIMITED USD', 'MTL', 'JME201400012', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=303', 'http://www.margaritavillecaribbean.com/', 'Y'),
('MPC CARIBBEAN CLEAN ENERGY LIMITED USD', 'MPCCEL', 'TTE000000116', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600079', 'https://www.mpc-cleanenergy.com/', 'Y'),
('PRODUCTIVE BUSINESS SOLUTION LTD USD ORDINARY SHARES', 'PBS', 'JME201700155', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600018', 'https://www.grouppbs.com/', 'Y'),
('PRODUCTIVE BUSINESS SOLUTIONS LTD. 9.25 % PERPETUAL CUM (USD)', 'PBS9.25', 'JME202200221', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600133', '', 'N'),
('PROVEN GROUP LIMITED USD', 'PROVEN', 'JMP792231061', 'USD', 'FINANCE', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=252', '', 'N'),
('STERLING INVESTMENTS LIMITED USD', 'SIL', 'JME201400046', 'USD', 'FINANCE', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=332', 'http://www.sterlingasset.net/nmcms.php?snippet=services&p=service_details&id=27&menu_id=1', 'Y'),
('SYGNUS CREDIT INVESTMENTS 8% PREFERENCE SHARES - CLASS I', 'SCIUSD8.00I', 'JME202500026', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600163', 'https://sygnusgroup.com/alternative-investment-platform/sygnus-credit-investments/', 'Y'),
('SYGNUS CREDIT INVESTMENTS 8.00% PREFERENCE SHARES DUE 2025 - CLASS D', 'SCIUSD8.00D', 'JME202300138', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600152', '', 'N'),
('SYGNUS CREDIT INVESTMENTS 8.50% PREFERENCE SHARES DUE 2026 - CLASS E', 'SCIUSD8.50E', 'JME202300146', 'USD', 'FINANCE', 'PREFERENCE', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600153', '', 'N'),
('SYGNUS CREDIT INVESTMENTS LIMITED USD (SCIUSD) ORDINARY SHARES', 'SCIUSD', 'JME201700114', 'USD', 'FINANCE', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600067', 'https://sygnusgroup.com/', 'Y'),
('SYGNUS REAL ESTATE FINANCE LIMITED (USD)', 'SRFUSD', 'JME201900128', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600119', 'https://sygnusgroup.com/', 'Y'),
('TRANSJAMAICAN HIGHWAY LIMITED (USD)', 'TJH', 'JME202000068', 'USD', 'OTHER', 'ORDINARY', 'USD', 'https://www.jamstockex.com/trading/instruments/?instrument=1600101', 'https://www.h2k-jio.com/', 'Y'),
('BARITA INVESTMENTS LIMITED 10.00% FR UNSECURED BOND DUE JUNE 2028 - TRANCHE IV', 'BIL10FR2028T4', 'JMC202500434', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600172', '', 'N'),
('BARITA INVESTMENTS LIMITED 10.50% FR UNSECURED BOND DUE JUNE 2030 -TRANCHE V', 'BIL10.5FR2030T5', 'JMC202500442', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600173', '', 'N'),
('BARITA INVESTMENTS LIMITED 10.75% FR UNSECURED BOND DUE JUNE 2032 - TRANCHE VI', 'BIL10.75F2032T6', 'JMC202500459', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600174', '', 'N'),
('BARITA INVESTMENTS LIMITED 10.90% FR UNSECURED BOND DUE JUNE 2035 - TRANCHE VII', 'BIL10.9FR2035T7', 'JMC202500467', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600175', '', 'N'),
('BARITA INVESTMENTS LIMITED 9.75% FR UNSECURED BOND DUE JUNE 2027 - TRANCHE III', 'BIL9.75FR2027T3', 'JMC202500426', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600171', '', 'N'),
('MAYBERRY INVESTMENTS LIMITED $2.06B 10.75% FR SECURED BOND DUE MARCH 2026', 'MIL10.75FR2026', 'JMC202400916', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600161', 'https://www.jamstockex.com/listings/listed-companies/www.mayberryinv.com', 'Y'),
('MAYBERRY INVESTMENTS LIMITED $3.456B 10.25% FR SECURED BOND DUE MARCH 2027', 'MIL10.25FR2027', 'JMC202500251', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600168', '', 'N'),
('MAYBERRY INVESTMENTS LIMITED J$1.981B 12% FR SECURED BOND DUE JAN 2026- TRANCHE 4', 'MIL12FR2026T4', 'JMC202300140', 'JMD', '', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600143', '', 'N'),
('MAYBERRY JAMAICAN EQUITIES LTD 10% FR SECURED BOND TRANCHE 2 DUE AUG 2026', 'MJE10FR2026T2', 'JME202400684', 'JMD', '', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600158', 'https://www.mayberryinv.com/mje/investor-relations/', 'Y'),
('MAYBERRY JAMAICAN EQUITIES LTD 10.5% FR SECURED BOND TRANCHE 3 DUE JUNE 2027', 'MJE10.5FR2027T3', 'JME20240069', 'JMD', '', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600159', 'https://www.mayberryinv.com/mje/investor-relations/', 'Y'),
('MAYBERRY JAMAICAN EQUITIES LTD 9.25% FR SECURED BOND TRANCHE 1 DUE JULY 2025', 'MJE9.25FR2025T1', 'JME202400676', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600157', 'https://www.mayberryinv.com/mje/investor-relations/', 'Y'),
('VM INVESTMENTS LTD 10% FR UNSECURED BOND DUE DEC 2026 - TRANCHE E', 'VMIL10FR1226E', 'JMC202401021', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600166', '', 'N'),
('VM INVESTMENTS LTD 10.5% VR UNSECURED BOND DUE DEC 2027 - TRANCHE F', 'VMILVR1227F', 'JMC202401039', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600167', '', 'N'),
('VM INVESTMENTS LTD 9.75% FR UNSECURED BOND DUE JUNE 2026 - TRANCHE D', 'VMIL9.75FR0626D', 'JMC202401013', 'JMD', 'FINANCE', 'BOND', 'BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600165', '', 'N'),
('BARITA INVESTMENTS LIMITED 7% FR USD UNSECURED BOND DUE JUNE 2027 - TRANCHE I', 'BILUSD7FR2027T1', 'JMC202500400', 'USD', 'FINANCE', 'BOND', 'USD BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600169', '', 'N'),
('BARITA INVESTMENTS LIMITED 7.5% FR USD UNSECURED BOND DUE JUNE 2028 - TRANCHE II', 'BILUSD7.5FR28T2', 'JMC202500418', 'USD', 'FINANCE', 'BOND', 'USD BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600170', '', 'N'),
('EXPRESS CATERING LTD 8.50% FR USD SENIOR UNSECURED BOND DUE MAR 2027', 'ECL8.50FR2027', 'JMC202400197', 'USD', 'RETAIL', 'BOND', 'USD BOND', 'https://www.jamstockex.com/trading/instruments/?instrument=1600155', '', 'N');

-- Verify import
SELECT COUNT(*) as total_records FROM instruments;

-- Sample query to show data structure
SELECT 
    security_name,
    symbol,
    currency,
    sector,
    type,
    market,
    has_website
FROM instruments 
LIMIT 10;

-- Additional useful queries for data analysis

-- Count by market type
SELECT market, COUNT(*) as count 
FROM instruments 
GROUP BY market 
ORDER BY count DESC;

-- Count by currency
SELECT currency, COUNT(*) as count 
FROM instruments 
GROUP BY currency 
ORDER BY count DESC;

-- Count by sector
SELECT sector, COUNT(*) as count 
FROM instruments 
WHERE sector IS NOT NULL AND sector != ''
GROUP BY sector 
ORDER BY count DESC;

-- Count by instrument type
SELECT type, COUNT(*) as count 
FROM instruments 
GROUP BY type 
ORDER BY count DESC;

-- Companies with websites
SELECT 
    COUNT(CASE WHEN has_website = 'Y' THEN 1 END) as with_website,
    COUNT(CASE WHEN has_website = 'N' THEN 1 END) as without_website,
    COUNT(*) as total
FROM instruments;