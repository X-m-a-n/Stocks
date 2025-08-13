jse_company_names: dict[str, list[str]] = {
# Financial Sector - Main Market
'138SL': ['138SL', '138 Student Living', '138 Student Living Jamaica', '138 Student Living Jamaica Limited', 'Student Living'],
'BIL': ['BIL', 'Barita', 'Barita Investments', 'Barita Investments Limited'],
'CPFV': ['CPFV', 'Eppley Caribbean Property Fund', 'Eppley Caribbean Property Fund Limited', 'ECPF', 'Caribbean Property Fund'],
'EPLY': ['EPLY', 'Eppley', 'Eppley Limited'],
'JSE': ['JSE', 'Jamaica Stock Exchange', 'Stock Exchange'],
'JMMBGL': ['JMMBGL', 'JMMB', 'JMMB Group', 'JMMB Group Limited'],
'NCBFG': ['NCBFG', 'NCB', 'NCB Financial', 'NCB Financial Group', 'NCB Financial Group Limited', 'National Commercial Bank'],
'PROVEN': ['PROVEN', 'Proven', 'Proven Group', 'Proven Group Limited'],
'SGJ': ['SGJ', 'Scotia', 'Scotia Group Jamaica', 'Scotia Group Jamaica Limited', 'Scotiabank Jamaica'],

# Financial Sector - Main Market (Additional)
'FIRSTROCKJMD': ['FIRSTROCKJMD', 'First Rock', 'First Rock Real Estate', 'First Rock Real Estate Investments', 'First Rock Real Estate Investments Limited', 'First Rock REIT'],
'GENAC': ['GENAC', 'General Accident', 'General Accident Insurance', 'General Accident Insurance Company', 'General Accident Insurance Company (JA) Limited'],
'GHL': ['GHL', 'Guardian Holdings', 'Guardian Holdings Limited'],
'KEY': ['KEY', 'Key Insurance', 'Key Insurance Company', 'Key Insurance Company Limited'],
'KPREIT': ['KPREIT', 'Kingston Properties', 'Kingston Properties Limited', 'Kingston REIT'],
'MASSY': ['MASSY', 'Massy Holdings', 'Massy Holdings Ltd'],
'MGL': ['MGL', 'Mayberry Group', 'Mayberry Group Ltd.'],
'MJE': ['MJE', 'Mayberry Jamaican Equities', 'Mayberry Jamaican Equities Limited'],
'PULS': ['PULS', 'Pulse Investments', 'Pulse Investments Limited'],
'QWI': ['QWI', 'QWI Investments', 'QWI Investments Limited'],
'SCIJMD': ['SCIJMD', 'Sygnus Credit Investments', 'Sygnus Credit Investments Limited', 'SCI'],
'SELECTF': ['SELECTF', 'Sagicor Select Funds Financial', 'Sagicor Select Funds Limited Financial'],
'SELECTMD': ['SELECTMD', 'Sagicor Select Funds Manufacturing & Distribution', 'Sagicor Select Funds Limited Manufacturing & Distribution'],
'SIL': ['SIL', 'Sterling Investments', 'Sterling Investments Limited'],
'SRFJMD': ['SRFJMD', 'Sygnus Real Estate Finance', 'Sygnus Real Estate Financial Limited', 'SRF'],
'VMIL': ['VMIL', 'VM Investments', 'VM Investments Limited'],
'XFUND': ['XFUND', 'Sagicor Real Estate X Fund', 'Sagicor Real Estate X Fund Ltd.', 'X Fund'],

# Manufacturing Sector - Main Market
'ASBH': ['ASBH', 'A.S. Bryden', 'AS Bryden', 'Bryden', 'A.S. Bryden & Sons', 'A.S. Bryden & Sons Holdings Limited'],
'BRG': ['BRG', 'Berger', 'Berger Paints', 'Berger Paints Jamaica', 'Berger Paints Jamaica Ltd'],
'CCC': ['CCC', 'Caribbean Cement', 'Caribbean Cement Company', 'Caribbean Cement Company Ltd', 'Carib Cement'],
'LASM': ['LASM', 'Lasco', 'Lasco Manufacturing', 'Lasco Manufacturing Limited'],
'SALF': ['SALF', 'Salada', 'Salada Foods', 'Salada Foods Jamaica', 'Salada Foods Jamaica Ltd'],
'SEP': ['SEP', 'Seprod', 'Seprod Limited'],
'WISYNCO': ['WISYNCO', 'Wisynco', 'Wisynco Group', 'Wisynco Group Ltd'],

# Manufacturing Sector - Main Market (Additional)
'CAR': ['CAR', 'Carreras', 'Carreras Limited'],
'CPJ': ['CPJ', 'Caribbean Producers', 'Caribbean Producers Jamaica', 'Caribbean Producers Jamaica Limited'],
'WIG': ['WIG', 'Wigton Energy', 'Wigton Energy Limited'],

# Conglomerate Sector - Main Market
'GK': ['GK', 'GraceKennedy', 'Grace Kennedy', 'GraceKennedy Limited'],
'JP': ['JP', 'Jamaica Producers', 'Jamaica Producers Group', 'Jamaica Producers Group Ltd', 'JP Group'],
'PJAM': ['PJAM', 'Pan Jamaica', 'Pan Jamaica Group', 'Pan Jamaica Group Limited'],
'SJ': ['SJ', 'Sagicor', 'Sagicor Jamaica', 'Sagicor Group Jamaica', 'Sagicor Group Jamaica Limited'],

# Conglomerate Sector - Main Market (Additional)
'JBG': ['JBG', 'Jamaica Broilers', 'Jamaica Broilers Group'],
'KW': ['KW', 'Kingston Wharves', 'Kingston Wharves Limited'],
'SVL': ['SVL', 'Supreme Ventures', 'Supreme Ventures Limited'],

# Energy & Utilities Sector - Main Market (Additional)
'ENERGY': ['ENERGY', 'Innovative Energy', 'Innovative Energy Group', 'Innovative Energy Group Limited'],
'JPS': ['JPS', 'Jamaica Public Service', 'Jamaica Public Service Company', 'Jamaica Public Service Co. Ltd.', 'JPS Company'],
'MPCCEL': ['MPCCEL', 'MPC Caribbean Clean Energy', 'MPC Caribbean Clean Energy Limited'],
'TJH': ['TJH', 'Transjamaican Highway', 'Transjamaican Highway Limited'],

# Entertainment & Media Sector - Main Market (Additional)
'PAL': ['PAL', 'Palace', 'Palace Amusement', 'Palace Amusement Co. Ltd.'],
'RJR': ['RJR', 'Radio Jamaica', 'Radio Jamaica Limited'],

# Real Estate Sector - Main Market (Additional)
'MTL': ['MTL', 'Margaritaville', 'Margaritaville (Turks) Limited'],
'PJX': ['PJX', 'Portland JSX', 'Portland JSX Limited'],
'SML': ['SML', 'Stanley Motta', 'Stanley Motta Limited'],

# Financial Sector - Junior Market
'AFS': ['AFS', 'Access Financial', 'Access Financial Services', 'Access Financial Services Limited'],
'ISP': ['ISP', 'ISP Finance', 'ISP Finance Services', 'ISP Finance Services Limited'],
'LASF': ['LASF', 'Lasco Financial', 'Lasco Financial Services', 'Lasco Financial Services Limited'],
'DOLLA': ['DOLLA', 'Dolla', 'Dolla Financial', 'Dolla Financial Services', 'Dolla Financial Services Limited'],

# Financial Sector - Junior Market (Additional)
'MFS': ['MFS', 'MFS Capital Partners', 'MFS Capital Partners Limited'],
'ROC': ['ROC', 'Ironrock Insurance', 'Ironrock Insurance Company', 'Ironrock Insurance Company Limited'],

# Manufacturing Sector - Junior Market
'AMG': ['AMG', 'AMG Packaging', 'AMG Packaging & Paper', 'AMG Packaging & Paper Company Limited'],
'BPOW': ['BPOW', 'Blue Power', 'Blue Power Group', 'Blue Power Group Limited'],
'KREMI': ['KREMI', 'Caribbean Cream', 'Caribbean Cream Limited', 'Kremi'],
'CFF': ['CFF', 'Caribbean Flavours', 'Caribbean Flavours & Fragrances', 'Caribbean Flavours & Fragrances Limited', 'Caribbean F&F'],
'PURITY': ['PURITY', 'Consolidated Bakeries', 'Consolidated Bakeries Jamaica', 'Consolidated Bakeries (Jamaica) Limited', 'Purity'],
'HONBUN': ['HONBUN', 'Honey Bun', 'Honey Bun (1982) Limited'],
'JAMT': ['JAMT', 'Jamaican Teas', 'Jamaican Teas Limited', 'Jamaica Teas'],
'OMNI': ['OMNI', 'Omni Industries', 'Omni Industries Limited'],
'PTL': ['PTL', 'Paramount Trading', 'Paramount Trading Jamaica', 'Paramount Trading (Jamaica) Limited'],

# Manufacturing Sector - Junior Market (Additional)
'ELITE': ['ELITE', 'Elite Diagnostic', 'Elite Diagnostic Limited'],
'FESCO': ['FESCO', 'Future Energy Source', 'Future Energy Source Company', 'Future Energy Source Company Ltd'],
'JFP': ['JFP', 'JFP Limited'],
'TROPICAL': ['TROPICAL', 'Tropical Battery', 'Tropical Battery Company', 'Tropical Battery Company Limited'],

# Distribution Sector - Junior Market
'CAC': ['CAC', 'CAC 2000', 'CAC 2000 Limited'],
'DTL': ['DTL', 'Derrimon', 'Derrimon Trading', 'Derrimon Trading Company', 'Derrimon Trading Company Limited'],
'FTNA': ['FTNA', 'Fontana', 'Fontana Limited'],
'INDIES': ['INDIES', 'Indies Pharma', 'Indies Pharma Jamaica', 'Indies Pharma Jamaica Limited'],
'JETCON': ['JETCON', 'Jetcon', 'Jetcon Corporation', 'Jetcon Corporation Limited'],
'MDS': ['MDS', 'Medical Disposables', 'Medical Disposables & Supplies', 'Medical Disposables & Supplies Limited'],
'RPL': ['RPL', 'Regency Petroleum', 'Regency Petroleum Company', 'Regency Petroleum Co. Limited'],

# Distribution Sector - Junior Market (Additional)
'AHPC': ['AHPC', 'Atlantic Hardware', 'Atlantic Hardware and Plumbing', 'Atlantic Hardware and Plumbing Company Limited'],
'LASD': ['LASD', 'Lasco Distributors', 'Lasco Distributors Limited'],
'LUMBER': ['LUMBER', 'Lumber Depot', 'Lumber Depot Limited'],
'RAWILL': ['RAWILL', 'R.A. Williams Distributors', 'R.A. Williams Distributors Limited', 'RA Williams'],

# Technology Sector - Junior Market
'LEARN': ['LEARN', 'Edufocal', 'Edufocal Limited'],
'TTECH': ['TTECH', 'tTech', 'T-Tech', 'tTech Limited'],
'LAB': ['LAB', 'Limners and Bards', 'The Limners and Bards', 'The Limners and Bards Limited'],

# Technology Sector - Junior Market (Additional)
'1GS': ['1GS', 'One Great Studio', 'One Great Studio Company', 'One Great Studio Company Limited'],
'IPCL': ['IPCL', 'Image Plus Consultants', 'Image Plus Consultants Limited'],
'ONE': ['ONE', 'One on One Educational Services', 'One on One Educational Services Limited'],

# Services Sector - Junior Market
'CABROKERS': ['CABROKERS', 'Caribbean Assurance', 'Caribbean Assurance Brokers', 'Caribbean Assurance Brokers Limited'],
'CHL': ['CHL', 'Cargo Handlers', 'Cargo Handlers Limited'],
'EFRESH': ['EFRESH', 'Everything Fresh', 'Everything Fresh Limited'],
'ECL': ['ECL', 'Express Catering', 'Express Catering Limited'],
'FOSRICH': ['FOSRICH', 'Fosrich', 'Fosrich Company', 'Fosrich Company Limited'],
'GWEST': ['GWEST', 'GWest', 'GWest Corporation', 'GWest Corporation Limited'],
'MAILPAC': ['MAILPAC', 'Mailpac', 'Mailpac Group', 'Mailpac Group Limited'],
'MEEG': ['MEEG', 'Main Event', 'Main Event Entertainment', 'Main Event Entertainment Group'],
'SOS': ['SOS', 'Stationery and Office Supplies', 'Stationery and Office Supplies Limited'],
'SPURTREE': ['SPURTREE', 'Spur Tree', 'Spur Tree Spices', 'Spur Tree Spices Jamaica', 'Spur Tree Spices Jamaica Limited'],

# Services Sector - Junior Market (Additional)
'DCOVE': ['DCOVE', 'Dolphin Cove', 'Dolphin Cove Limited'],
'KEX': ['KEX', 'Knutsford Express', 'Knutsford Express Services', 'Knutsford Express Services Limited'],
'KLE': ['KLE', 'K.L.E. Group', 'K.L.E. Group Limited'],
'KNTYR': ['KNTYR', 'Kintyre Holdings', 'Kintyre Holdings (JA) Limited', 'iCreate', 'iCreate Limited', 'iCreate Institute']
}