# Importo las librerias necesarias para trabajar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import seaborn as sns

# Descargo el archivo zip con el dataset original desde https://microdata.worldbank.org/index.php/catalog/394/data-dictionary/F8?file_name=FinStructure_2012_September_Update2x

# Cargo el archivo FinStructure_2012_September_Update2x.csv
df = pd.read_csv(r"C:\Users\Victor\Desktop\EDA_Victor_Bandin\src\data\FinStructure_2012_September_Update2x.csv")

# Limpio el dataset original para poder trabajar con el
df=df.fillna('Latin America & the Caribbean')
fields = ['dbacba','llgdp','cbagdp','dbagdp','ofagdp','pcrdbgdp','pcrdbofgdp','bdgdp',
          'fdgdp','bcbd','ll_usd','overhead','netintmargin','concentration','roa','roe','costinc','zscore','inslife',
          'insnonlife','stmktcap','stvaltraded','stturnover','listco_pc','prbond','pubond','intldebt','intldebtnet','nrbloan',
          'offdep','remit']
regions = ['High-income OECD members','High-income nonOECD members','Middle East and North Africa',
           'East Asia and Pacific','Latin America & the Caribbean','Europe and Central Asia','South Asia',
           'Sub-Saharan Africa']

incLevel = ['Low-income economies','Upper-middle-income economies','High-income nonOECD members',
            'Lower-middle-income economies','High-income OECD members']
for f in fields:
    for r in regions:
        noNaN = df[(df.region == r)][f] != '1.79769313486232e+308'
        media = pd.to_numeric(df[(df.region == r)][noNaN][f], downcast='float').mean()
        mediana = pd.to_numeric(df[(df.region == r)][noNaN][f], downcast='float').median()
        dif = abs(media - mediana)
        if dif >= media/4:
            df.loc[(df[f] == '1.79769313486232e+308') & (df['region'] == r), f] = mediana
        if dif < media/4:
            df.loc[(df[f] == '1.79769313486232e+308') & (df['region'] == r), f] = media

for l in incLevel:
    noNaN = df[df.incgr == l]['prbond'] != '1.79769313486232e+308'
    media = pd.to_numeric(df[df.incgr == l][noNaN]['prbond'], downcast='float').mean()
    mediana = pd.to_numeric(df[df.incgr == l][noNaN]['prbond'], downcast='float').median()
    dif = abs(media - mediana)

    for i in range(len(df.loc[(df['prbond'] == '1.79769313486232e+308') & (df['region'] == 'Middle East and North Africa') & (df.incgr == l), ['prbond']])-1):
        if dif >= media/4:
            df.loc[(df['prbond'] == '1.79769313486232e+308') & (df['region'] == 'Middle East and North Africa') & (df.incgr == l), ['prbond']] = mediana
        if dif < media/4:
            df.loc[(df['prbond'] == '1.79769313486232e+308') & (df['region'] == 'Middle East and North Africa') & (df.incgr == l), ['prbond']] = media
for f in fields:
    df[f] = pd.to_numeric(df[f], downcast='float')

# Tras confirmar que todo esta ok, guardo una copia en nuevo csv para tener una base de trabajo sin alterar el original

df.to_csv(r'C:\Users\Victor\Desktop\EDA_Victor_Bandin\src\data\df_EDA_Victor_Bandin.csv', index=False, header=True)

# Cargo el nuevo dataset
df = pd.read_csv(r"C:\Users\Victor\Desktop\EDA_Victor_Bandin\src\data\df_EDA_Victor_Bandin.csv")

# Preparo las funciones para generar tablas especificas y graficos por WB Region, WB Income Group, pais y años y periodos especificos

def hist_metric(metric='stturnover',startYear=1960,stepYear=1):
    """
    Genera un histograma de la variable indicada, pudiendo seleccionar el año de inicio de las observaciones, asi como el salto en el año de las observaciones.

    Args:
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
        startYear (int): Año de inicio en las observaciones
        stepYear (int): Salto de año en año, de dos en dos, etc
    """
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}                                       
    return sns.distplot(df[['year',metric]][(df.year > startYear)&(df.year%stepYear==0)][metric]).set(title = str(metric)+'\n'+str(labels[metric]));

def hist_metric_filtered(metric='stturnover',territoryFilter='cn',territory='Thailand',startYear=1960,stepYear=1):
    """
    Genera un histograma de la variable indicada, pudiendo seleccionar, además de el año de inicio y el salto en el año de las observaciones, datos especificos
    de un pais, region o grupo de nivel de ingresos del Banco Mundial para su análisis especifico.

    Args:
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
        startYear (int): Año de inicio en las observaciones
        stepYear (int): Salto de año en año, de dos en dos, etc
        territoryFilter (str): Selección del tipo de filtrado: paises, regiones o grupos por nivel de ingresos segun la clasificación del BM
        territory (str): Selección del pais, región o grupo especifico. 
    """
    labels={'cn':'Country','incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}                                       
    return sns.distplot(df[['year',metric]][(df.year > startYear)&(df.year%stepYear==0)&(df[territoryFilter]==territory)][metric]).set(title = str(metric)+'\n'+str(labels[metric])+'\n'+str(labels[territoryFilter])+'\n'+str(territory));

def year_table_byCountry(metric='stturnover',country='Thailand',startYear=1988):
    """
    Genera un tabla de la evolución de la variable indicada, año a año, pudiendo seleccionar, además de el año de inicio, 
    un país concreto para su análisis especifico.

    Args:
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
        startYear (int): Año de inicio en las observaciones
        country (str): Selección del pais especifico. 
    """
    data=df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)].groupby(['cn','year']).mean().round(2)                                        
    return data

def byYear_grouped_table(groupedBy='region', metric='dbacba',year=2010):
    """
    Genera un tabla de la evolución de la variable indicada, año a año, pudiendo seleccionar, además de el año de inicio, 
    una región BM concreta para su análisis especifico.

    Args:
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
        startYear (int): Año de inicio en las observaciones
        groupedBy (str): Selección de la región BM especifica. 
    """    
    byYear_grouped_by = df[[groupedBy,metric]][df.year==year].groupby([groupedBy]).mean().round(2)
    byYear_grouped_by = byYear_grouped_by.reset_index()
    byYear_grouped_by=byYear_grouped_by.sort_values(by=[metric],ascending=False)
    return byYear_grouped_by

def mean_grouped_table(groupedBy='region', metric='dbacba'):
    """
    Genera un tabla resumen con la media la variable indicada en una región BM concreta para su análisis especifico.

    Args:
        groupedBy (str): Selección de la región BM especifica. 
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
    """
    mean_grouped_by = df[[groupedBy,metric]].groupby([groupedBy]).mean().round(2)
    mean_grouped_by = mean_grouped_by.reset_index()
    mean_grouped_by=mean_grouped_by.sort_values(by=[metric],ascending=False)
    return mean_grouped_by

def compare_first_last_year_mean_grouped_table(groupedBy='region', metric='dbacba'):
    """
    Genera un tabla resumen que incluye la media, el dato de 1960 y el de 2010 de la variable indicada en una región BM concreta para su análisis especifico.
    Args:
        groupedBy (str): Selección de la región BM especifica. 
        metric (str): Métrica del Banco Mundial (Columna en el dataset)
    """
    first_year_grouped_by = df[[groupedBy,metric]][df.year==1960].groupby([groupedBy]).mean().round(2)
    first_year_grouped_by = first_year_grouped_by.reset_index()
    first_year_grouped_by=first_year_grouped_by.sort_values(by=[metric],ascending=False)
    last_year_grouped_by = df[[groupedBy,metric]][df.year==2010].groupby([groupedBy]).mean().round(2)
    last_year_grouped_by = last_year_grouped_by.reset_index()
    last_year_grouped_by=last_year_grouped_by.sort_values(by=[metric],ascending=False)
    mean_year_grouped_by = df[[groupedBy,metric]].groupby([groupedBy]).mean().round(2)
    mean_year_grouped_by = mean_year_grouped_by.reset_index()
    mean_year_grouped_by=mean_year_grouped_by.sort_values(by=[metric],ascending=False)
    fVslYear_grouped_by = pd.concat([first_year_grouped_by.sort_values(by=[groupedBy],ascending=False).rename({metric: '1960'}, axis=1), last_year_grouped_by.sort_values(by=[groupedBy],ascending=False)[metric]], axis=1,)
    fVslYear_grouped_by=fVslYear_grouped_by.rename({metric: '2010'}, axis=1)
    fVslYear_grouped_by = pd.concat([fVslYear_grouped_by, mean_year_grouped_by[metric]], axis=1,)
    fVslYear_grouped_by=fVslYear_grouped_by.sort_values(by=['2010'],ascending=False).rename({metric: 'Mean '+str(metric)}, axis=1)
    if groupedBy == 'incgr':
        fVslYear_grouped_by=fVslYear_grouped_by.replace({'incgr': {'High-income OECD members':'H Inc OECD',
                                                               'High-income nonOECD members':'H Inc noOECD', 
                                                               'Upper-middle-income economies':'U-M Inc', 
                                                               'Lower-middle-income economies':'L-M Inc', 
                                                               'Low-income economies':'Low Inc'}})
    if groupedBy == 'region':
        fVslYear_grouped_by=fVslYear_grouped_by.replace({'region': {'High-income OECD members':'OECD',
                                                         'High-income nonOECD members':'HI noOECD', 
                                                         'East Asia and Pacific':'EAsia P', 
                                                         'Europe and Central Asia':'EuCAsia', 
                                                         'Latin America & the Caribbean':'Latam',
                                                         'Middle East and North Africa':'ME NA',
                                                         'Sub-Saharan Africa':'SS Africa',
                                                         'South Asia':'S Asia'}})
    return fVslYear_grouped_by

def metric_biennial_graph_byCountry(metric='stturnover',country='Thailand',startYear=1988):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    x_year= df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%2==0)].year
    y_country_data=df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%2==0)][metric]
    data=df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%2==0)].groupby(['cn','year']).mean()                                        
    return sns.barplot(x=x_year, y=y_country_data, data=data).set(title = str(metric)+'\n'+str(labels[metric])+'\n'+str(country));

def metric_quadrennial_graph_byCountry(metric='stturnover',country='Thailand',startYear=1960):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    x_year= df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%4==0)].year
    y_country_data=df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%4==0)][metric]
    data=df[['year','cn',metric]][(df.cn==country)&(df.year > startYear)&(df.year%4==0)].groupby(['cn','year']).mean()                                        
    return sns.barplot(x=x_year, y=y_country_data, data=data).set(title = str(metric)+'\n'+str(labels[metric])+'\n'+str(country));

def metric_quadrennial_graph_grouped(metric='stturnover',groupedBy='region',group='High-income OECD members',startYear=1960):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    x_year= df[['year',groupedBy,metric]][(df[groupedBy]==group)&(df.year > startYear)&(df.year%4==0)].year
    y_group_data=df[['year',groupedBy,metric]][(df[groupedBy]==group)&(df.year > startYear)&(df.year%4==0)][metric]
    data=df[['year',groupedBy,metric]][(df[groupedBy]==group)&(df.year > startYear)&(df.year%4==0)].groupby([groupedBy,df.year]).mean()                                        
    return sns.barplot(x=x_year, y=y_group_data, data=data).set(title = str(metric)+'\n'+str(labels[metric])+'\n'+str(group));

def byYear_grouped_graph(groupedBy='region', metric='dbacba',year=2010):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    byYear_grouped_by = df[[groupedBy,metric]][df.year==year].groupby([groupedBy]).mean().round(2)
    byYear_grouped_by = byYear_grouped_by.reset_index()
    byYear_grouped_by=byYear_grouped_by.sort_values(by=[metric],ascending=False)
    if groupedBy == 'incgr':
        byYear_grouped_by=byYear_grouped_by.replace({'incgr': {'High-income OECD members':'H Inc OECD',
                                                               'High-income nonOECD members':'H Inc noOECD', 
                                                               'Upper-middle-income economies':'U-M Inc', 
                                                               'Lower-middle-income economies':'L-M Inc', 
                                                               'Low-income economies':'Low Inc'}})
    if groupedBy == 'region':
        byYear_grouped_by=byYear_grouped_by.replace({'region': {'High-income OECD members':'OECD',
                                                         'High-income nonOECD members':'HI noOECD', 
                                                         'East Asia and Pacific':'EAsia P', 
                                                         'Europe and Central Asia':'EuCAsia', 
                                                         'Latin America & the Caribbean':'Latam',
                                                         'Middle East and North Africa':'ME NA',
                                                         'Sub-Saharan Africa':'SS Africa',
                                                         'South Asia':'S Asia'}})
    return sns.barplot(x=groupedBy, y=metric, data=byYear_grouped_by).set(title = str(labels[metric])+'\n'+str(labels[groupedBy])+'\n'+str(year));

def mean_grouped_graph(groupedBy='region', metric='dbacba'):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    mean_grouped_by = df[[groupedBy,metric]].groupby([groupedBy]).mean().round(2)
    mean_grouped_by = mean_grouped_by.reset_index()
    mean_grouped_by=mean_grouped_by.sort_values(by=[metric],ascending=False)
    if groupedBy == 'incgr':
        mean_grouped_by=mean_grouped_by.replace({'incgr': {'High-income OECD members':'H Inc OECD',
                                                               'High-income nonOECD members':'H Inc noOECD', 
                                                               'Upper-middle-income economies':'U-M Inc', 
                                                               'Lower-middle-income economies':'L-M Inc', 
                                                               'Low-income economies':'Low Inc'}})
    if groupedBy == 'region':
        mean_grouped_by=mean_grouped_by.replace({'region': {'High-income OECD members':'OECD',
                                                         'High-income nonOECD members':'HI noOECD', 
                                                         'East Asia and Pacific':'EAsia P', 
                                                         'Europe and Central Asia':'EuCAsia', 
                                                         'Latin America & the Caribbean':'Latam',
                                                         'Middle East and North Africa':'ME NA',
                                                         'Sub-Saharan Africa':'SS Africa',
                                                         'South Asia':'S Asia'}})
    return sns.barplot(x=groupedBy, y=metric, data=mean_grouped_by).set(title = str(metric)+'\n'+str(labels[metric])+'\n'+str(labels[groupedBy]));

def compare_first_last_year_mean_grouped_graph(groupedBy='region', metric='dbacba'):
    labels={'incgr':'World Bank Income Group','region':'World Bank Region','dbacba':'Deposit Money Bank Assets to (Deposit Money + Central) Bank Assets',
        'llgdp':'Liquid liabilities to GDP (%)','cbagdp':'Central bank assets to GDP (%)',
        'dbagdp':'Deposit money banks assets to GDP (%)','ofagdp':'Other Finantial Institutions assets to GDP (%)',
        'pcrdbgdp':'Private credit by deposit money banks to GDP (%)','pcrdbofgdp':'Private credit by deposit money banks and other financial institutions to GDP (%)',
        'bdgdp':'Bank deposits to GDP (%)','fdgdp':'Financial system deposits to GDP (%)','bcbd':'Bank credit to bank deposits (%)',
        'll_usd':'Liquid liabilities in millions USD (2010 constant)','overhead':'Bank overhead costs to total assets (%)',
        'netintmargin':'Net interest margin (NIM) (%)','concentration':'Bank concentration (%)\nShare of assets of a countrys three largest banks',
        'roa':'Banks Return on Assets (ROA) (%)','roe':'Banks Return on Equity (ROE) (%)','costinc':'Cost-to-income ratio\nCalculated by dividing the operating expenses by the operating income generated',
        'zscore':'z-score\nCompares buffers (capitalization and returns)\nwith risk (volatility of returns) to measure a bank’s solvency risk',
        'inslife':'Life insurance premium volume to GDP (%)', 'insnonlife':'Non-life insurance premium volume to GDP (%)',
        'stmktcap':'Stock Market Capitalization / GDP (%)','stvaltraded':'Stock Market Total Value Traded / GDP (%)',
        'stturnover':'Stock Market Turnover Ratio\nTurnover ratio is the value of domestic shares traded divided by their market capitalization',
        'listco_pc':'Nº of listed companies per 10K population','prbond':'Private bond market capitalization to GDP (%)',
        'pubond':'Public bond market capitalization to GDP (%)','intldebt':'International debt issues to GDP (%)',
        'intldebtnet':'Loans from nonresident banks (net) to GDP (%)','nrbloan':'Loans from Non-Resident Banks, Amounts Outstanding, to GDP (%)',
          'offdep':'Offshore bank deposits to domestic bank deposits (%)','remit':'Personal remittances received to GDP (%)'}
    first_year_grouped_by = df[[groupedBy,metric]][df.year==1960].groupby([groupedBy]).mean().round(2)
    first_year_grouped_by = first_year_grouped_by.reset_index()
    first_year_grouped_by=first_year_grouped_by.sort_values(by=[metric],ascending=False)
    last_year_grouped_by = df[[groupedBy,metric]][df.year==2010].groupby([groupedBy]).mean().round(2)
    last_year_grouped_by = last_year_grouped_by.reset_index()
    last_year_grouped_by=last_year_grouped_by.sort_values(by=[metric],ascending=False)
    mean_year_grouped_by = df[[groupedBy,metric]].groupby([groupedBy]).mean().round(2)
    mean_year_grouped_by = mean_year_grouped_by.reset_index()
    mean_year_grouped_by=mean_year_grouped_by.sort_values(by=[metric],ascending=False)
    fVslYear_grouped_by = pd.concat([first_year_grouped_by.sort_values(by=[groupedBy],ascending=False).rename({metric: '1960'}, axis=1), last_year_grouped_by.sort_values(by=[groupedBy],ascending=False)[metric]], axis=1,)
    fVslYear_grouped_by=fVslYear_grouped_by.rename({metric: '2010'}, axis=1)
    fVslYear_grouped_by = pd.concat([fVslYear_grouped_by, mean_year_grouped_by[metric]], axis=1,)
    fVslYear_grouped_by=fVslYear_grouped_by.sort_values(by=['2010'],ascending=False).rename({metric: 'Mean '+str(metric)}, axis=1)
    if groupedBy == 'incgr':
        fVslYear_grouped_by=fVslYear_grouped_by.replace({'incgr': {'High-income OECD members':'H Inc OECD',
                                                               'High-income nonOECD members':'H Inc noOECD', 
                                                               'Upper-middle-income economies':'U-M Inc', 
                                                               'Lower-middle-income economies':'L-M Inc', 
                                                               'Low-income economies':'Low Inc'}})
    if groupedBy == 'region':
        fVslYear_grouped_by=fVslYear_grouped_by.replace({'region': {'High-income OECD members':'OECD',
                                                         'High-income nonOECD members':'H noOECD', 
                                                         'East Asia and Pacific':'EAsia P', 
                                                         'Europe and Central Asia':'EuCAsia', 
                                                         'Latin America & the Caribbean':'Latam',
                                                         'Middle East and North Africa':'ME NA',
                                                         'Sub-Saharan Africa':'SS Africa',
                                                         'South Asia':'S Asia'}})
    
    fVslYear_grouped_by=fVslYear_grouped_by.set_index(groupedBy)

    n = len(fVslYear_grouped_by.index)
    x = np.arange(n)
    width = 0.25
    plt.bar(x - width, fVslYear_grouped_by['1960'], width=width, label='1960')
    plt.bar(x, fVslYear_grouped_by['2010'], width=width, label='2010')
    plt.bar(x + width, fVslYear_grouped_by['Mean '+str(metric)], width=width, label='Media')
    plt.xticks(x, fVslYear_grouped_by.index)
    plt.legend(loc='best')
    plt.title(str(labels[metric])+'\n'+str(labels[groupedBy])+'\n');
    return plt.show()