#Este Projeto foi desenvolvido pelo Aluno Fábio Soares, para conclusão do Curso Python para o Mercado Financeiro, pela Trading com Dados

import streamlit as st
import babel
from babel import Locale
#import locale
import datetime

from datetime import datetime, timedelta
from datetime import date

import pandas as pd
import yfinance as yf
from urllib.request import Request, urlopen
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import vectorbt as vbt
import statsmodels.api as sm
from GoogleNews import GoogleNews


#configuracao da pagina:
st.set_page_config(layout= 'wide', page_title = 'BDRs!')

#configuracao para esconder o menu deploy do streamlit
#st.markdown("""
#    <style> .reportview-container { margin-top: -2em; } #MainMenu {visibility: hidden;} .stDeployButton {display:none;}
#        footer {visibility: hidden;} #stDecoration {display:none;} </style> """, unsafe_allow_html=True)


#ajuste das cores de fundo ----------------------------------------------
# Define as cores do tema
primaryColor="#FF4B4B"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans-serif"

# Define o tema
custom_theme = f"""
[data-testid="stTheme"][data-name="primaryColor"] div {{
  background-color: {primaryColor};
}}
[data-testid="stTheme"][data-name="backgroundColor"] div {{
  background-color: {backgroundColor};
}}
[data-testid="stTheme"][data-name="secondaryBackgroundColor"] div {{
  background-color: {secondaryBackgroundColor};
}}
[data-testid="stTheme"][data-name="textColor"] div {{
  color: {textColor};
}}
[data-testid="stTheme"][data-name="font"] div {{
  font-family: {font};
}}
"""

# Aplica o tema
st.markdown(f'<style>{custom_theme}</style>', unsafe_allow_html=True)
# fim do ajuste das cores de fundo ---------------------------------------------------------

# setar locale para português
locale = Locale('pt_BR')
#locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')


#ajustando o cabecalho
st.write('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)


#funcao que faz o webscraping dos dados de bdrs do site investnews...........................................
@st.cache_data
def load_bdr(url):
    request_site = Request(url, headers = {'User-Agent':'Mozilla/5.0'}) # o pedido ao site
    webpage = urlopen(request_site).read() 
    df = pd.read_html(webpage)[0]                    #como ele trouxe em forma de lista, vamos iterar com o elemento ZERO
    df.columns = df.iloc[0]                          #passando o nome das colunas que estava na linha zero do dataframe
    df.drop([0], axis =0, inplace = True)            #tiramos a linha zero que esta com as mesmas informacoes que o nome das colunas
    return(df)
#fim da funcao................................................................................................

#funcao para criar o df_escolha------------------------------------------------------------------------------
@st.cache_data(ttl=15*60, show_spinner = 'Baixando dados do Yahoo Finance...')  # <<< 15 minutos é ideal
def cria_df_escolha(tickers_escolha_yf,inicio, fim):
    tickers_escolha_yf = sorted(tickers_escolha_yf) #organiza os ativos selecionados para nao fazer buscas a toa no yahoo finance por conta do cache
    df_escolha = pd.DataFrame()
    df_escolha = yf.download(tickers_escolha_yf, start = inicio, end = fim, rounding = True, threads=False)['Close']
    df_escolha = pd.DataFrame(df_escolha)
        
    if len(tickers_escolha_yf)== 1: #quando so tem 1 ticker escolhido, da erro no filtro e no fundamentos
        df_escolha.rename(columns = {'Close': tickers_escolha_yf[0]}, inplace = True) 

    #filtro para excluir tickers que tenham muitos nans. vamos pegar somente tickers com 90% de dados
    #a lista dos ativos que tem nans / a qtde de linhas que tem no dataframe filtrando o true que der 90% ou mais 
    df_escolha_filtrados = df_escolha[df_escolha.columns[df_escolha.isna().sum() / df_escolha.shape[0] < 0.9]]
    #preenchendo os poucos nans que restarem com o preco do etf pra frente
    df_escolha_filtrados = df_escolha_filtrados.ffill(axis = 0)
    return (df_escolha_filtrados)
#fim da funcao................................................................................................

#funcao para Criacao da carteira de investimentos:-----------------------------------------------------------
def cria_carteira(df_escolha_filtrados_retorno):    
    carteira = df_escolha_filtrados_retorno
    #criacao do array de pesos iguais , de acordo com a qtde de acoes que existam no setor
    total_ativos = df_escolha_filtrados_retorno.columns.shape[0]
    try:
        divisao_pesos = (1/total_ativos)
    except ZeroDivisionError:
        st.error('BDR indisponivel ou com dados faltantes! Escolha outro ativo', icon="🚨")

    numeros = []
    for i in range(0, total_ativos):
        numeros.append(divisao_pesos)
    pesos = np.array(numeros)

    carteira = (carteira * pesos).dropna()                       # multiplica pelo peso de cada ativo na carteira
    carteira = round((carteira) * 100,2)
    carteira['retorno diario'] = carteira.sum(axis=1)            # calcula o retorno diario da carteira, somando o retorno de cada papel no dia pelas colunas
    carteira['retorno carteira']= round(((carteira['retorno diario']).cumsum()),2)# retorno acumulado da carteira

    return(carteira,pesos,total_ativos)
#fim da funcao de criacao de carteira ------------------------------------------------------------------------------------------

#funcao para calculo do beta, alpha, volatilidade da carteira
def calcula_beta_vol(carteira, ibov_retornos,df_escolha_filtrados_retorno, pesos):
    carteira_retorno_diario = carteira['retorno diario']
    ibov_retornos = ibov_retornos*100
    beta_carteira = pd.merge(carteira_retorno_diario, ibov_retornos, how = 'inner', left_index= True, right_index = True)
    beta_carteira.dropna(inplace = True)

    Y = beta_carteira['retorno diario']     #criamos a variavel dependente
    X = beta_carteira['ibov']               #criamos a variavel independente
    X = sm.add_constant(X)                  #conforme a equacao
    modelo = sm.OLS(Y,X)                    #metodo do python para o resultado, passa a variavel dependente e depois a independente
    resultado = modelo.fit()                #criou um objeto dentro do status model
    # o beta representa a sensibilidade do ativo, com relacao ao risco sistematico no caso o ibovespa
    alpha_carteira = resultado.params[0]
    beta_carteira  = resultado.params[1]

    with col_dados:
        st.metric('BETA da Carteira:', value = str((beta_carteira).round(2))+ ' %')
        st.metric('ALPHA da Carteira:', value = str((alpha_carteira).round(2))+ ' %')

    #calculando da volatilidade da carteira:
    cov_carteira = df_escolha_filtrados_retorno.cov()
    vol_carteira = np.sqrt(np.dot(pesos.T, np.dot(cov_carteira, pesos))) #a raiz quadrada da soma produto de matrizes.
    vol_ano = vol_carteira * np.sqrt(252) # aqui usamos 252 dias uteis para trazer o resultad para ano
    
    with col_dados:
        st.metric('Volatilidade/Oscilação ANUAL da carteira:', value = str((vol_ano*100).round(2)) + '% ')
    return()
#fim da funcao -----------------------------------------------------------------------------------------


#DRAWDOWN: 1 - carteira original  com pesos iguais---------------------------------------------------  
def calcula_dd(df_escolha_filtrados, pesos):
    # Ajusta o índice do DataFrame para dias úteis
    #df_escolha_filtrados = df_escolha_filtrados.asfreq('B')  # 'B' representa frequência de dias úteis
    df_escolha_filtrados.index = df_escolha_filtrados.index.to_period('D').to_timestamp()
    
    with col_dd:
      try:
          # Cria o portfólio com os dados ajustados
          drawdown_carteira_original = vbt.Portfolio.from_orders(close=df_escolha_filtrados, size=pesos, size_type='targetpercent',
          group_by=True, cash_sharing=True)
          fig = drawdown_carteira_original.plot_underwater().update_layout(title='Drawdown do portfólio', height=350, width = 490, yaxis=dict( title='Queda %', tickformat = '.2%'))
          st.plotly_chart(fig)
        
      except Exception as e:
            # Emite aviso caso ocorra um erro
            st.warning("Não foi possível calcular o drawdown. Por favor, selecione um período maior para análise.")
            st.error(f"Detalhes do erro: {e}")
            fig = None  # Define o retorno como None em caso de erro
    return(fig)
#fim da funcao drawdown -----------------------------------------------------------------------------



#criando dataframe fundamentalista com base nas empresas escolhidas:
@st.cache_data(ttl=6*60*60)  # <<< 6 horas
def cria_df_fundamentalista(df_escolha_filtrados):

    data = []

    # 🔑 NORMALIZAÇÃO PARA CACHE
    stocks_fundamental = sorted(df_escolha_filtrados.columns.tolist())

    for ticker in stocks_fundamental:
        try:
            company     = yf.Ticker(ticker)

            fundamental_data = company.info
            financials  = company.financials
            balance     = company.balancesheet

            # ================= ALTERAÇÃO 1 =================
            # Validação mínima dos dados fundamentais
            if not fundamental_data or financials is None or balance is None:
                continue  # <<< pula ticker sem dados
            # =================================================

            # ================= ALTERAÇÃO 2 =================
            # Validação de linhas críticas nos demonstrativos
            if (
                'Net Income' not in financials.index
                or 'Stockholders Equity' not in balance.index
            ):
                continue  # <<< evita IndexError
            # =================================================

            fundamental_items = {
                'CÓDIGO': ticker,
                'Company': fundamental_data.get('longName'),
                'Sector': fundamental_data.get('sector'),
                'MarketCap': fundamental_data.get('marketCap'),
                'Revenue': fundamental_data.get('totalRevenue'),
                'beta': fundamental_data.get('beta'),
                'ebitda': fundamental_data.get('ebitda'),
                'dividendRate': fundamental_data.get('dividendRate'),
                'dividendYield': fundamental_data.get('dividendYield'),
                'Gross Profit': fundamental_data.get('grossProfits'),

                # ================= ALTERAÇÃO 3 =================
                # Acesso seguro aos valores contábeis
                'NetIncome': financials.loc['Net Income'].iloc[0],
                'Total Equity': balance.loc['Stockholders Equity'].iloc[0],
                # =================================================

                'ROE': fundamental_data.get('returnOnEquity'),
                'Sumario': fundamental_data.get('longBusinessSummary'),
                'Website': fundamental_data.get('website')
            }

            data.append(fundamental_items)

        except Exception as e:
            # ================= ALTERAÇÃO 4 =================
            # Falha controlada por ticker (log silencioso)
            # st.warning(f"Ticker ignorado por falta de dados: {ticker}")
            continue
            # =================================================

    # ================= ALTERAÇÃO 5 =================
    # Evita retornar DataFrame vazio
    if not data:
        return pd.DataFrame()
    # =================================================

    df_fundamentalista = pd.DataFrame(data)
    df_fundamentalista.index = df_fundamentalista['CÓDIGO']
    df_fundamentalista['CÓDIGO'] = df_fundamentalista['CÓDIGO'].str[0:-3]

    return df_fundamentalista
#fim da funcao................................................................................................


#grafico de precos:-------------------------------------------------------------
def grafico_preco(df_escolha_filtrados):
    fig = px.line(df_escolha_filtrados, x=df_escolha_filtrados.index, y=df_escolha_filtrados.columns, title = '<b>Análise de Preços de BDRs</b>')
     #fig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
    fig.update_layout(title_text='<b>Análise de Preços de BDRs</b>', title_font=dict(size=24),  title_x = 0.3, template='simple_white', yaxis_title = '<b>Preço: R$</b>', xaxis_title = '<b>Período</b>', legend_title_text='BDRs:')
    st.plotly_chart(fig, use_container_width = True)
    return(fig)
#fim do grafico de precos-------------------------------------------------------


#grafico de retorno %-----------------------------------------------------------
def grafico_retorno(df_escolha_filtrado_normalizado, resultados):
    fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    for i in df_escolha_filtrado_normalizado.columns:
        legenda = str(resultados[resultados['CÓDIGO'].str.contains(i[0:-3])]['Company'].values)
        fig.add_trace(go.Scatter(name= legenda , x = df_escolha_filtrado_normalizado.index, y = (df_escolha_filtrado_normalizado[i]*100)-100), row=1, col=1)

    fig.update_layout(title_text='Análise dos <b>Rendimentos</b> de BDRs', title_font=dict(size=24), title_x = 0.3,  template='simple_white',yaxis_title ='<b>Retorno em %</b>',xaxis_title ='<b>Período</b>')
    st.plotly_chart(fig, use_container_width = True)
    return(fig)
#fim do grafico de retornos-------------------------------------------------------

#Grafico do retorno acumulado:------------------------------------------------------------------------------
def grafico_acumulado(acumulado):
    fig=px.bar(acumulado, x = acumulado.index.str[0:-3], y = acumulado.Retorno , color = acumulado.index, title = 'Análise dos Rendimentos Acumulados por Ativo')
    fig.update_yaxes(title = 'Retorno em %') # antes estava color = 'Retorno'
    fig.update_xaxes(title = 'Ativos')
    fig.update_layout(showlegend = False)
    st.write(fig)
    return(fig)
#fim do grafico de acumulado----------------------------------------------------------------------

#graficos DA DISTRIBUICAO DOS PESOS para analise------------------------
def grafico_pesos(df_escolha_filtrados, pesos):
    #ativos que compoem a carteira: df_escolha_filtrados.columns) PESOS ORIGINAL DA CARTEIRA (PESOS)
    df_pesos = pd.DataFrame(data={'pesos':pesos},index=list(df_escolha_filtrados.columns.str[0:-3]))
    fig = px.pie(df_pesos, values = pesos, names = df_pesos.index, title = '<b>Composição<b> da carteira ATUAL')
    fig.update_traces(textposition = 'inside', textinfo= 'percent+label', hole = .6, pull= [0,0,0,0,0], showlegend = False)
    fig.update_layout(title_x=0.2, width = 400, height = 400, margin = dict(l=0, r=0, t=70, b=1))
    st.write(fig)
    return(fig)
#fim da funcao de grafico dos pesos-------------------------------------

#funcao para calcular o ibov-------------------------------------------
@st.cache_data(ttl=24*60*60)  # <<< 24 horas
def calcula_ibov(inicio):
    #benchmark para comparacao:
    ibov = yf.download('^BVSP', start = inicio)['Close']
    ibov.columns.name = None
    ibov.rename(columns={'index': 'Date', '^BVSP': 'ibov'}, inplace=True)
    #ibov.rename('ibov', inplace = True)  
    ibov = pd.DataFrame(ibov)
    ibov_retornos = ibov.pct_change().dropna()
    ibov_retornos_acm = round(((1+ibov_retornos).cumprod() * 100) - 100, 2) #retorno_acm_ibov
    return (ibov_retornos_acm,ibov_retornos)
#fim da funcao de calculo do ibov

#comparativo carteira x benchmark:--------------------------------------
def compara(carteira, ibov_retornos_acm):    
    carteira.index = pd.to_datetime(carteira.index)
    df_compara = pd.merge(carteira['retorno carteira'], ibov_retornos_acm, how = 'inner', on = 'Date')
    df_compara.rename(columns= {'retorno carteira': 'Carteira', 'ibov': 'Ibov'}, inplace = True)
    return(df_compara)
#-----------------------------------------------------------------------

# grafico barras comparativo carteira x benchmark
def grafico_barras(df_compara):
    ultimo = pd.DataFrame(df_compara.iloc[-1])
    ultimo = ultimo.T

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ultimo.index, y=ultimo.Carteira, marker_color='darkblue', name = 'Carteira'))
    fig.add_trace(go.Bar(x=ultimo.index, y=ultimo.Ibov, marker_color='orangered', name = 'Ibov'))
    fig.update_layout(title_text='Comparativo',template='simple_white',title_x=0.5, yaxis_title ='<b>Retorno em %</b>',xaxis_title ='<b>Acumulado</b>', width = 500, height = 500)
    st.write(fig)
    return(fig)
#fim funcao de grafico barras


#grafico comparativo carteira x benchmark:----------------------------
def grafico_carteira_benchmark(df_compara):
    fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(name = 'Carteira',  x = df_compara.index, y = df_compara['Carteira']), row=1, col=1)
    fig.add_trace(go.Scatter(name = 'Ibov', x = df_compara.index, y = df_compara['Ibov']), row=1, col=1)
    fig.update_layout(title_text='Carteira BDRs <b>x</b> Benchmark',template='simple_white',yaxis_title ='<b>Retorno em %</b>',xaxis_title ='<b>Período</b>')
    st.plotly_chart(fig, use_container_width=True)
    return(fig)
#-------------------------------------------------------------------

#funcao de otimizacao de portfolio markowitz
def otimiza_markowitz(df_escolha_filtrados,df_escolha_filtrados_retorno,total_ativos):
    acoes_escolhidas = df_escolha_filtrados.columns.values #tickes para formar a carteira
    df_escolha_filtrados_retorno_medio = df_escolha_filtrados_retorno.mean() #meida dos retornos diarios dos ativos escolhidos
    df_escolha_filtrados_retorno_medio_anual = df_escolha_filtrados_retorno_medio * 252
    covarianca = df_escolha_filtrados_retorno.cov()
    covarianca_anual = covarianca * 252
    correlacao = df_escolha_filtrados_retorno.corr()

    numero_de_carteiras = 10000
    #lista  para popular os dados
    lista_retornos      = []
    lista_volatilidade  = []
    lista_pesos         = []
    lista_sharpe_ratio  = []
    for i in range(numero_de_carteiras):
        peso    = np.random.random(total_ativos)    #criando uma configuracao aleatoria de pesos
        peso   /= np.sum(peso)                      #para o vetor de pesos nao ter uma soma maior que 1 ou 100% normalizado
        lista_pesos.append(peso)                    #apendamos a lista de pesos com os pesos aleatorios que encontramos

        retorno_esperado = np.dot(peso, df_escolha_filtrados_retorno_medio_anual) #vamos calcular o retorno do portfolio ao ano
        lista_retornos.append(retorno_esperado)                                   #apendamos os retornos calculados na lista de retornos

        volatilidade = np.sqrt(np.dot(peso.T, np.dot(covarianca_anual, peso)))    #vamos calcular a volatilidade do portfolio ao ano
        lista_volatilidade.append(volatilidade)                                   #apendamos as volatilidades calculadas na lista de volatilidade

        sharpe_ratio = retorno_esperado/volatilidade # calculamos o sharpe ratio de cada retorno esperado
        lista_sharpe_ratio.append(sharpe_ratio)      #apendamos o sharpe na lista de sharpe para analise

    dic_carteiras       = {'Retorno':lista_retornos, 'Volatilidade': lista_volatilidade, 'Sharpe Ratio': lista_sharpe_ratio}
    for contar, acao in enumerate(acoes_escolhidas): #o enumerate enumera cada item da lista
        dic_carteiras[acao + ' peso'] = [peso[contar] for peso in lista_pesos] #p/ cada peso na lista de pesos associe uma coluna p cada acao
                    
    df_portfolios = pd.DataFrame.from_dict(dic_carteiras, orient = 'index')
    df_portfolios = df_portfolios.T
    return(df_portfolios)
#termino da funcao markowitz


#grafico fundamentalista-------------------------------------------------------------------------
def grafico_fundamentalista(resultados):
    
    resultados.index = resultados['Company'] #ajustar o indice do dataframe resultados

    fig = make_subplots(rows=5, cols=2, row_heights=[7,7,7,7,7], column_widths=[7,7],
                        subplot_titles = ('MarketCap: Vlr de Mercado', 'Revenue: Receita', 'beta', ' ebitda', 'dividendRate: taxa dividendos', 'dividendYield: rendimentos dividendos' , 'Gross Profit: Lucro Bruto', 'NetIncome: Resultado Líquido',
                                        'Total Equity: Patrimônio Líquido', 'ROE: Retorno sobre Patrimônio'), shared_xaxes=False)
    fig.add_trace(go.Bar(name = 'MarketCap: Vlr de Mercado'             , x= resultados.index, y= resultados['MarketCap']), row=1, col=1)
    fig.add_trace(go.Bar(name = 'Revenue: Receita'                      , x= resultados.index, y= resultados['Revenue']), row=1, col=2)
    fig.add_trace(go.Bar(name = 'beta'          , x= resultados.index   , y= resultados['beta']), row=2, col=1)
    fig.add_trace(go.Bar(name = 'ebitda'        , x= resultados.index   , y= resultados['ebitda']), row=2, col=2)
    fig.add_trace(go.Bar(name = 'dividendRate: taxa dividendos'         , x= resultados.index, y= resultados['dividendRate']), row=3, col=1)
    fig.add_trace(go.Bar(name = 'dividendYield: rendimentos dividendos' , x= resultados.index, y= resultados['dividendYield']), row=3, col=2)
    fig.add_trace(go.Bar(name = 'Gross Profit: Lucro Bruto'             , x= resultados.index, y= resultados['Gross Profit']), row=4, col=1)
    fig.add_trace(go.Bar(name = 'NetIncome: Resultado Líquido'          , x= resultados.index, y= resultados['NetIncome']), row=4, col=2)
    fig.add_trace(go.Bar(name = 'Total Equity: Patrimônio Líquido'      , x= resultados.index, y= resultados['Total Equity']), row=5, col=1)
    fig.add_trace(go.Bar(name = 'ROE: Retorno sobre Patrimônio'           , x= resultados.index, y= resultados['ROE']), row=5, col=2)

    fig.update_layout(title_text = 'Análise FUNDAMENTALISTA:', title_font=dict(size=24), template = 'plotly_dark', showlegend=False, height=2000, width=1200)
    st.write(fig)
    return(fig)
#fim da funcao de grafico fundamentalista

#funcao de ultimas noticias-------------------------------------
def google_noticias(nova_pesquisa, inicio, fim):
    inicio  = inicio
    fim     = fim

    googlenews = GoogleNews(lang='pt-BR', start= inicio, end= fim )

    lista = nova_pesquisa
    noticias = []

    for i in lista:
        googlenews.clear()
        googlenews.search(i)
        bdr       = i
        google_resultado = googlenews.result()

        dicionario = {'bdr':i}
        try:
            dicionario['titulo'] = google_resultado[0]['title']
        except:
            dicionario = {'titulo' :'sem notícia recente disponível'}
   
        try:
            dicionario['midia'] = google_resultado[0]['media']
        except:
            dicionario = {'midia' :'sem notícia recente disponível'} 

        try:
            dicionario['dia'] = google_resultado[0]['date']
        except:
            dicionario = {'dia' :'sem notícia recente disponível'}

        try:
            link      = google_resultado[0]['link']
            link_real  = link.split('&')[0]
            dicionario['link'] = link_real
        except:
            dicionario = {'link' :'sem notícia recente disponível'}      

        noticias.append(dicionario)

    df_noticias = pd.DataFrame(noticias)
    df_noticias.dropna(inplace = True)
    

    for i, row in df_noticias.iterrows():
        st.write(f':green[Ativo     : ] {row["bdr"]}')
        st.write(f':green[Notícia   : ] {row["titulo"]}')
        st.write(f':green[Fonte     : ] {row["midia"]}')
        st.write(f':green[Quando    : ] {row["dia"]}')
        st.write(f':green[Link      : ] {row["link"]}')
        #st.write(row['link'])
        st.divider()

        #st.error('Não foi possível localizar notícias no momento, tente mais tarde', icon="⚠️")

    return()
#fim da funcao.................................................................


# webscraping dos dados de bdrs do site investnews
url = 'https://investnews.com.br/financas/veja-a-lista-completa-dos-bdrs-disponiveis-para-pessoas-fisicas-na-b3/'

df = load_bdr(url) #chamando a funcao para ir no investnews e pegar os dados dos bdrs
now = datetime.now()
st.text(datetime.strftime(now,'%A') +  ','+ str((datetime.strftime(now,' %d/%B'))))

st.title(':green[Investir em empresas do Exterior?]  YES, YOU CAN!')

#colunas para disposicao na tela dos graficos e resultados
col_explica, col_imagem = st.columns(2)

with col_explica:
    st.subheader('Conheça os BDRS:')
    st.markdown('Uma maneira bastante simples de aplicar em  ativos estrangeiros, os chamados BDRs - Brazilian Depositary Receipts, são certificados que representam ações emitidas por empresas em outros países, mas que são negociados aqui no Brasil')
    st.markdown('O BDR é um certificado emitido por instituições brasileiras que possibilita o acesso às ações das maiores empresas globais, sendo portanto uma alternativa para diversificação de portfólio.')
    st.markdown('Sem a necessidade de mandar dinheiro para o exterior e sem a preocupação com a conversão do câmbio, tributação do IOF e manutenção de contas, os BDRs ganharam muita força no mercado Brasileiro nos últimos tempos')
    st.markdown(f'Exitem :green[{df.shape[0]} BDRs] negociados no Brasil atualmente, segundo o site Investnews')
    st.markdown('Vamos conferir abaixo, os principais dados do mercado financeiro sobre BDRs:')
st.divider()

with col_imagem:
    st.image('https://elceo.com/wp-content/uploads/2021/01/faang-fotoartecl.jpg')
    st.markdown("<p style='text-align: center;'>foto: elceo.com</p>", unsafe_allow_html=True)


#--------------------- inicio do codigo -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5  = st.tabs([' Análise SETORIAL  |','  Análise FUNDAMENTALISTA  |', '  OTIMIZAÇÃO de Carteira  | ', '  Últimas Notícias  | '  ,  '  Sobre   |'])
bdrs = list(df['CÓDIGO'])
bdrs_yf = [i + '.SA' for i in bdrs] #para acrescentar o .sa na lista

hoje = datetime.now() #.strftime('%Y-%m-%d')
minimo = hoje - timedelta(days = 45)

lista_setores = list(df['SETOR'].unique())
lista_setores = sorted(lista_setores)


#ABA SETORIAL------------------------------------------------------------------------------------------------------------
with tab1:
    st.markdown(f"Dados do site InvestNews, apontam que existem na Bolsa de Valores aproximadamente :green[{df['SETOR'].value_counts().shape[0]} Setores] da Economia")

    lista_empresa = list(df.EMPRESA)

    col1, col2, col3, col4 = st.columns([0.40, 0.15, 0.15, 0.30])
    ajuda = st.checkbox('Quero Procurar por empresa:', help = 'Clique aqui, caso queira consultar um SETOR, através do nome da EMPRESA')
    if ajuda:
        with col1:
            pesquisa = st.selectbox('Pesquisa por Empresa', lista_empresa, label_visibility= "hidden")

    #col1, col2, col3, col4 = st.columns([0.40, 0.15, 0.15, 0.30])
    with col1:
        if ajuda == False :
            setor = st.selectbox('Escolha o Setor de atuação dos BDRs...', lista_setores, placeholder='Escolha o Setor de atuação dos BDRs...')
        else:
            setor = df[df['EMPRESA']== pesquisa]['SETOR'].values[0]
        if setor :
            tickers_escolha = df[df['SETOR'] == setor]['CÓDIGO'].values
            #st.write(f'Análise do Setor -   {str(setor)} - Empresas Listadas:')
            tickers_escolha_yf = [i + '.SA' for i in tickers_escolha ]
        
    with col2:
        inicio  = st.date_input('Data inicial da análise'   , value= minimo  , format="YYYY/MM/DD", disabled=False, label_visibility="visible")

    with col3:
        fim     = st.date_input('Data final da análise'     , value= "today", format="YYYY/MM/DD", disabled=False, label_visibility="visible")

    with col4:
        st.write('')
        st.write('')
        gera_analise = st.button('Analisar:')

    if gera_analise:
        botao = True
        df_escolha_filtrados    = cria_df_escolha(tickers_escolha_yf,inicio, fim )
        df_fundamentalista      = cria_df_fundamentalista(df_escolha_filtrados)

        #agora, fazer uma juncao do dataframe que tem a base dos dados da bolsa com o df fundamentalista
        resultados = pd.merge(df, df_fundamentalista, how='inner', on = 'CÓDIGO')
        resultados = (resultados[['CÓDIGO','EMPRESA', 'Company', 'SETOR', 'Sector','PAÍS DE ORIGEM', 'MarketCap', 'Revenue', 'beta', 'dividendRate', 'dividendYield','ebitda', 'Gross Profit', 'NetIncome', 'Total Equity', 'ROE', 'Sumario', 'Website']])
        df_setor = (resultados[['CÓDIGO','EMPRESA', 'Company', 'SETOR', 'Sector','PAÍS DE ORIGEM','Sumario', 'Website']])
        
        st.write(f'Análise do Setor -   {str(setor)} - Empresas Listadas:')
        st.dataframe(df_setor, hide_index = True, use_container_width=True)
  
        df_escolha_filtrados.index = pd.to_datetime(df_escolha_filtrados.index)

        #analise de variacao de preco dos papeis selecionados:
        st.markdown('Fechamento dos últimos 5 dias:')
        st.dataframe(df_escolha_filtrados.tail().sort_index(ascending = False).T.style.format("R$ {:.2f}"))

        
        df_escolha_filtrados_retorno = df_escolha_filtrados.pct_change().dropna()
        df_escolha_filtrado_normalizado = df_escolha_filtrados/df_escolha_filtrados.iloc[0]

        fig = grafico_preco(df_escolha_filtrados)
        fig = grafico_retorno(df_escolha_filtrado_normalizado, resultados)

        #maiores retornos acumulado no periodo em % das empresas do setor:
        acm = {'Retorno':round(((df_escolha_filtrado_normalizado.iloc[-1]*100)-100).sort_values(ascending = False),2)}
        acumulado = pd.DataFrame(acm)
                
        fig = grafico_acumulado(acumulado)
        
#Aba ANALISE FUNDAMENTALISTA----------------------------------------------------------------------------------------------------------
        with tab2:
            fig = grafico_fundamentalista(resultados)


#Aba Otimizacao de carteira----------------------------------------------------------------------------------------------------------
with tab3:
    st.write('Vamos criar uma carteira de BDRs e otimizar o potencial de Retorno:')

    col1, col2 = st.columns(2)
    col_pesos, col_acumulado = st.columns(2)
    grafico = st.empty()
    col_dados, col_dd = st.columns(2)


    with col1:
        menu_carteira = st.radio("Qual CARTEIRA você deseja analisar?",[ 'Carteira SETORIAL', ':orange[Carteira PERSONALIZADA]'],help = 'todas as empresas do setor ou montar a sua própria carteira', index = None, horizontal = True)
        if menu_carteira == ':orange[Carteira PERSONALIZADA]':
            st.markdown(""" <style> span[data-baseweb="tag"] { background-color: orange !important;} </style> """, unsafe_allow_html=True) #cor do menu bdrs
            lista_bdrs = st.multiselect('Selecione os BDRs listados:', bdrs_yf, placeholder = 'Escolha os ativos para continuar...')
            if st.button('Criar carteira:'):
               
                tickers_escolha_yf = lista_bdrs
                df_escolha_filtrados    = cria_df_escolha(tickers_escolha_yf,inicio, fim )
                df_escolha_filtrados.index = pd.to_datetime(df_escolha_filtrados.index)
                df_escolha_filtrados_retorno = df_escolha_filtrados.pct_change().dropna()

                st.session_state.filtrados         = df_escolha_filtrados   # incluir aqui ---------------------------------
                st.session_state.filtrados_retorno = df_escolha_filtrados_retorno


                st.write(f'Análise: {menu_carteira}')
                carteira, pesos, total_ativos = cria_carteira(df_escolha_filtrados_retorno)
                st.session_state.qtdeativos = total_ativos
                
                with col_pesos:
                    fig = grafico_pesos(df_escolha_filtrados, pesos)

                ibov_retornos_acm, ibov_retornos = calcula_ibov(inicio)
                df_compara = compara(carteira, ibov_retornos_acm)

                with col_acumulado:
                    fig = grafico_barras(df_compara)

                with grafico:
                    grafico_carteira_benchmark(df_compara)
                
                calcula_beta_vol(carteira, ibov_retornos,df_escolha_filtrados_retorno, pesos)
                calcula_dd(df_escolha_filtrados,pesos)

                with col_dados:
                    with st.expander('Saiba mais:'):
                        st.write('Confira o resultado diário da carteira e seus ativos:')
                        st.dataframe(carteira.tail(10))
                
                with tab4:
                    st.subheader(':green[Confira aqui as últimas notícias sobre o(s) BDR(s) selecionados:]')
                    st.divider()
                
                    pesquisa = tickers_escolha_yf
                    nova_pesquisa = []

                    for valor in pesquisa:
                        valor = valor.strip('.SA')
                        nova_pesquisa.append(valor)
                    
                    google_noticias(nova_pesquisa, inicio, fim)



#-------------------------------------------------------- se escolher a carteira setorial--------------------------------------
        if menu_carteira == 'Carteira SETORIAL':
            st.write(f'Análise: {menu_carteira}')
            df_escolha_filtrados    = cria_df_escolha(tickers_escolha_yf,inicio, fim )
            df_escolha_filtrados.index = pd.to_datetime(df_escolha_filtrados.index)
            df_escolha_filtrados_retorno = df_escolha_filtrados.pct_change().dropna()

            st.session_state.filtrados          = df_escolha_filtrados   # incluir aqui ---------------------------------
            st.session_state.filtrados_retorno  = df_escolha_filtrados_retorno

            carteira, pesos, total_ativos = cria_carteira(df_escolha_filtrados_retorno)
            st.session_state.qtdeativos = total_ativos
            
            with col_pesos:
                fig = grafico_pesos(df_escolha_filtrados, pesos)
            
            ibov_retornos_acm, ibov_retornos = calcula_ibov(inicio)
            df_compara = compara(carteira, ibov_retornos_acm)

            with col_acumulado:
                fig = grafico_barras(df_compara)

            with grafico:
                grafico_carteira_benchmark(df_compara)

            calcula_beta_vol(carteira, ibov_retornos,df_escolha_filtrados_retorno, pesos)
            calcula_dd(df_escolha_filtrados,pesos)

            with col_dados:
                with st.expander('Saiba mais:'):
                    st.write('Confira o resultado diário da carteira e seus ativos:')
                    st.dataframe(carteira.tail(10))

            with tab4:
                st.subheader(':green[Confira aqui as últimas notícias sobre o(s) BDR(s) selecionados:]')
                st.divider()
                
                pesquisa = tickers_escolha_yf
                nova_pesquisa = []

                for valor in pesquisa:
                  valor = valor.strip('.SA')
                  nova_pesquisa.append(valor)
                
                google_noticias(nova_pesquisa, inicio, fim)



    #--------------------------------------------------------------------------------------------------

    if st.toggle('OTIMIZAR a Carteira:'):
        togle = True
        if menu_carteira == None:
            st.write('Você ainda nao Criou sua carteira de investimentos')
        else:
            container = st.container()  
            with container:            
                    st.markdown('Otimização da Carteira de BDRs escolhida, utilizando o modelo de :green[Otimização de Markowitz:]')
                    st.markdown('Trata-se de um modelo matemático de ajuda na decisão de alocação de portfólio, procurando a melhor combinação de ativos para uma carteira ótima, ou seja, a que melhor relaciona risco x retorno')
                    st.write(f'Otimização da  {menu_carteira}')

# o erro esta aqui: quando o usuario seleciona a carteira personalisada, ele calcula normal, gera os resultados mas nao traz para esta parte
# do codigo o df_escolha_filtrados, o que nao permite executar esta parte do codigo. 
# quando seleciona a carteira setorial, roda normalmente. creio que seja algo relacionado ao if.... e a identacao que nao permite trazer para
# esta parte do codigo mas ainda nao achei a solucao...

                    #chama a funcao para criar o portfolio otimizado por markowitz
                    #df_portfolios = otimiza_markowitz(df_escolha_filtrados, df_escolha_filtrados_retorno, total_ativos)
                    df_portfolios = otimiza_markowitz( st.session_state.filtrados, st.session_state.filtrados_retorno, st.session_state.qtdeativos)

                    #escolher a forma de otimizar a carteira pelo perfil de risco)
                    tipo_otimizacao = st.radio('Escolha o modelo de Otimização, de acordo com seu Perfil de Risco:', ['menor_volatilidade', 'maior_sharpe', 'maior_retorno'], captions = ["Menor oscilação no resultado da carteira", "melhor retorno em relação ao risco", "Maior Retorno possivel da carteira"], horizontal = True) #, index = None
                    
                    if tipo_otimizacao == 'menor_volatilidade':
                        st.write( 'MININA VARIANCIA')
                        
                        #CARTEIRA COM MENOR VOLATILIDADE
                        menor_volatilidade = df_portfolios['Volatilidade'].min()
                        carteira_min_variancia = df_portfolios.loc[df_portfolios['Volatilidade'] == menor_volatilidade]
                        pesos_carteira_menor_volatilidade  = carteira_min_variancia.drop(['Retorno', 'Volatilidade', 'Sharpe Ratio'], axis = 1)
                        pesos_carteira_menor_volatilidade  = pesos_carteira_menor_volatilidade.values

                        escolha = str('pesos_carteira_'+ tipo_otimizacao)
                        st.write('')
                        
                        escolha = eval(escolha) #transformamos com o comando eval o str num array que tem os pesos escolhidos
                        
                        retorno_carteira_markowitz  = st.session_state.filtrados_retorno*(escolha[0])
                        #retorno_carteira_markowitz  = df_escolha_filtrados_retorno*(escolha[0])
                        retorno_carteira_markowitz = retorno_carteira_markowitz.sum(axis = 1)
                        retorno_carteira_markowitz.rename('retorno diario', inplace = True)

                        retorno_carteira_markowitz_acm = pd.DataFrame()
                        retorno_carteira_markowitz_acm = round(((1+ retorno_carteira_markowitz).cumprod() *100) -100, 2)
                        retorno_carteira_markowitz_acm.rename('Carteira Markowitz', inplace = True)

                        retorno_carteira_markowitz_acm.index = pd.to_datetime(retorno_carteira_markowitz_acm.index)

                    elif tipo_otimizacao == 'maior_sharpe':
                        st.write('maior SHARPE')

                        #CARTEIRA COM MAIOR SHARPE
                        maior_sharpe = df_portfolios['Sharpe Ratio'].max()
                        carteira_maior_sharpe = df_portfolios.loc[df_portfolios['Sharpe Ratio'] == maior_sharpe]
                        pesos_carteira_maior_sharpe = carteira_maior_sharpe.drop(['Retorno', 'Volatilidade', 'Sharpe Ratio'], axis = 1)
                        pesos_carteira_maior_sharpe = pesos_carteira_maior_sharpe.values

                        escolha = str('pesos_carteira_'+ tipo_otimizacao)
                        st.write('')
                        
                        escolha = eval(escolha) #transformamos com o comando eval o str num array que tem os pesos escolhidos
                        retorno_carteira_markowitz  = st.session_state.filtrados_retorno*(escolha[0])
                        #retorno_carteira_markowitz  = df_escolha_filtrados_retorno*(escolha[0])
                        retorno_carteira_markowitz = retorno_carteira_markowitz.sum(axis = 1)
                        retorno_carteira_markowitz.rename('retorno diario', inplace = True)

                        retorno_carteira_markowitz_acm = pd.DataFrame()
                        retorno_carteira_markowitz_acm = round(((1+ retorno_carteira_markowitz).cumprod() *100) -100, 2)
                        retorno_carteira_markowitz_acm.rename('Carteira Markowitz', inplace = True)

                        retorno_carteira_markowitz_acm.index = pd.to_datetime(retorno_carteira_markowitz_acm.index)

                    elif tipo_otimizacao == 'maior_retorno':
                        st.write('MAIOR RETORNO')

                        #CARTEIRA COM MAIOR RETORNO
                        maior_retorno = df_portfolios['Retorno'].max()
                        carteira_maior_retorno = df_portfolios.loc[df_portfolios['Retorno'] == maior_retorno]
                        pesos_carteira_maior_retorno = carteira_maior_retorno.drop(['Retorno', 'Volatilidade', 'Sharpe Ratio'], axis = 1)
                        pesos_carteira_maior_retorno = pesos_carteira_maior_retorno.values

                        
                        escolha = str('pesos_carteira_'+ tipo_otimizacao)
                        st.write('')
                        
                        escolha = eval(escolha) #transformamos com o comando eval o str num array que tem os pesos escolhidos
                        retorno_carteira_markowitz  = st.session_state.filtrados_retorno*(escolha[0])
                        #retorno_carteira_markowitz  = df_escolha_filtrados_retorno*(escolha[0])
                        retorno_carteira_markowitz = retorno_carteira_markowitz.sum(axis = 1)
                        retorno_carteira_markowitz.rename('retorno diario', inplace = True)

                        retorno_carteira_markowitz_acm = pd.DataFrame()
                        retorno_carteira_markowitz_acm = round(((1+ retorno_carteira_markowitz).cumprod() *100) -100, 2)
                        retorno_carteira_markowitz_acm.rename('Carteira Markowitz', inplace = True)

                        retorno_carteira_markowitz_acm.index = pd.to_datetime(retorno_carteira_markowitz_acm.index)
                    
                    try:
                        df_compara_final = pd.merge(df_compara, retorno_carteira_markowitz_acm, how = 'inner', on = 'Date')

                        #transformar num dataframe por conta da visualizacao do grafico.
                        df_pesos_carteira_otimizada_escolhida =pd.DataFrame(data={'pesos':escolha[0]},index=list(df_escolha_filtrados.columns.str[0:-3]))  
                
                        col_pesos_otimizado, col_acumulado_otimizado    = st.columns(2) 

                        #grafico dos pesos otimizados com markowitz
                        fig = px.pie(df_pesos_carteira_otimizada_escolhida, values = df_pesos_carteira_otimizada_escolhida['pesos'].values, names = df_pesos_carteira_otimizada_escolhida.index, title = 'Carteira Otimizada - <b>Markowitz')
                        fig.update_traces(textposition = 'inside', textinfo= 'percent+label', hole = .6, pull= [0,0,0,0,0], showlegend = False)
                        fig.update_layout(title_x=0.2, width = 400, height = 400, margin = dict(l=0, r=0, t=70, b=10))
                        
                        with col_pesos_otimizado:
                            st.write(fig)

                        ultimo_comparativo = pd.DataFrame(df_compara_final.iloc[-1])
                        ultimo_comparativo = ultimo_comparativo.T
            
                        #grafico do acumulado otimizado na coluna 2 
                        with col_acumulado_otimizado: 
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=ultimo_comparativo.index, y=ultimo_comparativo['Carteira'], marker_color='darkblue', name = 'Carteira'))
                            fig.add_trace(go.Bar(x=ultimo_comparativo.index, y=ultimo_comparativo['Carteira Markowitz'], marker_color='darkgreen', name = 'Carteira Markowitz'))
                            fig.add_trace(go.Bar(x=ultimo_comparativo.index, y=ultimo_comparativo['Ibov'], marker_color='orangered', name = 'Ibov'))
                            fig.update_layout(title_text='Comparativo',template='simple_white',title_x=0.5, yaxis_title ='<b>Retorno em %</b>',xaxis_title ='<b>Acumulado</b>', width = 500, height = 500)
                            st.write(fig)

                        #grafico desempenho das carteiras e da carteira  markowitz otimizada--------------------------------------------------:
                        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
                        fig.add_trace(go.Scatter(name = 'Carteira Tradicional',  x = df_compara_final.index, y = df_compara_final['Carteira']), row=1, col=1)
                        fig.add_trace(go.Scatter(name = 'Ibov (Benchmark)', x = df_compara_final.index, y = df_compara_final['Ibov']), row=1, col=1)
                        fig.add_trace(go.Scatter(name = 'Carteira <b>Otimizada</b> (MARKOWITZ)', x = df_compara_final.index, y = df_compara_final['Carteira Markowitz']), row=1, col=1)
                        fig.update_layout(title_text='Comparativo: Carteiras BDRs <b>x</b> Benchmark',template='simple_white',yaxis_title ='<b>Retorno em %</b>',xaxis_title ='<b>Período</b>')
                        
                        col_grafico_otimizado   = st.empty()
                        with col_grafico_otimizado:
                            st.plotly_chart(fig, use_container_width = True)

                        #colunas para distribuicao dos graficos de otimizacao de carteira
                        col_dados_otimizado, col_dd_otimizado   = st.columns(2)
                        with col_dados_otimizado:
                            carteira_markowitz_retorno_diario = retorno_carteira_markowitz*100
                            ibov_retornos = ibov_retornos*100
                            beta_carteira_markowitz = pd.merge(carteira_markowitz_retorno_diario, ibov_retornos, how = 'inner', left_index= True, right_index = True)
                            beta_carteira_markowitz.dropna(inplace = True)


                            Y = beta_carteira_markowitz['retorno diario']     #criamos a variavel dependente
                            X = beta_carteira_markowitz['ibov']               #criamos a variavel independente
                            X = sm.add_constant(X)                  #conforme a equacao
                            modelo = sm.OLS(Y,X)                    #metodo do python para o resultado, passa a variavel dependente e depois a independente
                            resultado = modelo.fit()                #criou um objeto dentro do status model
                            # o beta representa a sensibilidade do ativo, com relacao ao risco sistematico no caso o ibovespa
                            alpha_carteira_markowitz = resultado.params[0]
                            beta_carteira_markowitz  = resultado.params[1]
                                
                            st.metric('BETA da Carteira:', value = str((beta_carteira_markowitz).round(2))+ ' %')
                            st.metric('ALPHA da Carteira:', value = str((alpha_carteira_markowitz).round(2))+ ' %')

                            #calculando da volatilidade da carteira:
                            cov_carteira = df_escolha_filtrados_retorno.cov()
                            vol_carteira = np.sqrt(np.dot(escolha[0].T, np.dot(cov_carteira, escolha[0]))) #a raiz quadrada da soma produto de matrizes.
                            vol_ano = vol_carteira * np.sqrt(252) # aqui usamos 252 dias uteis para trazer o resultad para ano
                        
                            st.metric('Volatilidade/Oscilação ANUAL da carteira:', value = str((vol_ano*100).round(2)) + '% ')

                        with col_dd_otimizado:
                            #DRAWDOWN: 2 - carteira otimizada markowitz---------------------------------------------------
                            drawdown_carteira_markowitz = vbt.Portfolio.from_orders(close = df_escolha_filtrados, size= escolha[0],size_type = 'targetpercent', group_by=True, cash_sharing=True)
                            fig = drawdown_carteira_markowitz.plot_underwater().update_layout(title='Drawdown do portfólio Otimizado', height=350, width = 490, yaxis=dict( title='Queda %', tickformat = '.2%'))
                            st.plotly_chart(fig)

                    except: #NameError
                        st.error('Você ainda nao escolheu uma carteira para OTIMIZAR!')



#aba sobre as ultimas noticias------------------------------------------------------------------------------------------------
with tab4:
    st.write('')
    #st.subheader(':green[Confira aqui as últimas notícias sobre o(s) BDR(s) selecionados:]')
    #st.divider()

#estva aqui a funcao de noticias


#Aba Sobre o projeto ----------------------------------------------------------------------------------------------------------
with tab5:
    #import PIL.Image
    #perfil = st.image('FOTO_PERFIL.PNG')
    #perfil = PIL.Image.open("C:/Users/Paola Bitencourt/TCC/FOTO_PERFIL.PNG")
    st.write('Este Projeto foi desenvolvido pelo Aluno Fábio Soares, para conclusão do Curso Python para o Mercado Financeiro, pela Trading com Dados.')
    col1, col2 = st.columns([0.25,0.75])
    with col1:
        st.image('FOTO_PERFIL.PNG', width = 250)
        st.write('✉︎ ofabiosoares@outlook.com')
    with col2:
        st.write('')
        st.write('* Atuo no Mercado Financeito há 30 anos, sendo os últimos 11 anos como Gestor de Agência de Banco de Varejo')
        st.write('* Especialista em Investimentos CEA - Anbima')
        st.write('* Pós Graduado em Gestão de Negócios - IBMEC')
        st.write('* Graduado em Análise de Sistemas pela Estácio')







