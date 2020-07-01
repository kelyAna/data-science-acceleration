#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[20]:





# ## Parte 1

# ### _Setup_ da parte 1

# In[21]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[22]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[23]:


sns.distplot(dataframe.normal, label="Normal")
sns.distplot(dataframe.binomial, label="Binomial")
plt.xlabel('Valores da distribuição')
plt.ylabel('Probabilidade dos valores ocorrerem')
plt.title('Amostragem normal X Amostragem binomial')
plt.legend();


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[28]:


def q1():
    diferencas = np.quantile(dataframe.normal, [.25, .5, .75], axis = 0) - np.quantile(dataframe.binomial, [.25, .5, .75], axis = 0)
    
    return tuple([round(x,3) for x in diferencas])
    pass

q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[29]:


# Plotando o  ECDF dos dados da variável normal:
ecdf = ECDF(dataframe.normal)
plt.title('Função de distribuição acumulada empírica (ECDF')
plt.xlabel('Valores da distribuição')
plt.ylabel('Probabilidade dos valores ocorrerem')
plt.scatter(ecdf.x, ecdf.y);


# In[31]:


def q2():
    ecdf = ECDF(dataframe.normal)
    
    average, std = dataframe.normal.mean(),dataframe.normal.std()
    
    return float(round(ecdf(average + std) - ecdf(average - std),3))
    pass
q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# In[32]:


def gap():
    # CDF empírica da variável:
    ecdf = ECDF(dataframe.normal)

    # Média e desvio padrão:
    average, std = dataframe.normal.mean(),dataframe.normal.std()

    # Área acumulada superior menos a área acumulada inferior:
    gap_2 = float(round(ecdf(average + 2*std) - ecdf(average - 2*std),3))
    gap_3 = float(round(ecdf(average + 3*std) - ecdf(average - 3*std),3))
    print(f"{gap_2}\n{gap_3}")
gap()


# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[34]:


def q3():
    m_norm,m_binom = dataframe.mean()
    
    v_norm,v_binom = dataframe.var()
    
    diferencas = (m_binom - m_norm, v_binom - v_norm)
    
    return tuple([round(x,3) for x in diferencas])
    pass

q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# In[36]:


def binomial():
    # Garantido que o resultado gerado seja sempre o mesmo:
    np.random.seed(42)

    # Aumentando o n da distribuição binomial:
    binomial_2 = sct.binom.rvs(100, 0.2, size=100000)

    # Média e variância da distribuição normal:
    m_norm, v_norm = dataframe.normal.mean(),dataframe.normal.var()

    # Média e variância da distribuição binomial:
    m_binom, v_binom = binomial_2.mean(),binomial_2.var()

    # Diferença entre as médias e as variâncias das variáveis binomial e normal:
    return tuple([round(x,3) for x in (m_binom - m_norm,v_binom - v_norm)])

binomial()


# ## Parte 2

# ### _Setup_ da parte 2

# In[38]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[39]:


# Sua análise da parte 2 começa aqui.
stars.head()


# In[40]:


stars.describe()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[42]:


sns.distplot(stars['mean_profile'])


# In[45]:


def q4():
    filtro = stars.loc[stars['target'] == False]['mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(filtro)
    
    ppf = sct.norm.ppf([.8, .9, .95])
    
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    
    return tuple([round(x,3) for x in ecdf([ppf[0], ppf[1], ppf[2]])])
    pass
q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# In[46]:


filter_stars = stars.loc[stars['target'] == False]['mean_profile'] # Filtrando os valores
false_pulsar_mean_profile_standardized = sct.zscore(filter_stars) # Padronizando

sns.distplot(false_pulsar_mean_profile_standardized);


# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[47]:


def q5():
    filtro = stars.loc[stars['target'] == False]['mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(filtro)

    quantis = np.quantile(false_pulsar_mean_profile_standardized, [.25, .5, .75], axis = 0)
    
    ppf = sct.norm.ppf([0.25, 0.5, 0.75])
    return tuple([round(x,3) for x in (quantis - ppf)])
    pass
q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
