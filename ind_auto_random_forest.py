#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_percentage_error

#%%
files = Path.cwd().parent / Path('files')
Path.parent
# Imports - data url: http://comexstat.mdic.gov.br/pt/geral/106042
file_imports = files / 'IMP_2000_2023_auto.csv'
# Exports - data url: http://comexstat.mdic.gov.br/pt/geral/106226 
file_exports = files / 'EXP_2000_2023_auto.csv'
# %%
# Importing the data
df_imp = pd.read_csv(file_imports, sep=';')
# Exports
df_exp = pd.read_csv(file_exports, sep=';')
#%%
df = pd.merge(
    df_imp, df_exp
    , on=[
        'Ano'
        , 'Mês'
        , 'Código ISIC Grupo'
        , 'Descrição ISIC Grupo'
        ]
    , how='outer'
    , suffixes=('_imp', '_exp')
    )
#%%
del df_imp, df_exp
#%%
# Renaming the columns
df = df.rename(columns={
    'Ano': 'ano',
    'Mês': 'mes',
    'Código ISIC Grupo': 'isic_grupo',
    'Descrição ISIC Grupo': 'isic_descricao',
    'Valor FOB (US$)_imp': 'fob_imp',
    'Quilograma Líquido_imp': 'kg_liquido_imp',
    'Valor FOB (US$)_exp': 'fob_exp',
    'Quilograma Líquido_exp': 'kg_liquido_exp'
})

# %%
# Processing the data
df.dropna(inplace=True)
df['ano'] = df['ano'].astype(int)
df['mes'] = df['mes'].astype(int)
#%%
df = df.groupby(['ano', 'mes', 'isic_descricao']).sum().reset_index()
#%%
# Splitting the data into training and testing
df_train = df[df['ano'] < 2023]
df_test = df[df['ano'] == 2023]

#%%
# Separating the features
X_train = df_train[['ano', 'mes', 'isic_descricao']]
X_test = df_test[['ano', 'mes', 'isic_descricao']]
#%%
# Separating the target variables
y_train = df_train[['fob_imp', 'kg_liquido_imp', 'fob_exp', 'kg_liquido_exp']].values
#%%
# Preprocessing the data
categorical_features = ['isic_descricao']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
#%%
# Model definition
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
#%%
# Model training
model.fit(X_train, y_train)
#%%
# Model prediction
y_pred = model.predict(X_test)
#%%
# Separating the predictions
y_pred_fob_imp, y_pred_kg_liq_imp = y_pred[:, 0], y_pred[:, 1]
y_pred_fob_exp, y_pred_kg_liq_exp = y_pred[:, 2], y_pred[:, 3]
#%%
# Creating a DataFrame for comparison
comparison_df = pd.DataFrame({
    'ISIC': df_test['isic_descricao'],
    'Mês': df_test['mes'],
    'Real FOB Imports': df_test['fob_imp'],
    'Predicted FOB Imports': y_pred_fob_imp,
    'Real KG Liq Imports': df_test['kg_liquido_imp'],
    'Predicted KG Liq Imports': y_pred_kg_liq_imp,
    'Real FOB Exports': df_test['fob_exp'],
    'Predicted FOB Exports': y_pred_fob_exp,
    'Real KG Liq Exports': df_test['kg_liquido_exp'],
    'Predicted KG Liq Exports': y_pred_kg_liq_exp
})
#%%
# Sorting the DataFrame
comparison_df.sort_values(by=['ISIC', 'Mês'], inplace=True)
comparison_df['Mês'] = df_test['mes']
unique_isics = comparison_df['ISIC'].unique()
#%%
# Calculating the metrics
mse_fob_imp = mean_squared_error(df_test['fob_imp'], y_pred_fob_imp)
mse_kg_liq_imp = mean_squared_error(df_test['kg_liquido_imp'], y_pred_kg_liq_imp)
r2_fob_imp = round(r2_score(df_test['fob_imp'], y_pred_fob_imp), 2)
r2_kg_liq_imp = round(r2_score(df_test['kg_liquido_imp'], y_pred_kg_liq_imp), 2)
rmsle_fob_imp = round(np.sqrt(mean_squared_log_error(df_test['fob_imp'], y_pred_fob_imp)), 3)
rmsle_kg_liq_imp = round(np.sqrt(mean_squared_log_error(df_test['kg_liquido_imp'], y_pred_kg_liq_imp)), 3)
mape_fob_imp = round(mean_absolute_percentage_error(df_test['fob_imp'], y_pred_fob_imp) * 100, 3)
mape_kg_liq_imp = round(mean_absolute_percentage_error(df_test['kg_liquido_imp'], y_pred_kg_liq_imp) * 100, 3)

print(f"FOB Imports - Mean Squared Error: {mse_fob_imp}")
print(f"KG Liq Imports - Mean Squared Error: {mse_kg_liq_imp}")
print(f"FOB Imports - R^2 Score: {r2_fob_imp}")
print(f"KG Liq Imports - R^2 Score: {r2_kg_liq_imp}")
print(f"FOB Imports - RMSLE: {rmsle_fob_imp}")
print(f"KG Liq Imports - RMSLE: {rmsle_kg_liq_imp}")
print(f"FOB Imports - MAPE: {mape_fob_imp}")
print(f"KG Liq Imports - MAPE: {mape_kg_liq_imp}")


mse_fob_exp = mean_squared_error(df_test['fob_exp'], y_pred_fob_exp)
mse_kg_liq_exp = mean_squared_error(df_test['kg_liquido_exp'], y_pred_kg_liq_exp)
r2_fob_exp = round(r2_score(df_test['fob_exp'], y_pred_fob_exp), 2)
r2_kg_liq_exp = round(r2_score(df_test['kg_liquido_exp'], y_pred_kg_liq_exp), 2)
rmsle_fob_exp = round(np.sqrt(mean_squared_log_error(df_test['fob_exp'], y_pred_fob_exp)), 3)
rmsle_kg_liq_exp = round(np.sqrt(mean_squared_log_error(df_test['kg_liquido_exp'], y_pred_kg_liq_exp)), 3)
mape_fob_exp = round(mean_absolute_percentage_error(df_test['fob_exp'], y_pred_fob_exp) * 100, 3)
mape_kg_liq_exp = round(mean_absolute_percentage_error(df_test['kg_liquido_exp'], y_pred_kg_liq_exp) * 100, 3)

print(f"FOB Exports - Mean Squared Error: {mse_fob_exp}")
print(f"KG Liq Exports - Mean Squared Error: {mse_kg_liq_exp}")
print(f"FOB Exports - R^2 Score: {r2_fob_exp}")
print(f"KG Liq Exports - R^2 Score: {r2_kg_liq_exp}")
print(f"FOB Exports - RMSLE: {rmsle_fob_exp}")
print(f"KG Liq Exports - RMSLE: {rmsle_kg_liq_exp}")
print(f"FOB Exports - MAPE: {mape_fob_exp}")
print(f"KG Liq Exports - MAPE: {mape_kg_liq_exp}")

#%%
# Plotting the results
for isic in unique_isics:
    subset = comparison_df[comparison_df['ISIC'] == isic]
    
    # Definindo a figura para os subplots com duas linhas e duas colunas
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Previsões e Valores Reais para {isic} - 2023', fontsize=16)

    # Plot para Valor FOB de Importações com margem de erro
    sns.lineplot(x='Mês', y='Real FOB Imports', data=subset, marker='o', label='Real FOB Imports', ax=axs[0, 0])
    sns.lineplot(x='Mês', y='Predicted FOB Imports', data=subset, marker='>', label='Predicted FOB Imports', ax=axs[0, 0])
    axs[0, 0].fill_between(subset['Mês'], 
                           subset['Predicted FOB Imports'] * 0.8, 
                           subset['Predicted FOB Imports'] * 1.2, 
                           color='blue', alpha=0.2, label='Margin of Error')
    axs[0, 0].set_title('Valor FOB de Importações')
    axs[0, 0].set_xlabel('Mês')
    axs[0, 0].set_ylabel('Valor FOB (US$)')
    axs[0, 0].legend()

    # Plot para KG Líquido de Importações com margem de erro
    sns.lineplot(x='Mês', y='Real KG Liq Imports', data=subset, marker='o', label='Real KG Liq Imports', ax=axs[0, 1])
    sns.lineplot(x='Mês', y='Predicted KG Liq Imports', data=subset, marker='>', label='Predicted KG Liq Imports', ax=axs[0, 1])
    axs[0, 1].fill_between(subset['Mês'], 
                           subset['Predicted KG Liq Imports'] * 0.8, 
                           subset['Predicted KG Liq Imports'] * 1.2, 
                           color='blue', alpha=0.2, label='Margin of Error')
    axs[0, 1].set_title('KG Líquido de Importações')
    axs[0, 1].set_xlabel('Mês')
    axs[0, 1].set_ylabel('KG Líquido')
    axs[0, 1].legend()

    # Plot para Valor FOB de Exportações com margem de erro
    sns.lineplot(x='Mês', y='Real FOB Exports', data=subset, marker='o', label='Real FOB Exports', ax=axs[1, 0])
    sns.lineplot(x='Mês', y='Predicted FOB Exports', data=subset, marker='>', label='Predicted FOB Exports', ax=axs[1, 0])
    axs[1, 0].fill_between(subset['Mês'], 
                           subset['Predicted FOB Exports'] * 0.8, 
                           subset['Predicted FOB Exports'] * 1.2, 
                           color='blue', alpha=0.2, label='Margin of Error')
    axs[1, 0].set_title('Valor FOB de Exportações')
    axs[1, 0].set_xlabel('Mês')
    axs[1, 0].set_ylabel('Valor FOB (US$)')
    axs[1, 0].legend()

    # Plot para KG Líquido de Exportações com margem de erro
    sns.lineplot(x='Mês', y='Real KG Liq Exports', data=subset, marker='o', label='Real KG Liq Exports', ax=axs[1, 1])
    sns.lineplot(x='Mês', y='Predicted KG Liq Exports', data=subset, marker='>', label='Predicted KG Liq Exports', ax=axs[1, 1])
    axs[1, 1].fill_between(subset['Mês'], 
                           subset['Predicted KG Liq Exports'] * 0.8, 
                           subset['Predicted KG Liq Exports'] * 1.2, 
                           color='blue', alpha=0.2, label='Margin of Error')
    axs[1, 1].set_title('KG Líquido de Exportações')
    axs[1, 1].set_xlabel('Mês')
    axs[1, 1].set_ylabel('KG Líquido')
    axs[1, 1].legend()

    # Ajustando o layout para evitar sobreposição
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Exibindo o gráfico
    plt.show()
plt.close('all')

# %%
