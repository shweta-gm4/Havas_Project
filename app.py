
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os
import json
import logging
from io import BytesIO
from google.oauth2.credentials import Credentials


load_dotenv()


# Configurer le logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


       
# Suppression de GOOGLE_APPLICATION_CREDENTIALS s'il est défini
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# Charger les credentials depuis la variable d'environnement
creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if not creds_json:
    raise Exception("La variable d'environnement GOOGLE_CREDENTIALS_JSON n'est pas définie.")
creds_info = json.loads(creds_json)
credentials = Credentials.from_authorized_user_info(
    creds_info, scopes=["https://www.googleapis.com/auth/cloud-platform"])


app = Flask(__name__)

# Initialisation du client BigQuery
client = bigquery.Client(project="france-std-havas-de", credentials=credentials)



def run_query(query):
    logging.debug(f"Exécution de la requête:\n{query}")
    return client.query(query).to_dataframe(create_bqstorage_client=False)

# Mapping des indicateurs composites
INDICATORS = {
    'Aware': ['aided'],
    'Discover': ['adaware', 'wom', 'buzz'],
    'Ad awareness': ['adaware'],
    'Prefer': ['impression', 'quality', 'value'],
    'Good opinion': ['impression'],
    'Consider': ['consider'],
    'Clients': ['current_own'],
    'Recommend': ['reputation', 'recommend', 'satisfaction']
}
PURCHASE_FUNNEL_INDICATORS = ['Aware', 'Prefer', 'Good opinion', 'Consider', 'Clients', 'Recommend']

#Global Funnel Indicators
FUNNEL_INDICATORS = list(INDICATORS.keys())

@app.route('/')
def index():
    query_clients = "SELECT DISTINCT client FROM france-std-havas-de.converged_france.audience_records_metadata"
    clients = run_query(query_clients)['client'].tolist()
    query_agencies = "SELECT DISTINCT agency FROM france-std-havas-de.converged_france.audience_records_metadata"
    agencies = run_query(query_agencies)['agency'].dropna().tolist()
    query_categories = "SELECT DISTINCT category_name FROM converged-havas-de.yougov_bi_fr.bi_response"
    categories = run_query(query_categories)['category_name'].dropna().tolist()
    return render_template('index.html', clients=clients, agencies=agencies, categories=categories)

@app.route('/purchase_funnel')
def purchase_funnel():
    query_clients = "SELECT DISTINCT client FROM france-std-havas-de.converged_france.audience_records_metadata"
    clients = run_query(query_clients)['client'].tolist()
    query_agencies = "SELECT DISTINCT agency FROM france-std-havas-de.converged_france.audience_records_metadata"
    agencies = run_query(query_agencies)['agency'].dropna().tolist()
    query_categories = "SELECT DISTINCT category_name FROM converged-havas-de.yougov_bi_fr.bi_response"
    categories = run_query(query_categories)['category_name'].dropna().tolist()
    return render_template('purchase_funnel.html', clients=clients, agencies=agencies, categories=categories)

@app.route('/correlations')
def correlations():
    query_clients = "SELECT DISTINCT client FROM france-std-havas-de.converged_france.audience_records_metadata"
    clients = run_query(query_clients)['client'].tolist()
    query_agencies = "SELECT DISTINCT agency FROM france-std-havas-de.converged_france.audience_records_metadata"
    agencies = run_query(query_agencies)['agency'].dropna().tolist()
    query_categories = "SELECT DISTINCT category_name FROM converged-havas-de.yougov_bi_fr.bi_response"
    categories = run_query(query_categories)['category_name'].dropna().tolist()
    return render_template('correlations.html', clients=clients, agencies=agencies, categories=categories)

@app.route('/sem')
def sem():
    query_clients = "SELECT DISTINCT client FROM france-std-havas-de.converged_france.audience_records_metadata"
    clients = run_query(query_clients)['client'].tolist()
    query_agencies = "SELECT DISTINCT agency FROM france-std-havas-de.converged_france.audience_records_metadata"
    agencies = run_query(query_agencies)['agency'].dropna().tolist()
    query_categories = "SELECT DISTINCT category_name FROM converged-havas-de.yougov_bi_fr.bi_response"
    categories = run_query(query_categories)['category_name'].dropna().tolist()
    return render_template('sem.html', clients=clients, agencies=agencies, categories=categories)


# ----------- ENDPOINTS AJAX API ----------- #


@app.route('/get_clients', methods=['POST'])
def get_clients():
    data = request.get_json(force=True)
    agency = data.get('agency')
    query = f"""
    SELECT DISTINCT client 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE agency = '{agency}'
    """
    clients = run_query(query)['client'].tolist()
    return jsonify({'clients': clients})

@app.route('/get_audiences', methods=['POST'])
def get_audiences():
    data = request.get_json(force=True)
    client_val = data.get('client')
    query = f"""
    SELECT DISTINCT name 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE client = '{client_val}'
    """
    df = run_query(query)
    return jsonify({'audiences': df['name'].tolist()})

@app.route('/get_respondents', methods=['POST'])
def get_respondents():
    data = request.get_json(force=True)
    client_val = data.get('client')
    audiences = data.get('audiences', [])
    meta_query = f"""
    SELECT id AS audience_record_id, name 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE client = '{client_val}'
    """
    meta = run_query(meta_query)
    selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
    if not selected_ids:
        return jsonify({'count': 0, 'panelists': []})
    ids_str = ', '.join(f"'{id}'" for id in selected_ids)
    panelist_query = f"""
    SELECT DISTINCT panelist_id 
    FROM france-std-havas-de.converged_france.audience_records_panelists 
    WHERE audience_record_id IN ({ids_str})
    """
    df = run_query(panelist_query)
    return jsonify({'count': len(df), 'panelists': df['panelist_id'].tolist()})

@app.route('/get_brands_variables', methods=['POST'])
def get_brands_variables():
    data = request.get_json(force=True)
    panelist_ids = data.get('panelist_ids', [])
    if not panelist_ids:
        return jsonify({'brands': [], 'variables': []})
    ids_str = ', '.join(f"'{x}'" for x in panelist_ids)
    query = f"""
    SELECT DISTINCT brand_name, variable 
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE yougovid IN ({ids_str})
    """
    df = run_query(query)
    return jsonify({
        'brands': sorted(df['brand_name'].dropna().unique().tolist()),
        'variables': sorted(df['variable'].dropna().unique().tolist())
    })

@app.route('/get_brands_by_category', methods=['POST'])
def get_brands_by_category():
    data = request.get_json(force=True)
    category = data.get('category')
    query = f"""
    SELECT DISTINCT brand_name 
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE category_name = '{category}'
    """
    brands = run_query(query)['brand_name'].dropna().tolist()
    return jsonify({'brands': brands})


@app.route('/get_all_brands', methods=['GET'])
def get_all_brands():
    query = "SELECT DISTINCT brand_name FROM converged-havas-de.yougov_bi_fr.bi_response"
    brands = run_query(query)['brand_name'].dropna().tolist()
    return jsonify({'brands': brands})


def calculate_avg_respondents(pivoted):
    """
    Calcule la moyenne du nombre de répondants par période
    à partir du pivoted DataFrame fourni, en suivant la même logique que get_filtered_count.
    """
    unique_pairs = pivoted[['yougovid', 'period']].drop_duplicates()
    respondents_by_period = unique_pairs.groupby('period')['yougovid'].nunique()
    avg_respondents = round(respondents_by_period.mean(), 1) if not respondents_by_period.empty else 0
    return avg_respondents


@app.route('/get_filtered_count', methods=['POST'])
def get_filtered_count():
    data = request.get_json(force=True)
    panelist_ids = data.get('panelist_ids', [])
    brand = data.get('brand')
    if isinstance(brand, str):
        brand = [brand]
    elif not isinstance(brand, list) or not brand:
        return jsonify({'average_by_brand': {}, 'min_respondents': 0})
    
    granularity = data.get('granularity', 'semaine')
    variable = data.get('selectedVariable')
    if not panelist_ids or not brand or not variable or variable not in INDICATORS:
        return jsonify({'average_by_brand': {}, 'min_respondents': 0})
    
    var_list = INDICATORS[variable]
    ids_str = ', '.join(f"'{x}'" for x in panelist_ids)
    var_str = ', '.join(f"'{v}'" for v in var_list)
    brand_str = ', '.join(f"'{b}'" for b in brand)

    query = f"""
    SELECT yougovid, date, variable, value, brand_name
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE yougovid IN ({ids_str})
      AND brand_name IN ({brand_str})
      AND variable IN ({var_str})
    """
    df = run_query(query)
    if df.empty:
        return jsonify({'average_by_brand': {}, 'min_respondents': 0})
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['positive'] = df['value'].apply(lambda x: 1 if str(x) == '1' else 0)
    
    pivoted = df.pivot_table(
        index=['yougovid', 'date', 'brand_name'], 
        columns='variable', 
        values='positive', 
        fill_value=0
    ).reset_index()
    
    pivoted['composite'] = pivoted[var_list].mean(axis=1)
    
    # Appliquer la granularité
    if granularity == 'jour':
        pivoted['period'] = pivoted['date'].dt.date
    elif granularity == 'semaine':
        pivoted['period'] = pivoted['date'].dt.to_period('W').apply(lambda r: r.start_time.date())
    elif granularity == 'mois':
        pivoted['period'] = pivoted['date'].dt.to_period('M').apply(lambda r: r.start_time.date())
    elif granularity == 'année':
        pivoted['period'] = pivoted['date'].dt.to_period('Y').apply(lambda r: r.start_time.date())
    else:
        pivoted['period'] = pivoted['date'].dt.date

    respondents_by_period = pivoted.groupby('period')['yougovid'].nunique()    
    avg_resp_value = calculate_avg_respondents(pivoted)
    min_resp_value = int(respondents_by_period.min()) if not pd.isna(respondents_by_period.min()) else 0
    
    #avg_resp = round(respondents_by_period.mean()) if not respondents_by_period.empty else 0
    #min_resp = int(respondents_by_period.min()) if not respondents_by_period.empty else 0
        
    return jsonify({
        'avg_respondents': avg_resp_value,
        'min_respondents': min_resp_value
    })

def get_graph_data(client_val, audiences, brands, selected_variable, start_date, end_date, granularity):
    if isinstance(brands, str):
        brands = [brands]

    if isinstance(selected_variable, list):
        selected_variable = selected_variable[0]

    if selected_variable not in INDICATORS:
        return None, None, None, None, None, None
    var_list = INDICATORS[selected_variable]

    meta_query = f"""
    SELECT id AS audience_record_id, name 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE client = '{client_val}'
    """
    meta = run_query(meta_query)
    selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
    if not selected_ids:
        return None, None, None, None, None, None

    ids_str = ', '.join(f"'{id}'" for id in selected_ids)
    panelist_query = f"""
    SELECT DISTINCT panelist_id 
    FROM france-std-havas-de.converged_france.audience_records_panelists 
    WHERE audience_record_id IN ({ids_str})
    """
    panelists = run_query(panelist_query)['panelist_id'].tolist()
    if not panelists:
        return None, None, None, None, None, None

    p_ids_str = ', '.join(f"'{x}'" for x in panelists)
    var_str = ', '.join(f"'{v}'" for v in var_list)
    brand_str = ', '.join(f"'{b}'" for b in brands)
    query = f"""
    SELECT yougovid, date, variable, value, brand_name
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE yougovid IN ({p_ids_str})
      AND brand_name IN ({brand_str})
      AND variable IN ({var_str})
    """
    df = run_query(query)
    if df.empty:
        return None, None, None, None, None, None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if df.empty:
        return None, None, None, None, None, None

    df['positive'] = df['value'].apply(lambda x: 1 if str(x) == '1' else 0)
    
    pivoted = df.pivot_table(index=['yougovid', 'date', 'brand_name'], columns='variable', values='positive', fill_value=0).reset_index()
    pivoted['composite'] = pivoted[var_list].mean(axis=1)

    # Appliquer la granularité sur la date
    if granularity == 'jour':
        pivoted['period'] = pivoted['date']
    elif granularity == 'semaine':
        pivoted['period'] = pivoted['date'].dt.to_period('W').apply(lambda r: r.start_time)
    elif granularity == 'mois':
        pivoted['period'] = pivoted['date'].dt.to_period('M').apply(lambda r: r.start_time)
    elif granularity == 'année':
        pivoted['period'] = pivoted['date'].dt.to_period('Y').apply(lambda r: r.start_time)
    else:
        pivoted['period'] = pivoted['date']
        
    pivoted['percentage'] = pivoted['composite'] * 100
    df_pct = pivoted.groupby(['brand_name', 'period'])['percentage'].mean().reset_index()
    df_pct.sort_values(['brand_name', 'period'], inplace=True)
    
    brand_lift = {}
    brand_start_end = {}
    for brand in df_pct['brand_name'].unique():
        brand_data = df_pct[df_pct['brand_name'] == brand].sort_values('period')
        if len(brand_data) >= 1:
            start_val = brand_data['percentage'].iloc[0]
            end_val = brand_data['percentage'].iloc[-1]
            brand_start_end[brand] = {'start': round(start_val, 2), 'end': round(end_val, 2)}
            
            if len(brand_data) >= 2: 
                brand_lift[brand] = round(end_val - start_val, 2)
            else: 
                brand_lift[brand] = None # Pas assez de données pour calculer le lift
        else:
            brand_lift[brand] = None
            brand_start_end[brand] = {'start': None, 'end': None}

    if len(brands) == 1:
        pivoted_filtered = pivoted[pivoted['brand_name'] == brands[0]]
        mean_val = calculate_avg_respondents(pivoted_filtered)
        avg_by_brand = {brands[0]: mean_val}
        avg_respondents_global = mean_val
        min_respondents_global = int(
            pivoted_filtered[['yougovid', 'period']].drop_duplicates()
            .groupby('period')['yougovid'].nunique()
            .min()
        ) if not pivoted_filtered.empty else 0
    else:
        unique_pairs_brand = pivoted[['brand_name', 'yougovid', 'period']].drop_duplicates()
        respondents_per_brand_period = unique_pairs_brand.groupby(['brand_name', 'period'])['yougovid'].nunique()
        avg_by_brand = respondents_per_brand_period.groupby('brand_name').mean().round(1).to_dict()
        # Pour le global, toutes marques confondues
        unique_pairs = pivoted[['yougovid', 'period']].drop_duplicates()
        respondents_by_period = unique_pairs.groupby('period')['yougovid'].nunique()
        avg_respondents_global = round(respondents_by_period.mean(), 1) if not respondents_by_period.empty else 0
        min_respondents_global = int(respondents_by_period.min()) if not respondents_by_period.empty else 0

    
    return df, df_pct, brand_lift, avg_by_brand, avg_respondents_global, min_respondents_global, brand_start_end

def meta_analysis (client_val, audiences, brands, selected_variable, start_date, end_date, granularity,
                  smoothing=0, graph_type="courbe"): 
 
    df_intermediate, df_pct, brand_lift, avg_by_brand, avg_respondents_global, min_respondents_global, brand_start_end = get_graph_data(
    client_val, audiences, brands, selected_variable, start_date, end_date, granularity)

    if df_pct is None or df_pct.empty:
        fig = go.Figure()
        fig.update_layout(title='Pas de données', xaxis_title='Date', yaxis_title='Pourcentage')
        return ({'plot': pio.to_json(fig)})
    
    df_pct['period'] = pd.to_datetime(df_pct['period'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')

    fig = go.Figure()
    for idx, brand in enumerate(brands):
        brand_df = df_pct[df_pct['brand_name'] == brand]
        if brand_df.empty:
            continue
        if granularity in ['mois', 'année'] or smoothing == 0:
            y_data = brand_df['percentage']
        else:
            y_data = brand_df['percentage'].rolling(window=smoothing, min_periods=1).mean() if smoothing > 0 else brand_df['percentage']

        trace = go.Scatter(
            x=brand_df['period'],
            y=y_data,
            mode='lines+markers',
            name=brand, 
            line_shape = 'spline'
        ) if graph_type == 'courbe' else go.Bar(
            x=brand_df['period'],
            y=y_data,
            name=brand
        )
        fig.add_trace(trace)

        ## Add start and end labels
        if len(brand_df) >= 2: 
            start_x, start_y = brand_df['period'].iloc[0], y_data.iloc[0]
            end_x, end_y = brand_df['period'].iloc[-1], y_data.iloc[-1]

            #slight vertical offset to prevent overlap with markers
            y_offset = (idx - len(brands)/2) * 0.3

            # Dynamic label placement 
            start_pos = 'bottom center'
            end_pos = 'top center' if end_y >= start_y else 'bottom center'
            
            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y + y_offset, end_y + y_offset],
                mode='markers+text',
                text=[f"{start_y:.1f}%", f"{end_y:.1f}%"],
                textposition=[start_pos, end_pos],
                textfont=dict(color='black', size=11),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"Évolution de {selected_variable}",
        xaxis_title='Date',
        yaxis_title='Pourcentage',
        legend_title='Marques',
        margin = dict(l=80, r=80, t=80, b=80),
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d', 
            showgrid=True, 
            zeroline=True
        ), 
        yaxis=dict(showgrid=True),
        template='plotly_white',
    )

    #df_pct['period'] = pd.to_datetime(df_pct['period'])

    return {
        'plot': pio.to_json(fig),
        'brand_lift':brand_lift,
        'avg_by_brand': avg_by_brand,
        'brand_start_end': brand_start_end,
        'avg_respondents': avg_respondents_global,
        'min_respondents': min_respondents_global
    }

@app.route('/meta_indicators', methods=['POST'])
def meta_indicators(): 
    data = request.get_json(force=True)
    logging.debug("Payload reçu pour /generate: %s", data)

    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    selected_variable = data['variable']
    start_date = data['start_date']
    end_date = data['end_date']
    #graph_type = data['graph_type']
    graph_type = data.get('graph_type', 'courbe')
    #smoothing = int(data['smoothing'])
    smoothing = int(data.get('smoothing', 0))
    #granularity = data['granularity']
    granularity = data.get('granularity', 'semaine')

    return jsonify(meta_analysis (client_val, audiences, brands, selected_variable, start_date, end_date, granularity, smoothing, graph_type))          

@app.route('/download_plot_data', methods=['POST'])
def download_plot_data():
    data = request.get_json(force=True)
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    selected_variable = data['variable']
    start_date = data['start_date']
    end_date = data['end_date']
    granularity = data['granularity']
    smoothing = int(data['smoothing'])

    _, df_pct, _, _, _, _, _ = get_graph_data(client_val, audiences, brands, selected_variable, start_date, end_date, granularity)
    if df_pct is None or df_pct.empty:
        return jsonify({'error': 'Pas de données'}), 400

    df_pct['smoothed'] = df_pct['percentage'].rolling(window=smoothing, min_periods=1).mean() if smoothing > 0 else df_pct['percentage']

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_pct.to_excel(writer, index=False, sheet_name='Graph Data')
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition": "attachment; filename=graph_data.xlsx"}
    )

@app.route('/download_intermediate_data', methods=['POST'])
def download_intermediate_data():
    data = request.get_json(force=True)
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    selected_variable = data['variable']
    start_date = data['start_date']
    end_date = data['end_date']
    granularity = data['granularity']

    df_intermediate, _, _, _, _, _, _ = get_graph_data(client_val, audiences, brands, selected_variable, start_date, end_date, granularity)
    if df_intermediate is None or df_intermediate.empty:
        return jsonify({'error': 'Pas de données'}), 400

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_intermediate.to_excel(writer, index=False, sheet_name='Intermediate Data')
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition": "attachment; filename=intermediate_data.xlsx"}
    )

def purchase_funnel_analysis (client_val, audiences, brands, start_date, end_date, granularity):
    if not brands:
        return {'data':{}}
    result = {}

    for indicator in PURCHASE_FUNNEL_INDICATORS: 
        variables = INDICATORS[indicator]
        var_str = ', '.join(f"'{v}'" for v in variables)
        brand_str = ', '.join(f"'{b}'" for b in brands)

        # Filtrer les audiences
        meta_query = f"""
        SELECT id AS audience_record_id, name
        FROM france-std-havas-de.converged_france.audience_records_metadata 
        WHERE client = '{client_val}'
        """
        meta = run_query(meta_query)
        selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
        if not selected_ids:
            continue
        ids_str = ', '.join(f"'{id}'" for id in selected_ids)

        panelist_query = f"""
        SELECT DISTINCT panelist_id 
        FROM france-std-havas-de.converged_france.audience_records_panelists 
        WHERE audience_record_id IN ({ids_str})
        """
        panelists = run_query(panelist_query)['panelist_id'].tolist()
        if not panelists:
            continue

        p_ids_str = ', '.join(f"'{p}'" for p in panelists)

        query = f"""
        SELECT yougovid, date, brand_name, variable, value
        FROM converged-havas-de.yougov_bi_fr.bi_response 
        WHERE yougovid IN ({p_ids_str})
          AND brand_name IN ({brand_str})
          AND variable IN ({var_str})
        """
        df = run_query(query)
        if df.empty:
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if df.empty:
            continue

        df['positive'] = df['value'].apply(lambda x: 1 if str(x) == '1' else 0)

        pivoted = df.pivot_table(index=['yougovid', 'date', 'brand_name'], columns='variable', values='positive', fill_value=0).reset_index()
        pivoted['composite'] = pivoted[variables].mean(axis=1)

        
        # Appliquer la granularité
        if granularity == 'jour':
            pivoted['period'] = pivoted['date'].dt.date
        elif granularity == 'semaine':
            pivoted['period'] = pivoted['date'].dt.to_period('W').apply(lambda r: r.start_time.date())
        elif granularity == 'mois':
            pivoted['period'] = pivoted['date'].dt.to_period('M').apply(lambda r: r.start_time.date())
        elif granularity == 'année':
            pivoted['period'] = pivoted['date'].dt.to_period('Y').apply(lambda r: r.start_time.date())
        else:
            pivoted['period'] = pivoted['date'].dt.date

        avg_period = pivoted.groupby(['brand_name', 'period'])['composite'].mean()
        avg_by_brand = avg_period.groupby('brand_name').mean() * 100

        result[indicator] = {brand: round(avg_by_brand.get(brand, 0), 1) for brand in brands}

    return {'data': result}


@app.route('/get_purchase_funnel_data', methods=['POST'])
def get_purchase_funnel_data():
    data = request.get_json(force=True)
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    start_date = data['start_date']
    end_date = data['end_date']
    granularity = data['granularity']

    result = purchase_funnel_analysis(client_val, audiences, brands, start_date, end_date, granularity)
    return jsonify(result)


def compute_correlations(df, selected_brands, granularity):

    INDICATORS = {
        'Aware': ['aided'],
        'Discover': ['adaware', 'wom', 'buzz'],
        'Ad awareness': ['adaware'],
        'Prefer': ['impression', 'quality', 'value'],
        'Good opinion': ['impression'],
        'Consider': ['consider'],
        'Clients': ['current_own'],
        'Recommend': ['reputation', 'recommend', 'satisfaction']
    }
    FUNNEL_INDICATORS = list(INDICATORS.keys())

    df['date'] = pd.to_datetime(df['date'])
    if granularity == 'jour':
        df['period'] = df['date'].dt.date
    elif granularity == 'semaine':
        df['period'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time.date())
    elif granularity == 'mois':
        df['period'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time.date())
    elif granularity == 'année':
        df['period'] = df['date'].dt.to_period('Y').apply(lambda r: r.start_time.date())
    else:
        df['period'] = df['date'].dt.date
    
    df = df[df['brand_name'].isin(selected_brands)]
    #df['positive'] = df['value'].astype(int)
    df['positive'] = df['value'].apply(lambda x: 1 if str(x) == '1' else 0)
    pivot = df.groupby(['period', 'variable'])['positive'].mean().unstack(fill_value=0)

    # Construction des macros
    for macro, fines in INDICATORS.items():
        for v in fines:
            if v not in pivot.columns:
                pivot[v] = None
        pivot[macro] = pivot[fines].mean(axis=1)

    # Matrice de corrélation
    pivot_funnel = pivot[FUNNEL_INDICATORS]

    print("\nhead test\n",pivot_funnel.head(10))
    print("\ndescribe test\n",pivot_funnel.describe())
    print("\nlen test\n",len(pivot_funnel))
    print("\nvar test\n",pivot_funnel.var())
    
    corr_matrix = pivot_funnel.corr(method='pearson')
    return corr_matrix[FUNNEL_INDICATORS].loc[FUNNEL_INDICATORS], pivot_funnel


def correlation_analysis (client_val, audiences, brands, start_date, end_date, granularity):
    meta_query = f"""
    SELECT id AS audience_record_id, name 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE client = '{client_val}'
    """
    
    meta = run_query(meta_query)
    selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
    if not selected_ids:
        return ({'indicators': [], 'corr_matrix': []})

    ids_str = ', '.join(f"'{id}'" for id in selected_ids)
    panelist_query = f"""
    SELECT DISTINCT panelist_id 
    FROM france-std-havas-de.converged_france.audience_records_panelists 
    WHERE audience_record_id IN ({ids_str})
    """
    panelists = run_query(panelist_query)['panelist_id'].tolist()
    if not panelists:
        return ({'indicators': [], 'corr_matrix': []})
    
    if not brands:
        return ({'indicators': [], 'corr_matrix': []})

    all_fine_variables = sorted({v for sub in INDICATORS.values() for v in sub})
    if not all_fine_variables:
        return ({'indicators': [], 'corr_matrix': []})

    p_ids_str = ', '.join(f"'{x}'" for x in panelists)
    query = f"""
    SELECT yougovid, date, variable, value, brand_name
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE yougovid IN ({p_ids_str})
      AND brand_name IN ({', '.join(f"'{b}'" for b in brands)})
      AND variable IN ({', '.join(f"'{v}'" for v in all_fine_variables)})
      AND date >= '{start_date}' AND date <= '{end_date}'
    """
    df = run_query(query)
    if df.empty:
        return {'indicators': [], 'corr_matrix': [], 'means': {}, 'pivot': []}
    
    #keep the fixed logical order
    #corr_avg = corr_avg.reindex(index=FUNNEL_INDICATORS, columns=FUNNEL_INDICATORS) 
    
    corr, pivot_funnel = compute_correlations(df, brands, granularity)
    indicators = FUNNEL_INDICATORS
    corr_matrix = corr[FUNNEL_INDICATORS].loc[FUNNEL_INDICATORS].values.tolist()
    macro_values = pivot_funnel.reset_index().to_dict(orient='records')

    #corr_matrix = corr_avg.values.tolist()
    #macro_values = pd.concat(pivot_list).groupby('period').mean().reset_index().to_dict(orient='records')

    return {'indicators': indicators,
        'corr_matrix': corr_matrix,
        'macro_values': macro_values}

@app.route('/get_correlations', methods=['POST'])
def get_correlations():

    data = request.get_json(force=True)
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    start_date = data['start_date']
    end_date = data['end_date']
    granularity = data.get('granularity', 'semaine')

    return jsonify (correlation_analysis(client_val, audiences, brands, start_date, end_date, granularity))

import numpy as np
from semopy import Model
import logging



import traceback
from sklearn.metrics import r2_score  # Assure-toi que sklearn est installé

def sem_analysis (client_val, audiences, brands, variables, start_date, end_date):

    def log(msg):
        app.logger.debug(msg)
        print("[DEBUG SEM]", msg)
        with open("debug_sem.txt", "a", encoding="utf8") as f:
            f.write(str(msg) + "\n") 


    try:
        data = request.get_json(force=True)
        log(f"[DEBUG SEM] Payload reçu : {data}")

        log("\n=== [DEBUG SEM] Analyse causale simple ===")
        log(f"[DEBUG SEM] Variables sélectionnées : {variables}")

        meta_query = f"""
        SELECT id AS audience_record_id, name 
        FROM france-std-havas-de.converged_france.audience_records_metadata 
        WHERE client = '{client_val}'
        """
        meta = run_query(meta_query)
        selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
        if not selected_ids:
            log("[DEBUG SEM] Aucune audience trouvée")
            return ({'error': 'Aucune audience trouvée.'}), 400

        ids_str = ', '.join(f"'{id}'" for id in selected_ids)
        panelist_query = f"""
        SELECT DISTINCT panelist_id 
        FROM france-std-havas-de.converged_france.audience_records_panelists 
        WHERE audience_record_id IN ({ids_str})
        """
        panelists = run_query(panelist_query)['panelist_id'].tolist()
        if not panelists:
            log("[DEBUG SEM] Aucun panelist trouvé")
            return ({'error': 'Aucun panelist trouvé.'}), 400

        p_ids_str = ', '.join(f"'{x}'" for x in panelists)

        all_vars = []
        for v in variables:
            all_vars += INDICATORS.get(v, [])
        var_str = ', '.join(f"'{v}'" for v in all_vars)
        brand_str = ', '.join(f"'{b}'" for b in brands)

        query = f"""
        SELECT yougovid, date, variable, value, brand_name
        FROM converged-havas-de.yougov_bi_fr.bi_response 
        WHERE yougovid IN ({p_ids_str})
          AND brand_name IN ({brand_str})
          AND variable IN ({var_str})
          AND date >= '{start_date}' AND date <= '{end_date}'
        """
        df = run_query(query)
        if df.empty:
            log("[DEBUG SEM] DataFrame vide après requête BQ")
            return ({'error': 'Pas de données.'}), 400

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if df.empty:
            log("[DEBUG SEM] DataFrame vide après filtrage dates")
            return ({'error': 'Pas de données sur la période.'}), 400

        df['positive'] = df['value'].apply(lambda x: 1 if str(x) == '1' else 0)

        pivot = df.pivot_table(
            index=['yougovid', 'brand_name'], 
            columns='variable', 
            values='positive', 
            aggfunc='mean'
        ).reset_index()

        log(f"[DEBUG SEM] Colonnes pivot : {pivot.columns.tolist()}")
        log(f"[DEBUG SEM] Preview pivot:\n{pivot.head()}")

        results = []
        for brand in brands:
            log(f"\n[DEBUG SEM] Analyse SEM pour la marque : {brand}")
            data_brand = pivot[pivot['brand_name'] == brand]
            if data_brand.empty:
                log(f"[DEBUG SEM] Pas de données pour {brand}")
                results.append({'brand': brand, 'indicators': variables, 'edges': [], 'r2': {}, 'error': "Pas de données", 'coef_tables': {}})
                continue
        
            if len(variables) < 2:
                    log(f"[DEBUG SEM] Skipping SEM for {brand} — not enough variables ({len(variables)})")
                    results.append({
                        'brand': brand,
                        'indicators': variables,
                        'edges': [],
                        'r2': {},
                        'error': "Pas assez de variables pour effectuer une analyse SEM.",
                        'coef_tables': {}
                    })
                    continue

            rename_map = {v: v.replace(" ", "") for v in variables}

            data_sem = pd.DataFrame()
            for macro in variables:
                fines = INDICATORS.get(macro, [])
                fines_exist = [v for v in fines if v in data_brand.columns]
                if not fines_exist:
                    data_sem[rename_map[macro]] = np.nan
                else:
                    data_sem[rename_map[macro]] = data_brand[fines_exist].mean(axis=1)


            data_sem = data_sem.dropna()
            log(f"[DEBUG SEM] Nombre lignes après dropna ({brand}): {len(data_sem)}")
            if len(data_sem) < 15:
                log(f"[DEBUG SEM] Trop peu de données pour {brand} ({len(data_sem)} lignes)")
                results.append({'brand': brand, 'indicators': variables, 'edges': [], 'r2': {}, 'error': "Trop peu de données", 'coef_tables': {}})
                continue

            brand_edges = []
            brand_r2 = {}
            coef_tables = {}
            error = ""

            for y_var in variables:
                y_var_clean = rename_map[y_var]
                x_vars = [rename_map[v] for v in variables if v != y_var]
                if not x_vars : 
                    log(f"[DEBUG SEM] Skipping {y_var_clean} — no predictors available.")
                    continue
                formula = f"{y_var_clean} ~ " + " + ".join(x_vars)
                log(f"[DEBUG SEM] Formule SEM pour {brand} - dépendante {y_var_clean}:\n{formula}")

                try:
                    model = Model(formula)
                    model.fit(data_sem)
                    est = model.inspect()
                    # Construire le tableau des coefficients et p-values pour la variable cible
                    coef_table = {}
                    if set(['op', 'lval', 'rval', 'Estimate', 'p-value']).issubset(est.columns):
                        for _, row in est.iterrows():
                            if row['op'] == '~' and row['lval'] == y_var_clean:
                                coef_table[row['rval']] = {
                                    'coef': float(row['Estimate']),
                                    'pvalue': float(row['p-value']),
                                    'significant': row['p-value'] < 0.05
                                }
                                brand_edges.append({
                                    'from': row['rval'],
                                    'to': row['lval'],
                                    'coef': float(row['Estimate']),
                                    'pvalue': float(row['p-value']),
                                    'significant': "Oui" if row['p-value'] < 0.05 else "Non"
                                })

                    coef_tables[y_var] = coef_table

                    y_true = data_sem[y_var_clean].values
                    y_pred = model.predict(data_sem)[y_var_clean].values
                    brand_r2[y_var] = float(r2_score(y_true, y_pred))

                except Exception as e:
                    tb_str = traceback.format_exc()
                    log(f"[DEBUG SEM] Exception SEM pour {brand} sur variable {y_var_clean}: {str(e)}\n{tb_str}")
                    error = str(e)
                    break

            results.append({'brand': brand, 'indicators': variables, 'edges': brand_edges, 'r2': brand_r2, 'error': error, 'coef_tables': coef_tables})

        log("[DEBUG SEM] Analyse causale terminée.")
        return ({'results': results, 'variables': variables})

    except Exception as e:
        tb_str = traceback.format_exc()
        app.logger.error(f"Erreur lors du calcul SEM : {str(e)}\n{tb_str}")
        print(f"Erreur lors du calcul SEM : {str(e)}\n{tb_str}")
        return ({'error': f"Erreur lors du calcul SEM : {str(e)}"}), 500

@app.route('/get_sem_causal', methods=['POST'])
def get_sem_causal():
    data = request.get_json(force= True)
    
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    if isinstance(brands, str):
        brands = [brands]
    start_date = data['start_date']
    end_date = data['end_date']
    variables = data.get('variables', FUNNEL_INDICATORS)

    return jsonify(sem_analysis(client_val, audiences, brands, variables, start_date, end_date))


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True) 
    logging.debug("Payload reçu pour /generate: %s", data)
    
    # Collect user inputs 
    client_val = data.get('client') 
    audiences = data.get('audiences') 
    brands = data.get('brands') or data.get('brand') 
    start_date = data.get('start_date') 
    end_date = data.get('end_date') 
    graph_type = data.get('graph_type', 'courbe')
    smoothing = int(data.get('smoothing', 0))
    granularity = data.get('granularity', 'semaine') 
    
    #One variable for Meta
    selected_variable = data.get ('variable', 'Aware')
    #List of variables for SEM
    variables = data.get('variables', list(INDICATORS.keys()))
    
    sem_result = sem_analysis(client_val, audiences, brands, variables, start_date, end_date)

    return jsonify({
        'meta': meta_analysis(client_val, audiences, brands, selected_variable, start_date, end_date, granularity, smoothing, graph_type), 
        'purchase_funnel': purchase_funnel_analysis(client_val, audiences, brands, start_date, end_date, granularity), 
        'correlation': correlation_analysis (client_val, audiences, brands, start_date, end_date, granularity), 
        'sem': sem_result
        })

if __name__ == '__main__':
    app.run(debug=True)
