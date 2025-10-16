def get_graph_data(client_val, audiences, brands, selected_variable, start_date, end_date, granularity):
    if selected_variable not in INDICATORS:
        return None, None
    var_list = INDICATORS[selected_variable]

    meta_query = f"""
    SELECT id AS audience_record_id, name 
    FROM france-std-havas-de.converged_france.audience_records_metadata 
    WHERE client = '{client_val}'
    """
    meta = run_query(meta_query)
    selected_ids = meta[meta['name'].isin(audiences)]['audience_record_id'].tolist()
    if not selected_ids:
        return None, None

    ids_str = ', '.join(f"'{id}'" for id in selected_ids)
    panelist_query = f"""
    SELECT DISTINCT panelist_id 
    FROM france-std-havas-de.converged_france.audience_records_panelists 
    WHERE audience_record_id IN ({ids_str})
    """
    panelists = run_query(panelist_query)['panelist_id'].tolist()
    if not panelists:
        return None, None

    p_ids_str = ', '.join(f"'{x}'" for x in panelists)
    var_str = ', '.join(f"'{v}'" for v in var_list)
    brand_str = ', '.join(f"'{b}'" for b in brands)
    query = f"""
    SELECT yougovid, date, variable, value, brand_name
    FROM converged-havas-de.yougov_bi_fr.bi_response 
    WHERE yougovid IN ({p_ids_str})
      AND brand_name = '{brand_str}'
      AND variable IN ({var_str})
    """
    df = run_query(query)
    if df.empty:
        return None, None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if df.empty:
        return None, None

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
        
    # Nouveau : grouper par marque et période
    pivoted['percentage'] = pivoted['composite'] * 100
    df_pct = pivoted.groupby(['brand_name', 'period'])['percentage'].mean().reset_index()
    df_pct.sort_values(['brand_name', 'period'], inplace=True)

    return df, df_pct

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    logging.debug("Payload reçu pour /generate: %s", data)
    client_val = data['client']
    audiences = data['audiences']
    brands = data['brand']
    selected_variable = data['variable']
    start_date = data['start_date']
    end_date = data['end_date']
    graph_type = data['graph_type']
    smoothing = int(data['smoothing'])
    granularity = data['granularity']
    
    if not brands:
        fig = go.Figure()
        fig.update_layout(title='Pas de données', xaxis_title='Date', yaxis_title='Pourcentage')
        return jsonify({'plot': pio.to_json(fig)})
    
    df_intermediate, df_pct = get_graph_data(client_val, audiences, brands, selected_variable,
                                             start_date, end_date, granularity)
    
    if df_pct is None or df_pct.empty:
        fig = go.Figure()
        fig.update_layout(title='Pas de données', xaxis_title='Date', yaxis_title='Pourcentage')
        return jsonify({'plot': pio.to_json(fig)})
    fig = go.Figure
    
    for brand in brands:
        brand_df = df_pct[df_pct['brand_name'] == brand]
        if brand_df.empty:
            continue
        y_data = brand_df['percentage'].rolling(window=smoothing, min_periods=1).mean() if smoothing > 0 else brand_df['percentage']
        
        if graph_type == 'courbe':
            trace = go.Scatter(
                x=brand_df['period'],
                y=y_data,
                mode='lines+markers',
                name=brand
            )
        else:
            trace = go.Bar(
                x=brand_df['period'],
                y=y_data,
                name=brand
            )
        fig.add_trace(trace)
        
    min_date = df_pct['period'].min()
    max_date = df_pct['period'].max()
    
    fig.update_layout(
        title=f"Évolution de {selected_variable}",
        xaxis_title='Date',
        yaxis_title='Pourcentage',
        legend_title='Marques',
        xaxis=dict(
            type='date',
            range=[min_date.isoformat(), max_date.isoformat()],
            tickformat='%Y-%m-%d'
        )
    )

    return jsonify({'plot': pio.to_json(fig)})