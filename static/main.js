$(document).ready(function () {
    const magma = [
        [0, '#7f3b08'],[0.125, '#b35806'],
        [0.25, '#e08214'],[0.375, '#fdb863'],
        [0.5, '#fee0b6'],[0.625, '#d8daeb'],
        [0.75, '#b2abd2'],[0.875, '#8073ac'],
        [1, '#542788']
    ];

    $('.tab-content').removeClass('active').hide();
    $('#tab-meta').addClass('active').show();
    
    // Hide all loading messages when the page first loads
    $('#loading-message-meta, #loading-message-funnel, #loading-message-corr, #sem-loading').hide();

    let lastResult = null;
    let panelistIDs = [];
    let selectedVariable = '';

    let lastMetaData = null;
    let lastSEMResult = null;
    let selectedKpi = $('#sem-variable').val();
    $('#sem-variable').on('change',function (){
        selectedKpi = $(this).val();
        if (lastSEMResult) renderSEM(lastSEMResult);
    });
    
    // ---- Onglets SPA ---- //
    $('.tab-btn').on('click', function () {
        $('.tab-btn').removeClass('active');
        $(this).addClass('active');
        $('.tab-content').hide();
        const tabId = $(this).data('tab');
        $('#' + tabId).show();

        if (lastResult) {
            if (tabId === 'tab-meta') renderMeta(lastResult.meta);
            if (tabId === 'tab-purchase-funnel') renderPurchaseFunnel(lastResult.purchase_funnel);
            if (tabId === 'tab-correlations') renderCorrelationMatrix(lastResult.correlation);
            if (tabId === 'tab-sem') renderSEM(lastResult.sem);
        }
    });
    

    // Initialisation Select2
    $('.select2').select2({
        closeOnSelect: false,
        placeholder: 'Sélectionnez une ou plusieurs options',
        width: 'resolve',
        tags: true
    });


    function renderMeta(metaData) {
        lastMetaData = metaData;
        const fig = JSON.parse(metaData.plot);
        Plotly.newPlot('plot', fig.data, fig.layout);

        console.log(metaData);
        $('#brand-lift-values').html('<p style="text-align:center; color:#888;">Chargement du tableau...</p>');
    }

    function renderPurchaseFunnel(funnelData) {
        console.log("Funnel data received:", funnelData);
        const data = funnelData?.data;
        const brands = $('#brand').val();
        if (!data || Object.keys(data).length === 0) {
            $('#funnel-table').html('<p style="text-align:center; color:#888; font-size:1px;">Aucune donnée disponible.</p>');
            return;
        }
        if (!brands || brands.length === 0) {
            $('#funnel-table').html('<p>Veuillez sélectionner une marque.</p>');
            return;
        }

        // Clear existing funnels
        $('#funnel-table').empty();

        const orderedIndicators = [
        'Aware', 'Prefer','Good opinion', 'Consider', 'Clients', 'Recommend'];

        brands.forEach(brand =>{
            const values = orderedIndicators.map(ind => data[ind]?.[brand] || 0);

            // Compute taux de passage (conversion rates)
            const conversionRates = values.map((v, i) => {
                if (i === 0) return null; // no conversion rate for "Aware"
                const prev = values[i - 1];
                return prev > 0 ? ((v / prev) * 100).toFixed(1) + '%' : '-';
            });

            //create a unique funnel for each brand
            const containerId = `funnel-${brand.replace(/\s+/g, '_')}`;
            $('#funnel-table').append(`
                <div style="margin-bottom:50px;">
                    <h3 style="text-align:center;">Purchase Funnel — ${brand}</h3>
                    <div id="${containerId}" style="height:500px;"></div>
                </div>
        `   );

            // Add conversion info into custom hover text
            const customHoverData = orderedIndicators.map((label, i) => [
                label, 
                values[i]?.toFixed(1) + '%',
                i === 0 ? '' : '<br>Taux de passage: ' + conversionRates[i]
            ]);

            const trace = {
                type: 'funnel',
                y: orderedIndicators,
                x: values,
                texttemplate: "%{x:.1f}%",
                textinfo: "text",
                hovertemplate: 
                    "<b>%{customdata[0]}</b>: %{customdata[1]}%{customdata[2]}<extra></extra>", 
                customdata: customHoverData,
                textposition: "inside",
                hoverlabel: {align: "left"},
                marker: {
                    color: ['#006c3b', '#00864d', '#00a05f', '#00ba71', '#00d483', '#00ee95', '#66ffbc'],
                    line: { color: '#fff', width: 1 }
                }
            };
        
            const layout = {
                title: '',
                margin: { l: 120, r: 50, t: 40, b: 60 },
                width: 700,
                height: 450
            };

            Plotly.newPlot(containerId, [trace], layout,{displayModeBar:false});
        });
    }

    function renderCorrelationMatrix(corrData) {
        if (!corrData.indicators || corrData.indicators.length === 0) {
            $('#correlation-plot').html('<p>Aucune donnée disponible.</p>');
            return;
        }
        const orderedIndicators = [
            'Aware', 'Discover', 'Ad awareness', 'Prefer',
            'Good opinion', 'Consider', 'Clients', 'Recommend'];

        const indicators = orderedIndicators.filter(i => corrData.indicators.includes(i));
        const corrMatrix = corrData.corr_matrix;
        let z = corrMatrix.map((row, i) => row.map((val, j) => (j > i ? null : val)));

        const data = [{
            z: z, x: indicators, y: indicators,
            type: 'heatmap', colorscale: magma,
            zmin: -1, zmax: 1, zmid : 0, 
            colorbar: { title: 'Corrélation', 
                tickvals : [-1, -0.5, 0, 0.5, 1],
                ticktext : ['Faible', '-0.5', '0', '0.5', 'Forte']
            },
            hovertemplate: 'Corrélation entre %{x} et %{y} : %{z:.2f}<extra></extra>',
            showscale: true,
            text: z.map(row => row.map(v => v !== null ? v.toFixed(2) : '')),
            texttemplate: "%{text}",
            //textfont: { color: "black", size: 14 }
        }];

        const layout = {
            margin: { l: 100, r: 40, b: 45, t: 75 },
            xaxis: { side: 'top', automargin: true },
            yaxis: {
                autorange: 'reversed', // 
                tickvals: indicators.slice().reverse(),
                ticktext: indicators.slice().reverse(), 
                automargin: true
            },
            height: 500,
            width: 650
        };

        Plotly.newPlot('correlation-plot', data, layout);
    }

    function renderSEM(semData) {
        console.log("🔍 SEM data received:", semData);
        if (!semData) {
        $('#sem-results').html('<p style="color:red;">Erreur : réponse vide du serveur.</p>');
        return;
        }
        if (semData.error) {
        $('#sem-results').html('<p style="color:red;">Erreur SEM : ' + semData.error + '</p>');
        return;
        }
        if (!semData.results || !Array.isArray(semData.results) || semData.results.length === 0) {
        $('#sem-results').html('<p>Aucun résultat SEM disponible.</p>');
        return;
        }

        // Clear old content
        $('#sem-results').empty();

        // Add legend
        $('#sem-results').append(`
            <div style="margin-bottom: 16px;">
                <strong>Légende :</strong>
                <span style="color:green;">&#9632; Barre Verte : Significatif</span> 
                <span style="color:red;">&#9632; Barre Rouge : Non Significatif</span>
            </div>
            `);

        const kpi = (typeof selectedKpi !== 'undefined' && selectedKpi) ? selectedKpi : 'Aware';
        const norm = s => String(s || '').replace(/\s+/g, '').toLowerCase();

        semData.results.forEach(brandRes => {
            if (brandRes.error|| !brandRes.edges) return;
        
            const brand = brandRes.brand || 'Marque';
            const containerId = `sem-${brand.replace(/\s+/g,'_')}`;
            $('#sem-results').append(`
                <div style = "margin-bottom: 40px;">
                    <h3>SEM — ${brand} (KPI : ${selectedKpi})</h3>
                    <div id="${containerId}" style="width:600px; height:400px;"></div>
                </div>
            `);

            const edges = (brandRes.edges || []).filter(e => norm(e.to) === norm(kpi));
            if (edges.length === 0 ){
                $(`#${containerId}`).html('<p style="color:red;">Aucun effet trouvé pour ce KPI.</p>');
                return; 
            }
            
            const grouped = {};
            edges.forEach(e => {
                const key = e.from;
                if (!grouped[key]) grouped[key] = { sumCoef: 0, count: 0, sigCount: 0 };
                grouped[key].sumCoef += Number(e.coef) || 0;
                grouped[key].count += 1;
                if (String(e.significant).toLowerCase().startsWith('o')) grouped[key].sigCount += 1; // "Oui"
                });
        
            //prepare data for Plotly
            const vars = [];
            const valuesRel = [];
            const colors = [];

             let sumAbs = 0;
            for (const key in grouped) sumAbs += Math.abs(grouped[key].sumCoef / grouped[key].count || 0);

            for (const key in grouped) {
                const avg = grouped[key].sumCoef / grouped[key].count || 0;
                vars.push(key);
                valuesRel.push(sumAbs ? (Math.abs(avg) / sumAbs) * 100 : 0);
                colors.push(grouped[key].sigCount > grouped[key].count / 2 ? 'green' : 'red');
                }

            //Draw Plotly bar chart
            const trace = {
                x: vars,
                y: valuesRel,
                type: 'bar',
                marker: { color: colors }
            };

            const layout = {
                title: `Influences relatives sur ${selectedKpi}`,
                xaxis: { title: 'Variable explicative', automargin: true },
                yaxis: { title: 'Part relative (%)', range: [0, 100], automargin: true },
                margin: { t: 50, b: 70, l: 60, r: 20 },
                showlegend: false
            };

            Plotly.newPlot(containerId, [trace], layout, { responsive: true });
        });
    }

    // ---- Graph principal ---- //
    function updateGraph() {
        if (!selectedVariable) return;
        $('#loading-message-meta').show();

        const granularity = $('#granularity').val();
        const smoothingValue = (granularity === 'mois' || granularity === 'année') ? 0 : $('#smoothing').val();

        const payload = {
            client: $('#client').val(),
            audiences: $('#audiences').val(),
            brand: $('#brand').val(),
            variable: selectedVariable,
            start_date: $('#start_date').val(),
            end_date: $('#end_date').val(),
            granularity: $('#granularity').val(),
            graph_type: $('#graph_type').val(),
            smoothing: smoothingValue
        };
        $.post('/meta_indicators', JSON.stringify(payload), function (res) {
            const fig = JSON.parse(res.plot); //changed meta here
            Plotly.newPlot('plot', fig.data, fig.layout).then(() => {
                const liftDiv = $('#brand-lift-values');
                liftDiv.empty();

                if (res.brand_lift && Object.keys(res.brand_lift).length > 0) { //changed meta here
                    let html = '<p style="font-weight:bold;">Brand Lift et Moyennes par marque :</p>';
                    html += '<table style="margin: 0 auto; border-collapse: collapse; border: 1px solid black;">';
                    html += '<thead><tr>';
                    html += '<th style="border:1px solid black; padding:5px;">Marque</th>';
                    html += '<th style="border:1px solid black; padding:5px;">Brand Lift (points)</th>';
                    html += '<th style="border:1px solid black; padding:5px;">Moyenne répondants (par marque)</th>';
                    html += '<th style="border:1px solid black; padding:5px;">Début (%)</th>';
                    html += '<th style="border:1px solid black; padding:5px;">Fin (%)</th>';
                    html += '</tr></thead><tbody>';

                    Object.entries(res.brand_lift) //changed meta here
                    .sort(([,a], [,b]) => (b ?? -Infinity) - (a ?? -Infinity))
                    .forEach(([brand, lift]) => {
                        const liftText = (lift === null) ? '<i>Donnée insuffisante</i>' : `${lift > 0 ? '+' : ''}${lift.toFixed(2)}`;
                        let avgBrand = '-';
                        if ($('#brand').val().length === 1) {
                            avgBrand = $('#average-respondents').text().match(/Moyenne\s*:\s*([\d.,]+)/);
                            avgBrand = avgBrand ? avgBrand[1] : '-';
                        } else if (res.avg_by_brand && res.avg_by_brand[brand]) { //changed meta here
                            avgBrand = res.avg_by_brand[brand]; //changed meta here
                        }
                        const se = (res.brand_start_end && res.brand_start_end[brand]) || {}; //changed meta here
                        const startVal = (se.start !== undefined && se.start !== null) ? se.start.toFixed(2) + '%' : '-';
                        const endVal = (se.end !== undefined && se.end !== null) ? se.end.toFixed(2) + '%' : '-';
                        html += `<tr>`;
                        html += `<td style="border:1px solid black; padding:5px;">${brand}</td>`;
                        html += `<td style="border:1px solid black; padding:5px;">${liftText}</td>`;
                        html += `<td style="border:1px solid black; padding:5px;">${avgBrand}</td>`;
                        html += `<td style="border:1px solid black; padding:5px;">${startVal}</td>`;
                        html += `<td style="border:1px solid black; padding:5px;">${endVal}</td>`;
                        html += `</tr>`;
                    });

                html += '</tbody></table>';
                liftDiv.html(html);

                liftDiv[0].offsetHeight; // Force reflow for animation
                liftDiv.show();

            } else {
                $('#brand-lift-values').html('<p style="color:#888;">Aucune donnée de Brand Lift disponible.</p>');
            }
            });


            $('#download').off('click').on('click', function () {
                Plotly.downloadImage('plot', {
                    format: 'png',
                    filename: 'graphique_dya',
                    height: 600,
                    width: 1000,
                    scale: 2
                });
            });
            $('#loading-message-meta').hide();
        }, 'json').fail(function () {
            $('#loading-message-meta').hide();
        });
    }

    function updateRespondentsInfo() {
        if (panelistIDs.length && $('#brand').val()) {
            const payload = {
                panelist_ids: panelistIDs,
                brand: $('#brand').val(),
                granularity: $('#granularity').val(),
                selectedVariable: selectedVariable
            };
            $.post('/get_filtered_count', JSON.stringify(payload), function (res) {
                // Pas de texte affiché ici
            });
        }
    }

    function downloadFile(url, filename) {
        const payload = {
            client: $('#client').val(),
            audiences: $('#audiences').val(),
            brand: $('#brand').val(),
            variable: selectedVariable || 'Aware',
            start_date: $('#start_date').val(),
            end_date: $('#end_date').val(),
            granularity: $('#granularity').val(),
            smoothing: $('#smoothing').val()
        };
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(response => response.blob())
            .then(blob => {
                const link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            })
            .catch(error => alert("Erreur lors du téléchargement : " + error));
    }

    $('#graph_type, #smoothing, #granularity').on('change', function () {
        const graphType = $('#graph_type').val();
        const granularity = $('#granularity').val();

        if (!lastMetaData ) {
            updateGraph(); //fallback to full update if no cached data
            return;
        }

        const fig = JSON.parse(lastMetaData.plot); 

        fig.data.forEach(trace => {
            if (graphType === 'barres') {
                trace.type = 'bar';
                delete trace.mode;
                delete trace.line;
            } else {
                trace.type = 'scatter';
            }
        });
        fig.layout.xaxis.tickformat = granularity === 'année' ? '%Y-%m' : '%Y-%m-%d';

        Plotly.react('plot', fig.data, fig.layout);

        /*
        if (selectedVariable) {
            updateRespondentsInfo();
            updateGraph();
        }
        */
        
    });

    $('#agency').on('change', function () {
        $('#client, #audiences, #brand').empty();
        $('#respondent-count, #average-respondents, #brand-lift-values').text('');
        $.post('/get_clients', JSON.stringify({ agency: $(this).val() }), function (res) {
            res.clients.forEach(client => {
                $('#client').append(`<option value="${client}">${client}</option>`);
            });
            $('#client').trigger('change');
        }, 'json');
    });

    $('#category').on('change', function () {
        const cat = $(this).val();
        $('#brand').empty();
        if (!cat) {
            $.get('/get_all_brands', function (res) {
                res.brands.forEach(b => {
                    $('#brand').append(`<option value="${b}">${b}</option>`);
                });
                $('#brand').select2({
                    closeOnSelect: false,
                    placeholder: 'Sélectionnez une ou plusieurs marques',
                    width: 'resolve',
                    tags: true
                });
                $('#brand').trigger('change');
            }, 'json');
        } else {
            $.post('/get_brands_by_category', JSON.stringify({ category: cat }), function (res) {
                $('#brand').append(`<option value="">-- Toutes marques --</option>`);
                res.brands.forEach(b => {
                    $('#brand').append(`<option value="${b}">${b}</option>`);
                });
                $('#brand').select2({
                    closeOnSelect: false,
                    placeholder: 'Sélectionnez une ou plusieurs marques',
                    width: 'resolve',
                    tags: true
                });
                $('#brand').trigger('change');
            }, 'json');
        }
    });

    $('#client').on('change', function () {
        $('#brand').empty();
        $('#respondent-count, #average-respondents, #brand-lift-values').text('');
        if ($('#audiences').hasClass("select2-hidden-accessible")) {
            $('#audiences').select2('destroy');
        }
        $('#audiences').empty();

        $.post('/get_audiences', JSON.stringify({ client: $(this).val() }), function (res) {
            res.audiences.forEach(a => {
                $('#audiences').append(`<option value="${a}">${a}</option>`);
            });

            $('#audiences').select2({
                closeOnSelect: false,
                placeholder: 'Sélectionnez une ou plusieurs options',
                width: 'resolve',
                tags: true
            });

            $('#audiences').trigger('change');
        }, 'json');
    }).trigger('change');

    $('#audiences').on('change', function () {
        const payload = {
            client: $('#client').val(),
            audiences: $('#audiences').val()
        };
        $.post('/get_respondents', JSON.stringify(payload), function (res) {
            $('#respondent-count').text(`${res.count} répondants trouvés.`);
            panelistIDs = res.panelists;
            $.post('/get_brands_variables', JSON.stringify({ panelist_ids: panelistIDs }), function (r) {
                if (!$('#category').val()) {
                    $('#brand').empty();
                    r.brands.forEach(b => {
                        $('#brand').append(`<option value="${b}">${b}</option>`);
                    });
                    $('#brand').select2({
                        closeOnSelect: false,
                        placeholder: 'Sélectionnez une ou plusieurs marques',
                        width: 'resolve',
                        tags: true
                    });
                    $('#brand').trigger('change');
                } else {
                    $('#category').trigger('change');
                }
            }, 'json');
        }, 'json');
    });

    // ---- Variable buttons ---- //
    $('.var-btn').on('click', function () {
        $('.var-btn').removeClass('active');
        $(this).addClass('active');
        selectedVariable = $(this).data('variable');
        updateRespondentsInfo();
        if ($('#tab-meta').is(':visible')) {
            updateGraph();
        }
    });

    // ---- Boutons de téléchargement ---- //
    $('#download-plot-data').on('click', function () {
        downloadFile('/download_plot_data', 'graph_data.xlsx');
    });
    $('#download-intermediate-data').on('click', function () {
        downloadFile('/download_intermediate_data', 'intermediate_data.xlsx');
    });

    // ---- Submit principal ---- //
    $('#submit').on('click', async function (e) {
        e.preventDefault();
        if (!selectedVariable) {
            selectedVariable = 'Aware';
            $('.var-btn').removeClass('active');
            $('.var-btn[data-variable="Aware"]').addClass('active');
        }

        $('#plot, #funnel-table, #correlation-plot, #sem-results').empty();
        $('#loading-message-meta, #loading-message-funnel, #loading-message-corr, #sem-loading').hide();


        const payload = {
            client: $('#client').val(),
            audiences: $('#audiences').val(),
            brand: $('#brand').val(),
            variable: selectedVariable,
            start_date: $('#start_date').val(),
            end_date: $('#end_date').val(),
            granularity: $('#granularity').val(),
            smoothing: $('#smoothing').val(),
            graph_type: $('#graph_type').val(),
            variables: ["Aware", "Discover", "Ad awareness", "Prefer", "Good opinion", "Consider", "Clients", "Recommend"],
            sem_kpi : selectedKpi
        };
        
        try {
        // STEP 1: Meta Indicators
            $('#loading-message-meta').show();
            const metaRes = await $.ajax({
                url: '/meta_indicators',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(payload)
            });
            renderMeta(metaRes);
            updateGraph();
            $('#loading-message-meta').hide();

            // STEP 2: Purchase Funnel
            $('#loading-message-funnel').show();
            const funnelRes = await $.ajax({
                url: '/get_purchase_funnel_data',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(payload)
            });
            renderPurchaseFunnel(funnelRes);
            $('#loading-message-funnel').hide();

            // STEP 3: Correlation
            $('#loading-message-corr').show();
            const corrRes = await $.ajax({
                url: '/get_correlations',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(payload)
            });
            renderCorrelationMatrix(corrRes);
            $('#loading-message-corr').hide();

            // STEP 4: SEM
            $('#sem-loading').show();
            try {
                const semRes = await $.ajax({
                    url: '/get_sem_causal',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify(payload)
                });
                lastSEMResult = semRes;
                renderSEM(semRes);
            } catch (err) {
                console.error('SEM AJAX error:', err);
                $('#sem-results').empty();

                let msg = 'Erreur SEM inconnue.';
                try {
                    const response = err.responseJSON || JSON.parse(err.responseText || '{}');
                    if (response.error) {
                        msg = response.error;
                    } else if (err.status === 400) {
                        msg = 'Pas de données';
                    }

                } catch (e) {
                    msg = 'Pas de données';
                }

                $('#sem-results').html('<p style="color:red;">Erreur SEM : ' + msg + '</p>');
            } finally {
                $('#sem-loading').hide();
            }

        } catch (e) {
            console.error('Rendering error:', e);
        } finally {
                $('#loading-message-meta, #loading-message-funnel, #loading-message-corr, #sem-loading').hide();
            }
    });


    // INIT : Tab 1 par défaut
    $('#tab-meta').show();
    $('#tab-purchase-funnel, #tab-correlations, #tab-sem').hide();
});
