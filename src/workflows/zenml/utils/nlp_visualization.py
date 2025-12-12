from typing import Any, Dict, List


def generate_nlp_html_visualization(
    library_status: Dict[str, Any],
    classical_results: Dict[str, Any],
    advanced_results: Dict[str, Any],
    sentiment_results: Dict[str, Any],
    insights: List[str],
) -> str:
    """
    Generate an HTML visualization for NLP pipeline results.
    
    Args:
        library_status: Status of available NLP libraries
        classical_results: Classical NLP analysis findings
        advanced_results: Advanced linguistic features
        sentiment_results: Sentiment analysis outcomes
        insights: Generated insights and recommendations
        
    Returns:
        HTML string ready for HTMLString
    """
    
    def safe_get(data: Dict[str, Any], key: str, default: Any = 0) -> Any:
        """Safely get value from dictionary."""
        return data.get(key, default) if isinstance(data, dict) and "error" not in data else default
    
    def format_status(available: bool) -> str:
        """Format library status."""
        if available:
            return '<span style="color: #38a169; font-weight: bold;">✅ Available</span>'
        return '<span style="color: #e53e3e; font-weight: bold;">❌ Not available</span>'
    
    def format_percentage(value: float, total: float) -> str:
        """Format percentage."""
        if total == 0:
            return "0.0%"
        return f"{(value / total) * 100:.1f}%"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Advanced NLP Analysis Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin-bottom: 40px;
                padding: 30px;
                background: #f7f9fc;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            
            .section h2 {{
                color: #2d3748;
                font-size: 1.8em;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .section h2::before {{
                content: '';
                width: 4px;
                height: 30px;
                background: #667eea;
                border-radius: 2px;
            }}
            
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            .card-label {{
                font-size: 0.9em;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }}
            
            .card-value {{
                font-size: 2em;
                font-weight: bold;
                color: #2d3748;
            }}
            
            .card-value.good {{
                color: #38a169;
            }}
            
            .card-value.bad {{
                color: #e53e3e;
            }}
            
            .card-value.neutral {{
                color: #718096;
            }}
            
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            
            .status-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            
            .status-item strong {{
                display: block;
                margin-bottom: 5px;
                color: #2d3748;
            }}
            
            .insights-list {{
                list-style: none;
                margin-top: 20px;
            }}
            
            .insights-list li {{
                padding: 15px;
                margin-bottom: 10px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            
            .insights-list li::before {{
                content: '💡';
                margin-right: 10px;
            }}
            
            .metric-bar {{
                background: #e2e8f0;
                height: 30px;
                border-radius: 15px;
                overflow: hidden;
                margin-top: 10px;
                position: relative;
            }}
            
            .metric-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: white;
                font-weight: bold;
                font-size: 0.9em;
            }}
            
            .entity-tag {{
                display: inline-block;
                padding: 5px 12px;
                margin: 5px;
                background: #667eea;
                color: white;
                border-radius: 20px;
                font-size: 0.85em;
            }}
            
            .error-box {{
                background: #fed7d7;
                color: #c53030;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #e53e3e;
                margin-top: 20px;
            }}
            
            @media (max-width: 768px) {{
                .header h1 {{
                    font-size: 1.8em;
                }}
                
                .content {{
                    padding: 20px;
                }}
                
                .section {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Advanced NLP Analysis Report</h1>
                <p>Comprehensive Natural Language Processing Pipeline Results</p>
            </div>
            
            <div class="content">
                <!-- Library Status Section -->
                <div class="section">
                    <h2>🔧 Library Status</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <strong>NLTK</strong>
                            {format_status(library_status.get("nltk_available", False))}
                        </div>
                        <div class="status-item">
                            <strong>spaCy</strong>
                            {format_status(library_status.get("spacy_available", False))}
                        </div>
                        <div class="status-item">
                            <strong>TextBlob</strong>
                            {format_status(library_status.get("textblob_available", False))}
                        </div>
                    </div>
                </div>
                
                <!-- Classical Analysis Section -->
                <div class="section">
                    <h2>📊 Classical NLP Analysis</h2>
    """
    
    if "error" not in classical_results:
        total_texts = safe_get(classical_results, "total_texts", 0)
        total_words = safe_get(classical_results, "total_words", 0)
        unique_words = safe_get(classical_results, "unique_words", 0)
        lexical_diversity = safe_get(classical_results, "lexical_diversity", 0)
        sent_dist = classical_results.get("sentiment_distribution", {})
        pos_texts = safe_get(sent_dist, "positive_texts", 0)
        neg_texts = safe_get(sent_dist, "negative_texts", 0)
        neutral_texts = safe_get(sent_dist, "neutral_texts", 0)
        
        html += f"""
                    <div class="grid">
                        <div class="card">
                            <div class="card-label">Total Texts Analyzed</div>
                            <div class="card-value">{total_texts:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Total Words</div>
                            <div class="card-value">{total_words:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Unique Words</div>
                            <div class="card-value">{unique_words:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Lexical Diversity</div>
                            <div class="card-value {'good' if lexical_diversity > 0.5 else 'bad' if lexical_diversity < 0.2 else 'neutral'}">{lexical_diversity:.3f}</div>
                        </div>
                    </div>
                    
                    <h3 style="margin-top: 30px; color: #2d3748;">Sentiment Distribution</h3>
                    <div style="margin-top: 15px;">
                        <div style="margin-bottom: 15px;">
                            <strong>Positive:</strong> {pos_texts} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(pos_texts, total_texts) if total_texts > 0 else '0%'}">
                                    {format_percentage(pos_texts, total_texts) if total_texts > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Negative:</strong> {neg_texts} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(neg_texts, total_texts) if total_texts > 0 else '0%'}">
                                    {format_percentage(neg_texts, total_texts) if total_texts > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Neutral:</strong> {neutral_texts} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(neutral_texts, total_texts) if total_texts > 0 else '0%'}">
                                    {format_percentage(neutral_texts, total_texts) if total_texts > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                    </div>
        """
    else:
        html += f"""
                    <div class="error-box">
                        <strong>Error:</strong> {classical_results.get("error", "Unknown error")}
                    </div>
        """
    
    html += """
                </div>
                
                <!-- Advanced Analysis Section -->
                <div class="section">
                    <h2>🎯 Advanced Linguistic Analysis</h2>
    """
    
    if "error" not in advanced_results:
        total_entities = safe_get(advanced_results, "total_entities", 0)
        unique_entity_types = safe_get(advanced_results, "unique_entity_types", 0)
        entity_dist = advanced_results.get("entity_type_distribution", {})
        sentence_analysis = advanced_results.get("sentence_analysis", {})
        avg_complexity = safe_get(sentence_analysis, "avg_complexity", 0)
        total_sentences = safe_get(sentence_analysis, "total_sentences", 0)
        
        html += f"""
                    <div class="grid">
                        <div class="card">
                            <div class="card-label">Total Entities Found</div>
                            <div class="card-value">{total_entities:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Unique Entity Types</div>
                            <div class="card-value">{unique_entity_types}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Sentences Analyzed</div>
                            <div class="card-value">{total_sentences:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Avg Complexity</div>
                            <div class="card-value">{avg_complexity:.1f}</div>
                        </div>
                    </div>
        """
        
        if entity_dist:
            html += """
                    <h3 style="margin-top: 30px; color: #2d3748;">Entity Type Distribution</h3>
                    <div style="margin-top: 15px;">
            """
            for entity_type, count in sorted(entity_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                html += f'<span class="entity-tag">{entity_type}: {count}</span>'
            html += """
                    </div>
            """
    else:
        html += f"""
                    <div class="error-box">
                        <strong>Error:</strong> {advanced_results.get("error", "Unknown error")}
                    </div>
        """
    
    html += """
                </div>
                
                <!-- Sentiment Analysis Section -->
                <div class="section">
                    <h2>😊 Enhanced Sentiment Analysis</h2>
    """
    
    if "error" not in sentiment_results:
        total_analyzed = safe_get(sentiment_results, "total_analyzed", 0)
        sent_dist = sentiment_results.get("sentiment_distribution", {})
        pos_count = safe_get(sent_dist, "positive", 0)
        neg_count = safe_get(sent_dist, "negative", 0)
        neu_count = safe_get(sent_dist, "neutral", 0)
        conf_metrics = sentiment_results.get("confidence_metrics", {})
        avg_compound = safe_get(conf_metrics, "avg_compound_score", 0)
        textblob_metrics = sentiment_results.get("textblob_metrics", {})
        avg_polarity = safe_get(textblob_metrics, "avg_polarity", 0)
        avg_subjectivity = safe_get(textblob_metrics, "avg_subjectivity", 0)
        
        html += f"""
                    <div class="grid">
                        <div class="card">
                            <div class="card-label">Texts Analyzed</div>
                            <div class="card-value">{total_analyzed:,}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Avg Compound Score</div>
                            <div class="card-value {'good' if avg_compound > 0.1 else 'bad' if avg_compound < -0.1 else 'neutral'}">{avg_compound:.3f}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Avg Polarity</div>
                            <div class="card-value">{avg_polarity:.3f}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Avg Subjectivity</div>
                            <div class="card-value">{avg_subjectivity:.3f}</div>
                        </div>
                    </div>
                    
                    <h3 style="margin-top: 30px; color: #2d3748;">Sentiment Breakdown</h3>
                    <div style="margin-top: 15px;">
                        <div style="margin-bottom: 15px;">
                            <strong>Positive:</strong> {pos_count} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(pos_count, total_analyzed) if total_analyzed > 0 else '0%'}; background: linear-gradient(90deg, #38a169 0%, #48bb78 100%);">
                                    {format_percentage(pos_count, total_analyzed) if total_analyzed > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Negative:</strong> {neg_count} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(neg_count, total_analyzed) if total_analyzed > 0 else '0%'}; background: linear-gradient(90deg, #e53e3e 0%, #fc8181 100%);">
                                    {format_percentage(neg_count, total_analyzed) if total_analyzed > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Neutral:</strong> {neu_count} texts
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {format_percentage(neu_count, total_analyzed) if total_analyzed > 0 else '0%'}; background: linear-gradient(90deg, #718096 0%, #a0aec0 100%);">
                                    {format_percentage(neu_count, total_analyzed) if total_analyzed > 0 else '0%'}
                                </div>
                            </div>
                        </div>
                    </div>
        """
    else:
        html += f"""
                    <div class="error-box">
                        <strong>Error:</strong> {sentiment_results.get("error", "Unknown error")}
                    </div>
        """
    
    html += """
                </div>
                
                <!-- Insights Section -->
                <div class="section">
                    <h2>💡 Insights & Recommendations</h2>
                    <ul class="insights-list">
    """
    
    for insight in insights:
        html += f"<li>{insight}</li>"
    
    html += """
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

