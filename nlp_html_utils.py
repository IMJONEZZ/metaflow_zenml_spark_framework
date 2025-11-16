"""HTML generation utilities for NLP pipeline visualization."""

from typing import List


def generate_nlp_html_report(
    texts: List[str],
    sample_count: int,
    avg_length: int,
    total_chars: int,
    positive_samples: int,
    negative_samples: int,
    neutral_samples: int,
    pos_pct: float,
    neg_pct: float,
    neutral_pct: float,
) -> str:
    """Generate HTML report for NLP analysis results.

    Args:
        texts: List of text samples to display
        sample_count: Total number of samples
        avg_length: Average length of texts in characters
        total_chars: Total number of characters
        positive_samples: Number of positive sentiment samples
        negative_samples: Number of negative sentiment samples
        neutral_samples: Number of neutral sentiment samples
        pos_pct: Percentage of positive samples
        neg_pct: Percentage of negative samples
        neutral_pct: Percentage of neutral samples

    Returns:
        Complete HTML string with the visualization
    """
    # Build text samples HTML
    text_samples_html = ""
    for i, text in enumerate(texts, 1):
        text_samples_html += f'                <li class="text-item"><strong>Sample {i}:</strong> {text}</li>\n'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Analysis Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .text-list {{
            list-style: none;
            padding: 0;
        }}
        .text-item {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .text-item:nth-child(even) {{
            border-left-color: #764ba2;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .sentiment-chart {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }}
        .sentiment-bar {{
            flex: 1;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }}
        .sentiment-fill {{
            width: 100%;
            border-radius: 10px 10px 0 0;
            transition: height 0.3s ease;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding-top: 10px;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .positive {{
            background: linear-gradient(180deg, #4CAF50 0%, #45a049 100%);
        }}
        .negative {{
            background: linear-gradient(180deg, #f44336 0%, #da190b 100%);
        }}
        .neutral {{
            background: linear-gradient(180deg, #ff9800 0%, #f57c00 100%);
        }}
        .sentiment-label {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #333;
        }}
        .percentage {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 NLP Analysis Results</h1>
        <p class="subtitle">Comprehensive Text Analysis and Sentiment Distribution</p>

        <div class="section">
            <h2>📝 Text Samples</h2>
            <ul class="text-list">
{text_samples_html}            </ul>
        </div>

        <div class="section">
            <h2>📊 Analysis Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{sample_count}</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_length}</div>
                    <div class="stat-label">Avg Length (chars)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_chars}</div>
                    <div class="stat-label">Total Characters</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>😊 Sentiment Distribution</h2>
            <div class="sentiment-chart">
                <div style="flex: 1;">
                    <div class="sentiment-bar">
                        <div class="sentiment-fill positive" style="height: {pos_pct}%;">
                            <span>{positive_samples}</span>
                        </div>
                    </div>
                    <div class="sentiment-label">
                        Positive<br>
                        <span class="percentage">{pos_pct:.1f}%</span>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="sentiment-bar">
                        <div class="sentiment-fill negative" style="height: {neg_pct}%;">
                            <span>{negative_samples}</span>
                        </div>
                    </div>
                    <div class="sentiment-label">
                        Negative<br>
                        <span class="percentage">{neg_pct:.1f}%</span>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="sentiment-bar">
                        <div class="sentiment-fill neutral" style="height: {neutral_pct}%;">
                            <span>{neutral_samples}</span>
                        </div>
                    </div>
                    <div class="sentiment-label">
                        Neutral<br>
                        <span class="percentage">{neutral_pct:.1f}%</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    return html_content

