import React from 'react';
import Plot from 'react-plotly.js';
import { benchmarks } from '../../utils/constants';

const BenchmarkComparison = ({ risk, formatPercent }) => {
  // No benchmark selected or data
  if (!risk.benchmark) {
    return null;
  }
  
  const { portfolio, benchmark, comparison } = risk;
  const benchmarkInfo = benchmarks.find(b => b.id === benchmark.id) || {};
  
  return (
    <div className="benchmark-comparison">
      <h3>Benchmark Comparison: {benchmarkInfo.name}</h3>
      
      <div className="comparison-metrics">
        <div className="metric-card">
          <h4>Tracking Error</h4>
          <div className="metric-value">{formatPercent(comparison.tracking_error)}</div>
          <div className="metric-desc">Variation of portfolio returns vs benchmark</div>
        </div>
        
        <div className="metric-card">
          <h4>Information Ratio</h4>
          <div className="metric-value">{comparison.information_ratio.toFixed(2)}</div>
          <div className="metric-desc">Risk-adjusted excess return</div>
        </div>
        
        <div className="metric-card">
          <h4>Beta</h4>
          <div className="metric-value">{comparison.beta.toFixed(2)}</div>
          <div className="metric-desc">Portfolio sensitivity to benchmark</div>
        </div>
        
        <div className="metric-card">
          <h4>Alpha (Annualized)</h4>
          <div className={`metric-value ${comparison.alpha >= 0 ? 'positive' : 'negative'}`}>
            {formatPercent(comparison.alpha)}
          </div>
          <div className="metric-desc">Excess return vs benchmark</div>
        </div>
      </div>
      
      <div className="benchmark-chart">
        <Plot
          data={[
            {
              x: ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Volatility'],
              y: [
                portfolio.var_95_pct_annual, 
                portfolio.var_99_pct_annual, 
                portfolio.cvar_95_pct_annual, 
                portfolio.cvar_99_pct_annual,
                portfolio.volatility
              ],
              type: 'bar',
              name: 'Portfolio',
              marker: { color: '#040084' }
            },
            {
              x: ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Volatility'],
              y: [
                benchmark.var_95_pct_annual, 
                benchmark.var_99_pct_annual, 
                benchmark.cvar_95_pct_annual, 
                benchmark.cvar_99_pct_annual,
                benchmark.volatility
              ],
              type: 'bar',
              name: benchmarkInfo.name,
              marker: { color: benchmarkInfo.color || '#FD7F00' }
            }
          ]}
          layout={{
            title: 'Risk Metrics Comparison (Annualized)',
            barmode: 'group',
            legend: { orientation: 'h', y: -0.2 },
            yaxis: { title: 'Percentage (%)', tickformat: '.1%' },
            margin: { l: 50, r: 30, t: 50, b: 100 }
          }}
          config={{
            responsive: true,
            displayModeBar: false
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
      
      <div className="benchmark-info">
        <p>
          <strong>About this benchmark:</strong> {benchmarkInfo.description}. 
          Comparing your portfolio to an appropriate benchmark helps assess 
          relative performance and risk characteristics.
        </p>
      </div>
    </div>
  );
};

export default BenchmarkComparison;