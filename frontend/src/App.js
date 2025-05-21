// At the top of App.js
import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

// Import components
import Sidebar from './components/dashboard/Sidebar';
import Header from './components/dashboard/Header';
import RiskCard from './components/shared/RiskCard';
import MatrixTable from './components/shared/MatrixTable';
import LoadingIndicator from './components/shared/LoadingIndicator';

// Base API URL - can be moved to environment variable
const API_BASE_URL = 'http://localhost:8000/api';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // Increased timeout for long operations
});

// Default CMA assumptions
const defaultCmaAssumptions = {
  assetClasses: {
    "Cash": {
      expectedReturn: 3.5,
      volatility: 0.5,
    },
    "AUS Equities": {
      expectedReturn: 8.2,
      volatility: 16.5,
    },
    "Global Equities": {
      expectedReturn: 7.5,
      volatility: 15.0,
    },
    "Emerging Markets": {
      expectedReturn: 9.0,
      volatility: 20.0,
    },
    "AUS Bonds": {
      expectedReturn: 4.5,
      volatility: 5.0,
    },
    "Global Bonds": {
      expectedReturn: 4.0,
      volatility: 5.5,
    },
    "Property": {
      expectedReturn: 6.5,
      volatility: 12.0,
    },
    "Infrastructure": {
      expectedReturn: 6.0,
      volatility: 10.0,
    }
  },
  mappings: {
    "VAS.AX": "AUS Equities",
    "VGS.AX": "Global Equities",
    "VGE.AX": "Emerging Markets",
    "VAF.AX": "AUS Bonds",
    "VGB.AX": "AUS Bonds",
    "VIF.AX": "Global Bonds",
    "BILL.AX": "Cash",
    "VAP.AX": "Property",
    "VBLD.AX": "Infrastructure",
    "VTS.AX": "Global Equities",
    "VEU.AX": "Global Equities",
    "VDCO.AX": "AUS Bonds",
    "VDGR.AX": "Global Equities",
    "VDHG.AX": "Global Equities",
    "VDBA.AX": "AUS Bonds",
    "VDMF.AX": "Global Equities",
    "IVV": "Global Equities",
    "AAPL": "Global Equities",
    "MSFT": "Global Equities",
    "GOOGL": "Global Equities",
    "VWO": "Emerging Markets",
    "VTI": "Global Equities",
    "BND": "Global Bonds",
    "A200.AX": "AUS Equities",
    "SPDW": "Global Equities",
    "SPEM": "Emerging Markets",
    "AUDUSD=X": "Cash"
  }
};

function App() {
  // State variables
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2023-12-31');
  const [currency, setCurrency] = useState('AUD');
  const [risk, setRisk] = useState(null);
  const [cma, setCma] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [benchmark, setBenchmark] = useState("asx200"); // Default to ASX 200
  const [holdings, setHoldings] = useState([]);
  const [activeTab, setActiveTab] = useState('Risk Metrics');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [loading, setLoading] = useState({
    portfolios: false,
    risk: false,
    cma: false,
    holdings: false,
    upload: false
  });
  const [error, setError] = useState({
    portfolios: null,
    risk: null,
    cma: null,
    holdings: null,
    upload: null,
    assumptions: null
  });
  const [cmaAssumptions, setCmaAssumptions] = useState(() => {
    // Try to get from localStorage first
    const savedAssumptions = localStorage.getItem('cmaAssumptions');
    return savedAssumptions ? JSON.parse(savedAssumptions) : defaultCmaAssumptions;
  });
  const [editingAssumptions, setEditingAssumptions] = useState(false);
  const [assumptionsJson, setAssumptionsJson] = useState('');

  // Save CMA assumptions to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('cmaAssumptions', JSON.stringify(cmaAssumptions));
  }, [cmaAssumptions]);

  // Format assumptions to JSON for editing
  useEffect(() => {
    if (editingAssumptions) {
      setAssumptionsJson(JSON.stringify(cmaAssumptions, null, 2));
    }
  }, [editingAssumptions, cmaAssumptions]);

  // Handle saving edited assumptions
  const handleSaveAssumptions = () => {
    try {
      const parsedJson = JSON.parse(assumptionsJson);
      setCmaAssumptions(parsedJson);
      setEditingAssumptions(false);
      
      // Success message
      setError(prev => ({ ...prev, assumptions: null }));
      // Optionally download as file
      downloadAssumptionsFile(parsedJson);
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        assumptions: 'Invalid JSON format. Please check your edits and try again.' 
      }));
    }
  };

  // Download assumptions as JSON file
  const downloadAssumptionsFile = (data) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cma_assumptions.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Handle file upload for assumptions
  const handleAssumptionsUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target.result);
        setCmaAssumptions(json);
        setAssumptionsJson(JSON.stringify(json, null, 2));
        setError(prev => ({ ...prev, assumptions: null }));
      } catch (err) {
        setError(prev => ({ 
          ...prev, 
          assumptions: 'Failed to parse JSON file. Please check the file format.' 
        }));
      }
    };
    reader.readAsText(file);
  };

  // Fetch portfolios on component mount
  useEffect(() => {
    fetchPortfolios();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-fetch holdings when portfolio is selected
  useEffect(() => {
    if (selected) {
      fetchHoldings();
    }
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch portfolios list
  const fetchPortfolios = async () => {
    try {
      setLoading(prev => ({ ...prev, portfolios: true }));
      setError(prev => ({ ...prev, portfolios: null }));
      
      const res = await api.get('/portfolios');
      setPortfolios(res.data);
      
      // Select first portfolio by default if none selected
      if (res.data.length > 0 && !selected) {
        setSelected(res.data[0].id);
      }
    } catch (err) {
      console.error('Error fetching portfolios:', err);
      setError(prev => ({ 
        ...prev, 
        portfolios: 'Failed to load portfolios. Please try again.' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, portfolios: false }));
    }
  };

  // Fetch risk metrics
  const fetchRisk = async () => {
  if (!selected) return;
  
  try {
    setLoading(prev => ({ ...prev, risk: true }));
    setError(prev => ({ ...prev, risk: null }));
    
    const res = await api.post('/risk_with_benchmark', {
      portfolio_id: selected,
      start_date: startDate,
      end_date: endDate,
      base_currency: currency,
      benchmark_id: benchmark
    });
    
    setRisk(res.data);
  } catch (err) {
    console.error('Error fetching risk metrics:', err);
    setError(prev => ({ 
      ...prev, 
      risk: 'Failed to calculate risk metrics. Please try again.' 
    }));
  } finally {
    setLoading(prev => ({ ...prev, risk: false }));
  }
};

  // Fetch CMA data
  const fetchCma = async () => {
    if (!selected) return;
    
    try {
      setLoading(prev => ({ ...prev, cma: true }));
      setError(prev => ({ ...prev, cma: null }));
      
      // Send assumptions to the backend
      const res = await api.post('/cma', {
        portfolio_id: selected,
        start_date: startDate,
        end_date: endDate,
        base_currency: currency,
        use_assumptions: true,
        assumptions: cmaAssumptions
      });
      
      setCma(res.data);
    } catch (err) {
      console.error('Error fetching CMA data:', err);
      setError(prev => ({ 
        ...prev, 
        cma: 'Failed to calculate CMA metrics. Please try again.' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, cma: false }));
    }
  };

  // Fetch holdings data
  const fetchHoldings = useCallback(async () => {
    if (!selected) return;
    
    try {
      setLoading(prev => ({ ...prev, holdings: true }));
      setError(prev => ({ ...prev, holdings: null }));
      
      const res = await api.get(`/holdings/${selected}`);
      setHoldings(res.data);
    } catch (err) {
      console.error('Error fetching holdings:', err);
      setError(prev => ({ 
        ...prev, 
        holdings: 'Failed to load holdings. Please try again.' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, holdings: false }));
    }
  }, [selected]);

  // Handle file upload
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !selected) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      setLoading(prev => ({ ...prev, upload: true }));
      setError(prev => ({ ...prev, upload: null }));
      setUploadStatus('Uploading...');
      
      const res = await api.post(`/upload/${selected}`, formData);
      setUploadStatus(`Successfully uploaded ${res.data.rows} rows`);
      
      // Refresh holdings after upload
      await fetchHoldings();
    } catch (err) {
      console.error('Error uploading file:', err);
      setUploadStatus('Upload failed');
      setError(prev => ({ 
        ...prev, 
        upload: 'Failed to upload file. Please check the format and try again.' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  };

  // Handle portfolio selection change
  const handlePortfolioChange = (e) => {
    setSelected(e.target.value);
    // Reset risk and CMA data when portfolio changes
    setRisk(null);
    setCma(null);
  };

  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percentage values
  const formatPercent = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };

  // Render daily VaR & CVaR section
  const DailyVarCVarSection = ({ risk }) => {
    return (
      <div className="metrics-group">
        <h3>Daily VaR & CVaR</h3>
        <div className="risk-cards">
          <RiskCard 
            title="VaR 95%" 
            value={formatCurrency(risk.var_95)} 
            percentage={formatPercent(risk.var_95_pct)} 
          />
          <RiskCard 
            title="VaR 99%" 
            value={formatCurrency(risk.var_99)} 
            percentage={formatPercent(risk.var_99_pct)} 
          />
          <RiskCard 
            title="CVaR 95%" 
            value={formatCurrency(risk.cvar_95)} 
            percentage={formatPercent(risk.cvar_95_pct)} 
          />
          <RiskCard 
            title="CVaR 99%" 
            value={formatCurrency(risk.cvar_99)} 
            percentage={formatPercent(risk.cvar_99_pct)} 
          />
        </div>
      </div>
    );
  };

  // Render annualized VaR & CVaR section
  const AnnualizedVarCVarSection = ({ risk }) => {
    return (
      <div className="metrics-group">
        <h3>Annualized VaR & CVaR</h3>
        <div className="risk-cards">
          <RiskCard 
            title="VaR 95%" 
            value={formatCurrency(risk.var_95_annual)} 
            percentage={formatPercent(risk.var_95_pct_annual)} 
          />
          <RiskCard 
            title="VaR 99%" 
            value={formatCurrency(risk.var_99_annual)} 
            percentage={formatPercent(risk.var_99_pct_annual)} 
          />
          <RiskCard 
            title="CVaR 95%" 
            value={formatCurrency(risk.cvar_95_annual)} 
            percentage={formatPercent(risk.cvar_95_pct_annual)} 
          />
          <RiskCard 
            title="CVaR 99%" 
            value={formatCurrency(risk.cvar_99_annual)} 
            percentage={formatPercent(risk.cvar_99_pct_annual)} 
          />
        </div>
      </div>
    );
  };

  // Render correlation/covariance matrix
  const renderMatrix = (matrix) => {
    if (!matrix || Object.keys(matrix).length === 0) {
      return <p>No matrix data available</p>;
    }
    
    const keys = Object.keys(matrix);
    return (
      <div className="matrix-container">
        <table className="matrix-table">
          <thead>
            <tr>
              <th></th>
              {keys.map(k => <th key={k}>{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {keys.map(row => (
              <tr key={row}>
                <td><strong>{row}</strong></td>
                {keys.map(col => (
                  <td key={col} className={row === col ? 'diagonal-cell' : ''}>
                    {matrix[row][col].toFixed(4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="app-container">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        collapsed={sidebarCollapsed} 
        setCollapsed={setSidebarCollapsed} 
      />
      
      <div className={`main-content ${sidebarCollapsed ? 'main-content-expanded' : ''}`}>
        <Header 
          portfolios={portfolios}
          selected={selected}
          startDate={startDate}
          endDate={endDate}
          currency={currency}
          benchmark={benchmark}
          handlePortfolioChange={handlePortfolioChange}
          setStartDate={setStartDate}
          setEndDate={setEndDate}
          setCurrency={setCurrency}
          setBenchmark={setBenchmark}
          loading={loading}
        />

        <div className="main">
          {activeTab === 'Risk Metrics' && (
            <section className="risk-metrics-section">
              <div className="section-header">
                <h2>Portfolio Risk Analysis</h2>
                <button 
                  onClick={fetchRisk} 
                  disabled={!selected || loading.risk}
                  className="primary-button"
                >
                  {loading.risk ? 'Calculating...' : 'Calculate Risk'}
                </button>
              </div>
              
              {error.risk && <div className="error-message">{error.risk}</div>}
              
              {loading.risk && <LoadingIndicator message="📊 Calculating risk metrics..." />}
              
              {risk && (
                <div className="risk-results">
                  {risk.warning && (
                    <div className="warning-banner">
                      ⚠️ {risk.warning}
                    </div>
                  )}
                  
                  <div className="metrics-container">
                    <DailyVarCVarSection risk={risk} />
                    <AnnualizedVarCVarSection risk={risk} />
                  </div>
                  
                  <div className="risk-description">
                    <h3>Risk Metrics Explanation</h3>
                    <p>
                      <strong>Value at Risk (VaR):</strong> The maximum potential loss over a defined period 
                      at a given confidence level. For example, a 95% VaR represents the loss that will not be 
                      exceeded in 95% of scenarios.
                    </p>
                    <p>
                      <strong>Conditional VaR (CVaR):</strong> Also known as Expected Shortfall, CVaR 
                      represents the expected loss given that the loss exceeds the VaR threshold. It provides 
                      insight into the severity of losses in the worst-case scenarios.
                    </p>
                  </div>
                </div>
              )}
            </section>
          )}

          {activeTab === 'CMA Calculator' && (
            <section className="cma-section">
              <div className="section-header">
                <h2>Capital Market Assumptions</h2>
                <button 
                  onClick={fetchCma} 
                  disabled={!selected || loading.cma}
                  className="primary-button"
                >
                  {loading.cma ? 'Calculating...' : 'Calculate CMA'}
                </button>
              </div>
              
              {error.cma && <div className="error-message">{error.cma}</div>}
              
              {loading.cma && <LoadingIndicator message="Calculating CMA metrics..." />}
              
              {cma && (
                <div className="cma-results">
                  <div className="metrics-container">
                    <h3>Portfolio Metrics</h3>
                    <div className="risk-cards">
                      <RiskCard 
                        title="Arithmetic Return" 
                        value={formatPercent(cma.arithmetic_return)} 
                      />
                      <RiskCard 
                        title="Geometric Return" 
                        value={formatPercent(cma.geometric_return)} 
                      />
                      <RiskCard 
                        title="Volatility" 
                        value={formatPercent(cma.volatility)} 
                      />
                      <RiskCard 
                        title="Prob. Negative" 
                        value={formatPercent(cma.negative_prob)} 
                      />
                      <RiskCard 
                        title="Neg Years in 20" 
                        value={cma.negative_years} 
                      />
                      {cma.sharpe_ratio !== null && (
                        <RiskCard 
                          title="Sharpe Ratio" 
                          value={cma.sharpe_ratio.toFixed(2)} 
                        />
                      )}
                    </div>
                  </div>

                  <div className="asset-stats">
                    <h3>Asset Return & Volatility</h3>
                    <table className="matrix-table">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          <th>Return (%)</th>
                          <th>Volatility (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.keys(cma.asset_returns).map(ticker => (
                          <tr key={ticker}>
                            <td>{ticker}</td>
                            <td className="numeric-cell">{cma.asset_returns[ticker].toFixed(2)}</td>
                            <td className="numeric-cell">{cma.asset_vols[ticker].toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="matrix-section">
                    <h3>Correlation Matrix (Heatmap)</h3>
                    <Plot
                      data={[{
                        z: Object.values(cma.correlation_matrix).map(row => Object.values(row)),
                        x: Object.keys(cma.correlation_matrix),
                        y: Object.keys(cma.correlation_matrix),
                        type: 'heatmap',
                        colorscale: 'RdBu',
                        zmin: -1,
                        zmax: 1,
                        showscale: true,
                        hovertemplate: 'Correlation between %{x} and %{y}: %{z:.4f}<extra></extra>'
                      }]}
                      layout={{
                        width: 700,
                        height: 500,
                        margin: { t: 30, l: 50, r: 50 },
                        font: { size: 12 },
                        paper_bgcolor: '#fff',
                        plot_bgcolor: '#fff'
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        toImageButtonOptions: {
                          format: 'png',
                          filename: 'correlation_matrix'
                        }
                      }}
                    />
                  </div>
                  
                  <div className="matrix-section">
                    <h3>Covariance Matrix</h3>
                    {renderMatrix(cma.covariance_matrix)}
                  </div>
                </div>
              )}
            </section>
          )}

          {activeTab === 'Manual Upload' && (
            <section className="upload-section">
              <h2>Manual Upload</h2>
              <p className="section-description">
                Upload a CSV file containing your portfolio holdings. The file should have at least the following columns:
                <code>ticker</code> and <code>exposure</code>.
              </p>
              
              {!selected && (
                <div className="warning-banner">
                  ⚠️ Please select a portfolio before uploading holdings.
                </div>
              )}
              
              <div className="upload-container">
                <input 
                  type="file" 
                  onChange={handleUpload} 
                  accept=".csv"
                  disabled={!selected || loading.upload}
                  id="file-upload"
                  className="file-input"
                />
                <label htmlFor="file-upload" className="file-label">
                  {loading.upload ? 'Uploading...' : 'Select CSV File'}
                </label>
              </div>
              
              {uploadStatus && (
                <div className={`upload-status ${uploadStatus.includes('failed') ? 'error' : 'success'}`}>
                  {uploadStatus}
                </div>
              )}
              
              {error.upload && <div className="error-message">{error.upload}</div>}
              
              <div className="csv-template">
                <h3>Example CSV Format</h3>
                <pre>
                  ticker,exposure,notes
                  AAPL,10000,Apple Inc.
                  MSFT,15000,Microsoft Corp.
                  GOOGL,12000,Alphabet Inc.
                </pre>
              </div>
            </section>
          )}

          {activeTab === 'Holdings' && (
            <section className="holdings-section">
              <div className="section-header">
                <h2>Portfolio Holdings</h2>
                <button 
                  onClick={fetchHoldings}
                  disabled={!selected || loading.holdings}
                  className="secondary-button"
                >
                  {loading.holdings ? 'Loading...' : 'Refresh Holdings'}
                </button>
              </div>
              
              {error.holdings && <div className="error-message">{error.holdings}</div>}
              
              {loading.holdings && <LoadingIndicator message="Loading holdings data..." />}
              
              {holdings.length > 0 ? (
                <>
                  <div className="holdings-chart">
                    <Plot
                      data={[{
                        type: 'pie',
                        labels: holdings.map(h => `${h.ticker} (${h.name})`),
                        values: holdings.map(h => h.exposure),
                        textinfo: 'label+percent',
                        hoverinfo: 'label+value+percent',
                        hovertemplate: '%{label}<br>Exposure: ' + currency + ' %{value:.2f}<br>%{percent}<extra></extra>'
                      }]}
                      layout={{
                        title: 'Holdings Breakdown',
                        showlegend: true,
                        legend: {
                          orientation: "h",
                          xanchor: "center",
                          yanchor: "bottom",
                          y: -0.2,
                          x: 0.5
                        },
                        margin: { t: 40, b: 40, l: 20, r: 20 },
                        paper_bgcolor: '#fff',
                        plot_bgcolor: '#fff',
                        font: { color: '#1e293b' }
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        toImageButtonOptions: {
                          format: 'png',
                          filename: 'holdings_breakdown'
                        }
                      }}
                      style={{ width: '100%', height: '400px' }}
                    />
                  </div>
                  
                  <div className="holdings-table">
                    <table className="matrix-table">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          <th>Name</th>
                          <th>Exposure ({currency})</th>
                          <th>% of Portfolio</th>
                        </tr>
                      </thead>
                      <tbody>
                        {holdings.map((row, idx) => (
                          <tr key={idx}>
                            <td className="ticker-cell">{row.ticker}</td>
                            <td>{row.name}</td>
                            <td className="numeric-cell">{formatCurrency(row.exposure)}</td>
                            <td className="numeric-cell">{formatPercent(row.pct_of_portfolio)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  {selected ? (
                    <p>No holdings found for this portfolio. Please upload holdings data.</p>
                  ) : (
                    <p>Please select a portfolio to view holdings.</p>
                  )}
                </div>
              )}
            </section>
          )}

          {activeTab === 'CMA Assumptions' && (
            <section className="cma-assumptions-section">
              <div className="section-header">
                <h2>Capital Market Assumptions</h2>
                <div className="button-group">
                  {editingAssumptions ? (
                    <>
                      <button 
                        onClick={handleSaveAssumptions}
                        className="primary-button"
                      >
                        Save Changes
                      </button>
                      <button 
                        onClick={() => setEditingAssumptions(false)}
                        className="secondary-button"
                        style={{ marginLeft: '10px' }}
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <button 
                        onClick={() => setEditingAssumptions(true)}
                        className="primary-button"
                      >
                        Edit JSON
                      </button>
                    </>
                  )}
                </div>
              </div>
              
              {error.assumptions && <div className="error-message">{error.assumptions}</div>}
              
              <div className="infobox">
                <div className="infobox-title">About Capital Market Assumptions</div>
                <div className="infobox-content">
                  These long-term capital market assumptions represent forward-looking estimates of expected returns 
                  and volatility for various asset classes. These assumptions are used in the CMA Calculator 
                  instead of historical data to provide a more forward-looking analysis.
                </div>
              </div>
              
              {editingAssumptions ? (
                <div className="edit-assumptions">
                  <p>Edit the JSON below to update the assumptions. You can also upload a JSON file.</p>
                  
                  <div className="upload-container">
                    <input 
                      type="file" 
                      onChange={handleAssumptionsUpload} 
                      accept=".json"
                      id="assumptions-upload"
                      className="file-input"
                    />
                    <label htmlFor="assumptions-upload" className="file-label">
                      Upload JSON
                    </label>
                  </div>
                  
                  <div className="json-editor">
                    <textarea
                      value={assumptionsJson}
                      onChange={(e) => setAssumptionsJson(e.target.value)}
                      rows={25}
                      style={{
                        width: '100%',
                        fontFamily: 'monospace',
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid #e6edf5'
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="view-assumptions">
                  <div className="assumptions-actions">
<button 
                      onClick={() => downloadAssumptionsFile(cmaAssumptions)}
                      className="secondary-button"
                      style={{ marginRight: '10px' }}
                    >
                      Download Assumptions File
                    </button>
                    <div className="upload-container" style={{ display: 'inline-block' }}>
                      <input 
                        type="file" 
                        onChange={handleAssumptionsUpload} 
                        accept=".json"
                        id="assumptions-upload-file"
                        className="file-input"
                      />
                      <label htmlFor="assumptions-upload-file" className="file-label">
                        Upload Assumptions File
                      </label>
                    </div>
                  </div>
                  
                  <h3>Asset Class Assumptions</h3>
                  <div className="table-container">
                    <table className="matrix-table">
                      <thead>
                        <tr>
                          <th>Asset Class</th>
                          <th>Expected Return (%)</th>
                          <th>Volatility (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(cmaAssumptions.assetClasses).map(([assetClass, data]) => (
                          <tr key={assetClass}>
                            <td>{assetClass}</td>
                            <td className="numeric-cell">{data.expectedReturn.toFixed(2)}</td>
                            <td className="numeric-cell">{data.volatility.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  <h3>Ticker to Asset Class Mappings</h3>
                  <div className="mappings-grid">
                    {Object.entries(cmaAssumptions.mappings).map(([ticker, assetClass]) => (
                      <div key={ticker} className="mapping-item">
                        <span className="mapping-ticker">{ticker}</span>
                        <span className="mapping-arrow">➝</span>
                        <span className="mapping-asset">{assetClass}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}
        </div>
        
        <footer>
          <p>Portfolio Risk Analysis Dashboard • {new Date().getFullYear()}</p>
        </footer>
      </div>
    </div>
  );
}

export default App;