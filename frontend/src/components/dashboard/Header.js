import React from 'react';

const Header = ({ 
  portfolios, 
  selected, 
  startDate, 
  endDate, 
  currency, 
  handlePortfolioChange, 
  setStartDate, 
  setEndDate, 
  setCurrency, 
  loading 
}) => {
  return (
    <header>
      <h1>Multi-Asset Risk Dashboard</h1>
      <div className="controls">
        <div className="control-group">
          <label>Portfolio:</label>
          <select 
            onChange={handlePortfolioChange} 
            value={selected}
            disabled={loading.portfolios}
          >
            <option value="">Select Portfolio</option>
            {portfolios.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}
                {p.description ? ` - ${p.description}` : ''}
              </option>
            ))}
          </select>
        </div>
        
        <div className="control-group">
          <label>Date Range:</label>
          <input 
            type="date" 
            value={startDate} 
            onChange={e => setStartDate(e.target.value)} 
          />
          <span>to</span>
          <input 
            type="date" 
            value={endDate} 
            onChange={e => setEndDate(e.target.value)} 
          />
        </div>
        
        <div className="control-group">
          <label>Currency:</label>
          <select 
            onChange={e => setCurrency(e.target.value)} 
            value={currency}
          >
            <option value="AUD">AUD</option>
            <option value="USD">USD</option>
          </select>
        </div>
      </div>
    </header>
  );
};

export default Header;