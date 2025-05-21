import React from 'react';
import DailyVarCVarSection from './DailyVarCVarSection';
import AnnualizedVarCVarSection from './AnnualizedVarCVarSection';
import BenchmarkComparison from './BenchmarkComparison';
import LoadingIndicator from '../shared/LoadingIndicator';

const RiskMetricsSection = ({ 
  selected, 
  fetchRisk, 
  loading, 
  error, 
  risk, 
  formatCurrency, 
  formatPercent 
}) => {
  return (
    <section className="risk-metrics-section">
      {/* Existing code */}
      
      {risk && (
        <div className="risk-results">
          {/* Existing metrics */}
          
          {/* Add benchmark comparison if available */}
          {risk.benchmark && (
            <BenchmarkComparison 
              risk={risk} 
              formatPercent={formatPercent} 
            />
          )}
          
          {/* Existing description */}
        </div>
      )}
    </section>
  );
};

export default RiskMetricsSection;