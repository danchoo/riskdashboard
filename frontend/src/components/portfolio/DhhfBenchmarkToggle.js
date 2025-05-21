import React from 'react';
import { dhhfBenchmark } from '../../utils/constants';

const DhhfBenchmarkToggle = ({ portfolio, benchmark, setBenchmark }) => {
  // Only show for DHHF portfolio
  if (portfolio?.id !== "dhhf") {
    return null;
  }
  
  const handleToggle = () => {
    if (benchmark === dhhfBenchmark.id) {
      setBenchmark("");  // Turn off benchmark
    } else {
      setBenchmark(dhhfBenchmark.id);  // Set to DHHF benchmark
    }
  };
  
  const isActive = benchmark === dhhfBenchmark.id;
  
  return (
    <div className="dhhf-benchmark-toggle">
      <button 
        className={`toggle-button ${isActive ? 'active' : ''}`}
        onClick={handleToggle}
      >
        {isActive ? 'Disable DHHF Benchmark' : 'Compare to DHHF Target Allocation'}
      </button>
      
      <div className="benchmark-tooltip">
        Using this benchmark will compare your current portfolio to DHHF's target allocation
        of 37% Australian equities, 40% developed markets, 10% emerging markets, and 13% other assets.
      </div>
    </div>
  );
};

export default DhhfBenchmarkToggle;