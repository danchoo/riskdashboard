// components/shared/RiskCard.js
import React from 'react';

const RiskCard = ({ title, value, percentage }) => {
  // Check if values are valid
  const displayValue = value || 'N/A';
  const displayPercentage = percentage || '';

  return (
    <div className="risk-card">
      <h4>{title}</h4>
      <div className="risk-value">{displayValue}</div>
      {displayPercentage && <div className="risk-percent">{displayPercentage}</div>}
    </div>
  );
};

export default RiskCard;