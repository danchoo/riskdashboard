// src/components/shared/RiskCard.js
import React from 'react';

const RiskCard = ({ title, value, percentage }) => {
  return (
    <div className="risk-card">
      <h4>{title}</h4>
      <div className="risk-value">{value}</div>
      {percentage && <div className="risk-percent">{percentage}</div>}
    </div>
  );
};

export default RiskCard;