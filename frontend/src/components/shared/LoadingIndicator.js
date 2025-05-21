// src/components/shared/LoadingIndicator.js
import React from 'react';

const LoadingIndicator = ({ message = 'Loading...' }) => {
  return (
    <div className="loading-indicator">
      <p>{message}</p>
    </div>
  );
};

export default LoadingIndicator;