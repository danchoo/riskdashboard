// src/components/shared/MatrixTable.js
import React from 'react';

const MatrixTable = ({ matrix }) => {
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

export default MatrixTable;