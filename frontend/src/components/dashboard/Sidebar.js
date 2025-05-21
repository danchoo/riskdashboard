// src/components/dashboard/Sidebar.js
import React from 'react';
import { 
  MdInsights, 
  MdPieChart, 
  MdUpload, 
  MdCalculate, 
  MdSettings, 
  MdChevronLeft, 
  MdChevronRight 
} from 'react-icons/md';

// If you don't have react-icons installed, you can use these emoji as fallbacks:
// const icons = {
//   'Risk Metrics': '📊',
//   'Holdings': '📈',
//   'Manual Upload': '📤',
//   'CMA Calculator': '🧮',
//   'CMA Assumptions': '⚙️',
// };

const Sidebar = ({ activeTab, setActiveTab, collapsed, setCollapsed }) => {
  const toggleSidebar = () => {
    setCollapsed(!collapsed);
  };
  
  // Navigation items with icons
  const navItems = [
    { id: 'Risk Metrics', icon: <MdInsights />, label: 'Risk Metrics' },
    { id: 'Holdings', icon: <MdPieChart />, label: 'Holdings' },
    { id: 'Manual Upload', icon: <MdUpload />, label: 'Manual Upload' },
    { id: 'CMA Calculator', icon: <MdCalculate />, label: 'CMA Calculator' },
    { id: 'CMA Assumptions', icon: <MdSettings />, label: 'CMA Assumptions' }
  ];
  
  return (
    <div className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="sidebar-logo">
        {!collapsed ? (
          <div className="full-logo">
            <h2>BetaShares</h2>
            <div className="logo-subtitle">Risk Dashboard</div>
          </div>
        ) : (
          <div className="mini-logo">BS</div>
        )}
        <button className="toggle-button" onClick={toggleSidebar}>
          {collapsed ? <MdChevronRight size={20} /> : <MdChevronLeft size={20} />}
        </button>
      </div>
      
      <ul className="nav-list">
        {navItems.map(item => (
          <li key={item.id} className="nav-item">
            <div 
              className={`nav-link ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span className="nav-link-icon">{item.icon}</span>
              <span className="nav-link-text">{item.label}</span>
            </div>
            {collapsed && <div className="nav-tooltip">{item.label}</div>}
          </li>
        ))}
      </ul>
      
      <div className="sidebar-footer">
        {!collapsed && <div className="version">v1.0.0</div>}
      </div>
    </div>
  );
};

export default Sidebar;