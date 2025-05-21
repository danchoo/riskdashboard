export const benchmarks = [
  {
    id: "asx200",
    name: "S&P/ASX 200",
    description: "Australian stock market benchmark",
    color: "#0066CC"
  },
  {
    id: "msci_world",
    name: "MSCI World Index",
    description: "Global developed markets benchmark",
    color: "#009933"
  },
  {
    id: "balanced_index",
    name: "70/30 Growth Index",
    description: "Composite index (70% equity, 30% bonds)",
    color: "#FF6600"
  }
];


export const dhhfBenchmark = {
  id: "dhhf_benchmark",
  name: "DHHF Target Allocation",
  description: "Custom benchmark based on DHHF's target asset allocation",
  composition: {
    "ASX200": 0.37,  // 37% Australian equities
    "MSCI World": 0.40,  // 40% Developed markets 
    "MSCI EM": 0.10,  // 10% Emerging markets
    "ASX Property": 0.13  // 13% Property/other
  },
  color: "#7030A0"  // Purple color
};