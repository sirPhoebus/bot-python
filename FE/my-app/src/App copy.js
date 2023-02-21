import './App.css';
import React, { useEffect, useState } from 'react';

function App() {
  const [selectedTf, setSelectedTf] = useState("1h");
  const [whiteSoldiersData, setWhiteSoldiersData] = useState([]);
  const [russianDollData, setRussianDollData] = useState([]);
  const [bbTrendData, setBbTrendData] = useState([]);
  const [vwapData, setVwapData] = useState([]);
  const [atrData, setAtrData] = useState([]);
  const [aggVolData, setAggVolData] = useState([]);
  const [globalData, setGlobalData] = useState([]);
  const [tableData, setTableData] = useState([]);

  useEffect(() => {
    const requestOptions = {
      method: 'GET',
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Origin, X-Requested',
        'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
      }
    };
  
    fetch(`http://localhost:8000/white_soldiers?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setWhiteSoldiersData(data.data));
  
    fetch(`http://localhost:8000/russian_doll?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setRussianDollData(data.data));
  
    fetch(`http://localhost:8000/bb_trend?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setBbTrendData(data.data));
  
    fetch(`http://localhost:8000/vwap?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setVwapData(data.data));
  
    fetch(`http://localhost:8000/atr?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setAtrData(data.data));
  
    fetch(`http://localhost:8000/ob?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setTableData(data.data));
  
    fetch('http://localhost:8000/agg_vol', requestOptions)
      .then(response => response.json())
      .then(data => setAggVolData(data.data));
  
    fetch('http://localhost:8000/global', requestOptions)
      .then(response => response.json())
      .then(data => setGlobalData(data.data));
  }, [selectedTf]);

  const handleTfChange = (event) => {
    setSelectedTf(event.target.value);
  }

  const tfOptions = ["1d", "6h", "4h", "2h", "1h", "30min", "15min", "5min", "1min"];

  return (
    <div className="App">
      <header className="App-header">
        <div>
          <select id="tf-select" value={selectedTf} onChange={handleTfChange}>
            {tfOptions.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </div>
        <div>
        whiteSoldiersData: {JSON.stringify(whiteSoldiersData)}
        </div>
        <div>
        russianDollData: {JSON.stringify(russianDollData)}
        </div>
        <div>
        bbTrendData: {JSON.stringify(bbTrendData)}
        </div>
        <div>
        vwapData: {JSON.stringify(vwapData)}
        </div>
        <div>
        atrData: {JSON.stringify(atrData)}
        </div>
        <div>
          <label>Max Bids: {JSON.stringify(tableData)}</label>
          <br />
          <label>Min Bids: {JSON.stringify(tableData)}</label>
        </div>
        <div>
        aggVolData: {JSON.stringify(aggVolData)}
        </div>
        <div>
        globalData: {JSON.stringify(globalData)}
        </div>
      </header>
    </div>
  );
}

export default App;